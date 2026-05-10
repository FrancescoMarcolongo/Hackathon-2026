"""Pipeline v2 Step 5: BLT-lite driven by the physical-coordinate optimizer."""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from typing import Dict

import numpy as np

from blt_bayesian_optimizer import (
    BLTConfig,
    BLTRewardConfig,
    BLT_JACOBIAN_UPDATE,
    BLT_REWARD_REGION,
    EPS,
    GateThresholds,
    _blt_row,
    _history_row,
    evaluate_blt_lite,
    gate1_cat_feasible,
    gate2_blt_valid,
    propose_blt_jacobian_update,
    reward_blt,
)
from physical_coordinates_reward import (
    ImprovedRewardConfig,
    PhysicalBounds,
    select_better_physical,
)
from two_points_with_noise import (
    BIAS_TOL_REL,
    OPTIMIZATION_START_X,
    TARGET_BIAS,
    NoiseConfig,
    TwoPointConfig,
    cleanup_local_qutip_cache,
    clear_measure_cache,
)


PHYSICAL_UPDATE = "PHYSICAL_UPDATE"


class PhysicalPathAdapter:
    """Small adapter exposing normalized physical coordinates to BLT Jacobian."""

    def __init__(self, bounds: PhysicalBounds, v0: np.ndarray, trust_region_length: float = 0.34) -> None:
        self.bounds = bounds
        self.bounds_array = bounds.as_array()
        self.v0 = np.asarray(v0, dtype=float)
        self.center = self.v0.copy()
        self.trust_region_length = float(trust_region_length)

    def _to_unit(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        low = self.bounds_array[:, 0]
        high = self.bounds_array[:, 1]
        u = np.empty(4, dtype=float)
        u[:2] = (v[:2] - low[:2]) / (high[:2] - low[:2])
        wrapped = (v[2:] + np.pi) % (2.0 * np.pi) - np.pi
        u[2:] = (wrapped - low[2:]) / (high[2:] - low[2:])
        return np.clip(u, 0.0, 1.0)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        low = self.bounds_array[:, 0]
        high = self.bounds_array[:, 1]
        v = low + u * (high - low)
        v[2:] = (v[2:] + np.pi) % (2.0 * np.pi) - np.pi
        return v

    def update_center(self, v: np.ndarray, improved: bool) -> None:
        self.center = np.asarray(v, dtype=float).copy()
        if improved:
            self.trust_region_length = min(0.55, self.trust_region_length * 1.08)
        else:
            self.trust_region_length = max(0.10, self.trust_region_length / 1.08)


def _physical_result_from_history(row: dict) -> dict:
    u = np.asarray(
        [
            float(row["incumbent_g2_real"]),
            float(row["incumbent_g2_imag"]),
            float(row["incumbent_eps_d_real"]),
            float(row["incumbent_eps_d_imag"]),
        ],
        dtype=float,
    )
    v = np.asarray(
        [
            float(row["incumbent_log_kappa2"]),
            float(row["incumbent_log_abs_alpha"]),
            float(row["incumbent_theta_alpha"]),
            float(row["incumbent_theta_g"]),
        ],
        dtype=float,
    )
    bias = float(row["incumbent_bias"])
    bias_error = abs(math.log(max(bias, EPS) / TARGET_BIAS))
    bias_rel_error = abs(bias / TARGET_BIAS - 1.0)
    return {
        "v": v,
        "x": u,
        "u": u,
        "reward": float(row["incumbent_reward"]),
        "reward_safe": float(row.get("incumbent_reward_safe", row["incumbent_reward"])),
        "loss_to_minimize": -float(row.get("incumbent_reward_safe", row["incumbent_reward"])),
        "is_feasible": bool(bias_rel_error <= BIAS_TOL_REL),
        "bias_error": float(bias_error),
        "bias_rel_error": float(bias_rel_error),
        "T_X": float(row["incumbent_T_X"]),
        "T_Z": float(row["incumbent_T_Z"]),
        "bias": bias,
        "geo_lifetime": float(row["incumbent_geo_lifetime"]),
        "update_type": PHYSICAL_UPDATE,
        "gate1_pass": 0,
        "gate2_pass": 0,
        "gate3_pass": 0,
        "N_markov": np.nan,
        "leakage_proxy": np.nan,
        "contrast_x": np.nan,
        "contrast_z": np.nan,
        "gamma_X": 1.0 / float(row["incumbent_T_X"]),
        "gamma_Z": 1.0 / float(row["incumbent_T_Z"]),
        "cost_units": 2.0,
        "cumulative_cost_units": np.nan,
        "trust_region_length": np.nan,
        "reason": "physical_coordinate_fallback_path",
    }


def _best_blt_state_from_observed(observed: dict) -> dict:
    return {
        "v": np.asarray(observed["v"], dtype=float),
        "u": np.asarray(observed["u"], dtype=float),
        "e_bias": math.log(max(float(observed["bias"]), EPS) / TARGET_BIAS),
        "log_gamma_product": math.log(max(float(observed["gamma_X"]) * float(observed["gamma_Z"]), EPS)),
        "N_markov": float(observed["N_markov"]),
        "leakage_proxy": float(observed["leakage_proxy"]),
        "valid_numerics": True,
        "T_X": float(observed["T_X"]),
        "T_Z": float(observed["T_Z"]),
        "bias": float(observed["bias"]),
        "geo_lifetime": float(observed["geo_lifetime"]),
    }


def _select_for_blt_physical(candidate: dict, incumbent: dict) -> bool:
    c_feasible = bool(candidate["is_feasible"])
    i_feasible = bool(incumbent["is_feasible"])
    if c_feasible or i_feasible:
        return select_better_physical(candidate, incumbent)
    c_error = float(candidate["bias_error"])
    i_error = float(incumbent["bias_error"])
    if c_error < 0.98 * i_error:
        return True
    if c_error <= 1.03 * i_error:
        return float(candidate["reward_safe"]) > float(incumbent["reward_safe"])
    return False


def run_blt_physical_based_optimizer(
    physical_history: list[dict],
    *,
    verbose: bool = True,
    blt_cfg: BLTConfig | None = None,
    reward_cfg: ImprovedRewardConfig | None = None,
    bounds: PhysicalBounds | None = None,
    thresholds: GateThresholds | None = None,
    weights: BLTRewardConfig | None = None,
) -> dict:
    blt_cfg = blt_cfg or BLTConfig(jacobian_start_epoch=8, jacobian_every=1)
    reward_cfg = reward_cfg or ImprovedRewardConfig()
    bounds = bounds or PhysicalBounds()
    thresholds = thresholds or GateThresholds()
    weights = weights or BLTRewardConfig(w_abs=0.32)
    sim_cfg = TwoPointConfig()
    noise_cfg = NoiseConfig(sigma=blt_cfg.noise_sigma, seed=blt_cfg.noise_seed)
    noise_rng = np.random.default_rng(blt_cfg.noise_seed + 113)
    clear_measure_cache()

    v0 = np.asarray(
        [
            float(physical_history[0]["incumbent_log_kappa2"]),
            float(physical_history[0]["incumbent_log_abs_alpha"]),
            float(physical_history[0]["incumbent_theta_alpha"]),
            float(physical_history[0]["incumbent_theta_g"]),
        ],
        dtype=float,
    )
    adapter = PhysicalPathAdapter(bounds, v0, trust_region_length=0.34)
    run_id = f"pipeline_v2_blt_physical_seed{blt_cfg.random_seed}"
    history: list[dict] = []
    cumulative_cost = 0.0

    start_eval = _physical_result_from_history(physical_history[0])
    cumulative_cost += 2.0
    start_eval["cumulative_cost_units"] = cumulative_cost
    start_eval["trust_region_length"] = adapter.trust_region_length
    incumbent = dict(start_eval)
    history.append(_history_row(run_id, 0, start_eval, incumbent))
    best_blt_state: dict | None = None
    start_time = time.time()

    max_epoch = min(blt_cfg.max_epochs, len(physical_history) - 1)
    for epoch in range(1, max_epoch + 1):
        observed: dict | None = None
        if (
            best_blt_state is not None
            and epoch >= blt_cfg.jacobian_start_epoch
            and epoch % blt_cfg.jacobian_every == 0
            and (
                not bool(incumbent.get("is_feasible", False))
                or epoch <= blt_cfg.jacobian_start_epoch + 3
            )
            and abs(float(best_blt_state.get("e_bias", np.inf))) <= thresholds.max_bias_abs_for_jacobian
        ):
            candidate_v, jac_blt, jac_reward, ok_jac = propose_blt_jacobian_update(
                np.asarray(best_blt_state["v"], dtype=float),
                best_blt_state,
                optimizer=adapter,
                sim_cfg=sim_cfg,
                noise_cfg=noise_cfg,
                bounds=bounds,
                rng=noise_rng,
                use_alpha_correction=blt_cfg.use_alpha_correction,
                thresholds=thresholds,
                weights=weights,
                config=blt_cfg,
            )
            cumulative_cost += blt_cfg.jacobian_extra_cost_units
            if ok_jac and candidate_v is not None and jac_blt is not None and jac_reward is not None:
                cumulative_cost += blt_cfg.blt_cost_units
                observed = _blt_row(
                    jac_blt,
                    jac_reward,
                    BLT_JACOBIAN_UPDATE,
                    True,
                    True,
                    True,
                    cost_units=blt_cfg.blt_cost_units + blt_cfg.jacobian_extra_cost_units,
                    cumulative_cost_units=cumulative_cost,
                    trust_region_length=adapter.trust_region_length,
                )

        if observed is None:
            physical_candidate = _physical_result_from_history(physical_history[epoch])
            candidate_v = np.asarray(physical_candidate["v"], dtype=float)
            blt = evaluate_blt_lite(
                candidate_v,
                sim_cfg=sim_cfg,
                noise_cfg=noise_cfg,
                bounds=bounds,
                rng=noise_rng,
                use_alpha_correction=blt_cfg.use_alpha_correction,
            )
            cumulative_cost += blt_cfg.blt_cost_units
            gate1, reason1 = gate1_cat_feasible(blt, thresholds, sim_cfg)
            gate2, reason2 = gate2_blt_valid(blt, thresholds)
            if gate1 and gate2:
                reward = reward_blt(blt, weights)
                observed = _blt_row(
                    blt,
                    reward,
                    BLT_REWARD_REGION,
                    True,
                    True,
                    False,
                    cost_units=blt_cfg.blt_cost_units,
                    cumulative_cost_units=cumulative_cost,
                    trust_region_length=adapter.trust_region_length,
                )
            else:
                physical_candidate["reason"] = reason1 or reason2
                physical_candidate["cost_units"] = 2.0
                physical_candidate["cumulative_cost_units"] = cumulative_cost
                physical_candidate["trust_region_length"] = adapter.trust_region_length
                observed = physical_candidate

        improved = _select_for_blt_physical(observed, incumbent)
        if observed["update_type"] in (BLT_REWARD_REGION, BLT_JACOBIAN_UPDATE):
            best_blt_state = _best_blt_state_from_observed(observed)
        if improved:
            incumbent = dict(observed)
        adapter.update_center(np.asarray(incumbent["v"], dtype=float), improved)
        row = _history_row(run_id, epoch, observed, incumbent)
        row["trust_region_length"] = float(adapter.trust_region_length)
        history.append(row)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == max_epoch):
            elapsed = time.time() - start_time
            print(
                f"{run_id} epoch={epoch:03d}/{max_epoch} "
                f"type={observed['update_type']} "
                f"incumbent bias={float(incumbent['bias']):.4g} "
                f"reward={float(incumbent['reward']):.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    cleanup_local_qutip_cache()
    return {
        "history": history,
        "incumbent": incumbent,
        "v0": v0,
        "x0": OPTIMIZATION_START_X.copy(),
        "sim_config": asdict(sim_cfg),
        "noise_config": asdict(noise_cfg),
        "blt_config": asdict(blt_cfg),
        "reward_config": asdict(weights),
        "gate_thresholds": asdict(thresholds),
        "physical_bounds": asdict(bounds),
    }
