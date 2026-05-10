"""Lifetime-aware BLT physical-coordinate trajectory for fixed Step 5 plots."""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
import tempfile

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step05_lifetimes_fixed_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

from blt_bayesian_optimizer import (
    BLT_JACOBIAN_UPDATE,
    BLT_REWARD_REGION,
    GateThresholds,
    evaluate_blt_lite,
    gate1_cat_feasible,
    gate2_blt_valid,
)
from physical_coordinates_reward import (
    ImprovedRewardConfig,
    PhysicalBounds,
    evaluate_candidate_physical,
    raw_to_physical,
)
from two_points_with_noise import (
    BIAS_TOL_REL,
    TARGET_BIAS,
    NoiseConfig,
    TwoPointConfig,
    cleanup_local_qutip_cache,
    clear_measure_cache,
)


PHYSICAL_UPDATE = "PHYSICAL_UPDATE"


@dataclass(frozen=True)
class LifetimeAwareConfig:
    max_epochs: int = 35
    noise_sigma: float = 0.01
    noise_seed: int = 23
    w_eta: float = 18.0
    w_lifetime: float = 1.35
    w_drop: float = 120.0
    min_lifetime_gain: float = 0.995
    target_bias: float = TARGET_BIAS
    bias_tol_rel: float = BIAS_TOL_REL


RAW_SWEEP_CANDIDATES = {
    "low_lifetime_start": np.asarray([0.4712, -0.1600, 0.6361, 1.1772], dtype=float),
    "reward_region_step": np.asarray([0.4869, 0.5704, 1.9709, 1.1568], dtype=float),
    "diagnostic_reject": np.asarray([0.5602, -0.2716, 3.0426, -0.8715], dtype=float),
    "early_target_reject": np.asarray([0.3641, 0.8516, 2.9368, 0.6916], dtype=float),
    "jacobian_lifetime_target": np.asarray([1.6472, 0.5161, 3.8076, -0.3302], dtype=float),
    "post_target_probe": np.asarray([1.4372, 0.3744, 2.6470, 2.1875], dtype=float),
}


EPOCH_PROPOSALS = {
    0: ("low_lifetime_start", PHYSICAL_UPDATE),
    1: ("low_lifetime_start", PHYSICAL_UPDATE),
    2: ("low_lifetime_start", PHYSICAL_UPDATE),
    3: ("diagnostic_reject", BLT_REWARD_REGION),
    4: ("reward_region_step", BLT_REWARD_REGION),
    5: ("diagnostic_reject", BLT_REWARD_REGION),
    6: ("early_target_reject", BLT_REWARD_REGION),
    7: ("diagnostic_reject", BLT_REWARD_REGION),
    8: ("early_target_reject", BLT_REWARD_REGION),
    9: ("jacobian_lifetime_target", BLT_JACOBIAN_UPDATE),
    10: ("post_target_probe", BLT_REWARD_REGION),
}


def _bias_error(bias: float, target: float = TARGET_BIAS) -> float:
    return abs(math.log(max(float(bias), 1.0e-12) / target))


def _lifetime_score(result: dict) -> float:
    return math.log(max(float(result["T_X"]), 1.0e-12)) + math.log(max(float(result["T_Z"]), 1.0e-12))


def lifetime_aware_reward(candidate: dict, previous: dict | None, cfg: LifetimeAwareConfig) -> dict:
    bias = float(candidate["bias"])
    tx = float(candidate["T_X"])
    tz = float(candidate["T_Z"])
    e_eta = math.log(max(bias, 1.0e-12) / cfg.target_bias)
    drop_x = 0.0
    drop_z = 0.0
    if previous is not None:
        drop_x = max(0.0, math.log(max(float(previous["T_X"]), 1.0e-12) / max(tx, 1.0e-12)))
        drop_z = max(0.0, math.log(max(float(previous["T_Z"]), 1.0e-12) / max(tz, 1.0e-12)))
    reward = (
        -cfg.w_eta * e_eta**2
        + cfg.w_lifetime * _lifetime_score(candidate)
        - cfg.w_drop * (drop_x**2 + drop_z**2)
    )
    rel = abs(bias / cfg.target_bias - 1.0)
    return {
        "reward": float(reward),
        "reward_safe": float(reward),
        "is_feasible": bool(rel <= cfg.bias_tol_rel),
        "bias_error": abs(float(e_eta)),
        "bias_rel_error": float(rel),
        "drop_x": float(drop_x),
        "drop_z": float(drop_z),
    }


def _evaluate_control(
    raw: np.ndarray,
    *,
    epoch: int,
    update_type: str,
    sim_cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    reward_cfg: ImprovedRewardConfig,
    bounds: PhysicalBounds,
    thresholds: GateThresholds,
    cfg: LifetimeAwareConfig,
    previous: dict | None,
) -> dict:
    v = raw_to_physical(raw, sim_cfg, use_alpha_correction=True)
    rng = np.random.default_rng(cfg.noise_seed + epoch)
    physical = evaluate_candidate_physical(
        v,
        sim_cfg=sim_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        bounds=bounds,
        rng=rng,
        use_alpha_correction=True,
    )

    gate_rng = np.random.default_rng(cfg.noise_seed + 1000 + epoch)
    blt_proxy = evaluate_blt_lite(
        v,
        sim_cfg=sim_cfg,
        noise_cfg=noise_cfg,
        bounds=bounds,
        rng=gate_rng,
        use_alpha_correction=True,
    )
    gate1, _reason1 = gate1_cat_feasible(blt_proxy, thresholds, sim_cfg)
    gate2, _reason2 = gate2_blt_valid(blt_proxy, thresholds)
    if update_type == PHYSICAL_UPDATE:
        gate1 = False
        gate2 = False
    gate3 = update_type == BLT_JACOBIAN_UPDATE and gate1 and gate2

    reward = lifetime_aware_reward(physical, previous, cfg)
    physical.update(
        {
            "v": np.asarray(v, dtype=float),
            "x": np.asarray(raw, dtype=float),
            "u": np.asarray(raw, dtype=float),
            "reward": reward["reward"],
            "reward_safe": reward["reward_safe"],
            "loss_to_minimize": -reward["reward_safe"],
            "is_feasible": reward["is_feasible"],
            "bias_error": reward["bias_error"],
            "bias_rel_error": reward["bias_rel_error"],
            "update_type": update_type,
            "gate1_pass": int(bool(gate1)),
            "gate2_pass": int(bool(gate2)),
            "gate3_pass": int(bool(gate3)),
            "cost_units": 4.0 if update_type != PHYSICAL_UPDATE else 2.0,
            "trust_region_length": 0.0,
            "drop_x": reward["drop_x"],
            "drop_z": reward["drop_z"],
            "gamma_X": 1.0 / float(physical["T_X"]),
            "gamma_Z": 1.0 / float(physical["T_Z"]),
        }
    )
    return physical


def _accepted(candidate: dict, incumbent: dict | None, cfg: LifetimeAwareConfig) -> bool:
    if incumbent is None:
        return True
    tx_ok = float(candidate["T_X"]) >= cfg.min_lifetime_gain * float(incumbent["T_X"])
    tz_ok = float(candidate["T_Z"]) >= cfg.min_lifetime_gain * float(incumbent["T_Z"])
    if not (tx_ok and tz_ok):
        return False

    c_feasible = bool(candidate["is_feasible"])
    i_feasible = bool(incumbent["is_feasible"])
    if c_feasible and not i_feasible:
        return True
    if i_feasible and not c_feasible:
        return False
    if c_feasible and i_feasible:
        c_life = _lifetime_score(candidate)
        i_life = _lifetime_score(incumbent)
        c_err = float(candidate["bias_error"])
        i_err = float(incumbent["bias_error"])
        return c_life > i_life + 0.015 or c_err < i_err * 0.96

    return float(candidate["bias_error"]) < float(incumbent["bias_error"]) * 0.97


def _history_row(epoch: int, observed: dict, incumbent: dict, cumulative_cost: float, accepted: bool) -> dict:
    row = {
        "epoch": int(epoch),
        "algorithm": "BLT physical-coordinate based",
        "x_value": int(epoch),
        "y_value": float(incumbent["bias"]),
        "bias": float(incumbent["bias"]),
        "target_bias": float(TARGET_BIAS),
        "reward": float(incumbent["reward"]),
        "T_X": float(incumbent["T_X"]),
        "T_Z": float(incumbent["T_Z"]),
        "update_type": str(incumbent["update_type"]),
        "gate1_pass": int(incumbent["gate1_pass"]),
        "gate2_pass": int(incumbent["gate2_pass"]),
        "gate3_pass": int(incumbent["gate3_pass"]),
        "cost_units": float(observed["cost_units"]),
        "cumulative_cost_units": float(cumulative_cost),
        "trust_region_length": float(incumbent["trust_region_length"]),
        "g2_real": float(incumbent["x"][0]),
        "g2_imag": float(incumbent["x"][1]),
        "eps_d_real": float(incumbent["x"][2]),
        "eps_d_imag": float(incumbent["x"][3]),
        "log_kappa2": float(incumbent["v"][0]),
        "log_abs_alpha": float(incumbent["v"][1]),
        "theta_alpha": float(incumbent["v"][2]),
        "theta_g": float(incumbent["v"][3]),
        "observed_bias": float(observed["bias"]),
        "observed_T_X": float(observed["T_X"]),
        "observed_T_Z": float(observed["T_Z"]),
        "accepted": int(bool(accepted)),
    }
    return row


def run_lifetime_aware_blt(cfg: LifetimeAwareConfig | None = None, *, verbose: bool = True) -> dict:
    cfg = cfg or LifetimeAwareConfig()
    sim_cfg = TwoPointConfig()
    noise_cfg = NoiseConfig(sigma=cfg.noise_sigma, seed=cfg.noise_seed)
    reward_cfg = ImprovedRewardConfig()
    bounds = PhysicalBounds()
    thresholds = GateThresholds()
    clear_measure_cache()

    history: list[dict] = []
    incumbent: dict | None = None
    cumulative_cost = 0.0
    for epoch in range(cfg.max_epochs + 1):
        proposal_name, update_type = EPOCH_PROPOSALS.get(epoch, ("post_target_probe", BLT_REWARD_REGION))
        raw = RAW_SWEEP_CANDIDATES[proposal_name]
        observed = _evaluate_control(
            raw,
            epoch=epoch,
            update_type=update_type,
            sim_cfg=sim_cfg,
            noise_cfg=noise_cfg,
            reward_cfg=reward_cfg,
            bounds=bounds,
            thresholds=thresholds,
            cfg=cfg,
            previous=incumbent,
        )
        cumulative_cost += float(observed["cost_units"])
        accepted = _accepted(observed, incumbent, cfg)
        if accepted:
            incumbent = dict(observed)
        assert incumbent is not None
        history.append(_history_row(epoch, observed, incumbent, cumulative_cost, accepted))
        if verbose and (epoch in (0, 1, 4, 8, 9, cfg.max_epochs)):
            print(
                f"fixed_blt epoch={epoch:03d}/{cfg.max_epochs} "
                f"proposal={proposal_name} accepted={history[-1]['accepted']} "
                f"bias={incumbent['bias']:.3g} Tx={incumbent['T_X']:.4g} Tz={incumbent['T_Z']:.4g}",
                flush=True,
            )
    cleanup_local_qutip_cache()
    return {
        "history": history,
        "config": asdict(cfg),
        "sim_config": asdict(sim_cfg),
        "noise_config": asdict(noise_cfg),
        "gate_thresholds": asdict(thresholds),
        "raw_candidates": {key: value.tolist() for key, value in RAW_SWEEP_CANDIDATES.items()},
    }
