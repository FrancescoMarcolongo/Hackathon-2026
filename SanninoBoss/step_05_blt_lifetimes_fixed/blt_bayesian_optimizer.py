"""Step 5: BLT-lite hybrid optimizer built on the Step 4B TUrBO optimizer."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

from physical_coordinates_reward import (
    ImprovedRewardConfig,
    PhysicalBounds,
    evaluate_candidate_physical,
    physical_to_raw,
    raw_bounds_ok,
    raw_to_physical,
    raw_to_physical_features,
    select_better_physical,
)
from turbo_bayesian_physical_optimizer import (
    TurboBayesianOptConfig,
    TurboBayesianPhysicalOptimizer,
)
from two_points_with_noise import (
    EPS,
    NoiseConfig,
    OPTIMIZATION_START_X,
    TARGET_BIAS,
    BIAS_TOL_REL,
    TwoPointConfig,
    cleanup_local_qutip_cache,
    clear_measure_cache,
    estimate_alpha,
    params_to_complex,
)


BAYESIAN_UPDATE = "BAYESIAN_UPDATE"
BLT_REWARD_REGION = "BLT_REWARD_REGION"
BLT_JACOBIAN_UPDATE = "BLT_JACOBIAN_UPDATE"


@dataclass(frozen=True)
class GateThresholds:
    min_abs_alpha: float = 0.65
    max_abs_alpha: float = 2.45
    max_g2_over_kappa_b: float = 0.35
    max_nbar_fraction: float = 0.82
    max_markov: float = 0.85
    max_leakage_proxy: float = 0.35
    min_contrast: float = 0.015
    max_jacobian_condition: float = 1500.0
    max_bias_abs_for_jacobian: float = 1.25


@dataclass(frozen=True)
class BLTRewardConfig:
    target_bias: float = TARGET_BIAS
    w_eta: float = 18.0
    w_abs: float = 0.22
    w_N: float = 0.40
    w_leak: float = 0.30
    eps: float = 1.0e-12


@dataclass(frozen=True)
class BLTConfig:
    max_epochs: int = 45
    random_seed: int = 7
    noise_seed: int = 11
    noise_sigma: float = 0.03
    jacobian_start_epoch: int = 2
    jacobian_every: int = 1
    jacobian_delta_unit: float = 0.045
    jacobian_step_size: float = 1.35
    jacobian_damping: float = 0.03
    jacobian_max_norm: float = 0.78
    jacobian_accept_margin: float = 0.02
    use_alpha_correction: bool = True
    blt_cost_units: float = 4.0
    bayesian_cost_units: float = 2.0
    jacobian_extra_cost_units: float = 24.0


_BLT_EXACT_CACHE: Dict[tuple[float, ...], dict] = {}


def _cache_key(g2: complex, eps_d: complex, cfg: TwoPointConfig) -> tuple[float, ...]:
    return (
        round(float(g2.real), 10),
        round(float(g2.imag), 10),
        round(float(eps_d.real), 10),
        round(float(eps_d.imag), 10),
        float(cfg.na),
        float(cfg.nb),
        float(cfg.kappa_b),
        float(cfg.kappa_a),
        float(cfg.tau_x),
        float(cfg.tau_z),
    )


def _simulate_blt_points(g2: complex, eps_d: complex, cfg: TwoPointConfig) -> dict:
    key = _cache_key(g2, eps_d, cfg)
    if key in _BLT_EXACT_CACHE:
        return dict(_BLT_EXACT_CACHE[key])

    alpha = estimate_alpha(g2, eps_d, kappa_b=cfg.kappa_b, kappa_a=cfg.kappa_a)
    alpha_abs = float(abs(alpha)) if np.isfinite(abs(alpha)) else np.nan
    if not np.isfinite(alpha_abs) or alpha_abs < 0.20:
        return {"valid": False, "reason": "invalid alpha", "alpha_abs": alpha_abs}

    na, nb = int(cfg.na), int(cfg.nb)
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))
    H = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )
    loss_b = jnp.sqrt(cfg.kappa_b) * b
    loss_a = jnp.sqrt(cfg.kappa_a) * a

    g_state = dq.coherent(na, alpha)
    e_state = dq.coherent(na, -alpha)
    overlap = float(np.exp(-2.0 * alpha_abs**2))
    plus_x = (g_state + e_state) / jnp.sqrt(2.0 + 2.0 * overlap)
    buffer_vac = dq.fock(nb, 0)

    sx = (1j * jnp.pi * a.dag() @ a).expm()
    sz_storage = g_state @ g_state.dag() - e_state @ e_state.dag()
    sz = dq.tensor(sz_storage, dq.eye(nb))

    psi_x = dq.tensor(plus_x, buffer_vac)
    psi_z = dq.tensor(g_state, buffer_vac)
    ts_x = jnp.asarray([0.0, float(cfg.tau_x), float(2.0 * cfg.tau_x)])
    ts_z = jnp.asarray([0.0, float(cfg.tau_z), float(2.0 * cfg.tau_z)])
    options = dq.Options(progress_meter=False)

    try:
        res_x = dq.mesolve(H, [loss_b, loss_a], psi_x, ts_x, options=options, exp_ops=[sx])
        res_z = dq.mesolve(H, [loss_b, loss_a], psi_z, ts_z, options=options, exp_ops=[sz])
    except Exception as exc:
        return {"valid": False, "reason": f"mesolve failed: {exc}", "alpha_abs": alpha_abs}

    result = {
        "valid": True,
        "reason": "",
        "alpha_abs": alpha_abs,
        "x_values": np.asarray(res_x.expects[0].real, dtype=float),
        "z_values": np.asarray(res_z.expects[0].real, dtype=float),
    }
    _BLT_EXACT_CACHE[key] = dict(result)
    return result


def _noisy_ratios(values: np.ndarray, sigma: float, rng: np.random.Generator) -> tuple[float, float, float, float]:
    noisy = np.asarray(values, dtype=float).copy()
    noisy[1] = float(np.clip(noisy[1] + rng.normal(0.0, sigma), -1.0, 1.0))
    noisy[2] = float(np.clip(noisy[2] + rng.normal(0.0, sigma), -1.0, 1.0))
    y0 = float(noisy[0])
    if abs(y0) <= EPS:
        return np.nan, np.nan, np.nan, np.nan
    ratio_tau = abs(float(noisy[1]) / y0)
    ratio_2tau = abs(float(noisy[2]) / y0)
    contrast = max(abs(float(noisy[1])), abs(float(noisy[2])))
    markov = abs(ratio_2tau - ratio_tau**2) / (abs(ratio_2tau) + EPS)
    return ratio_tau, ratio_2tau, markov, contrast


def evaluate_blt_lite(
    v: np.ndarray,
    *,
    sim_cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    bounds: PhysicalBounds,
    rng: np.random.Generator,
    use_alpha_correction: bool,
) -> dict:
    u = physical_to_raw(v, sim_cfg, use_alpha_correction=use_alpha_correction)
    if not raw_bounds_ok(u):
        return _failed_blt(v, u, "raw_bounds")
    g2, eps_d = params_to_complex(u)
    exact = _simulate_blt_points(g2, eps_d, sim_cfg)
    if not exact.get("valid", False):
        return _failed_blt(v, u, str(exact.get("reason", "invalid blt simulation")))

    x_tau, x_2tau, markov_x, contrast_x = _noisy_ratios(exact["x_values"], noise_cfg.sigma, rng)
    z_tau, z_2tau, markov_z, contrast_z = _noisy_ratios(exact["z_values"], noise_cfg.sigma, rng)
    if not all(np.isfinite(value) for value in (x_tau, x_2tau, z_tau, z_2tau)):
        return _failed_blt(v, u, "nonfinite noisy BLT ratios")
    if min(x_tau, x_2tau, z_tau, z_2tau) <= 0.0 or max(x_tau, x_2tau, z_tau, z_2tau) >= 1.0:
        return _failed_blt(v, u, "invalid BLT decay ratio")

    gamma_x_tau = -math.log(x_tau) / sim_cfg.tau_x
    gamma_x_2tau = -math.log(x_2tau) / (2.0 * sim_cfg.tau_x)
    gamma_z_tau = -math.log(z_tau) / sim_cfg.tau_z
    gamma_z_2tau = -math.log(z_2tau) / (2.0 * sim_cfg.tau_z)
    gamma_x = 0.75 * gamma_x_tau + 0.25 * gamma_x_2tau
    gamma_z = 0.75 * gamma_z_tau + 0.25 * gamma_z_2tau
    if gamma_x <= 0.0 or gamma_z <= 0.0:
        return _failed_blt(v, u, "nonpositive BLT rate")

    tx = 1.0 / gamma_x
    tz = 1.0 / gamma_z
    bias = gamma_x / gamma_z
    features = raw_to_physical_features(u, sim_cfg, use_alpha_correction=use_alpha_correction)
    nbar = float(features.get("abs_alpha", np.nan)) ** 2
    usable = max(1.0, float(sim_cfg.na) - sim_cfg.alpha_margin)
    leakage_proxy = max(0.0, nbar / usable - 0.70)
    markov = max(float(markov_x), float(markov_z))
    e_bias = math.log(max(bias, EPS) / TARGET_BIAS)
    log_gamma_product = math.log(max(gamma_x * gamma_z, EPS))

    result = {
        "v": np.asarray(v, dtype=float),
        "x": np.asarray(u, dtype=float),
        "u": np.asarray(u, dtype=float),
        "gamma_X": float(gamma_x),
        "gamma_Z": float(gamma_z),
        "T_X": float(tx),
        "T_Z": float(tz),
        "bias": float(bias),
        "eta_blt": float(bias),
        "geo_lifetime": float(math.sqrt(tx * tz)),
        "N_markov": float(markov),
        "markov_x": float(markov_x),
        "markov_z": float(markov_z),
        "leakage_proxy": float(leakage_proxy),
        "contrast_x": float(contrast_x),
        "contrast_z": float(contrast_z),
        "e_bias": float(e_bias),
        "log_gamma_product": float(log_gamma_product),
        "valid_numerics": True,
        "reason": "",
    }
    result.update(features)
    return result


def _failed_blt(v: np.ndarray, u: np.ndarray, reason: str) -> dict:
    return {
        "v": np.asarray(v, dtype=float),
        "x": np.asarray(u, dtype=float),
        "u": np.asarray(u, dtype=float),
        "gamma_X": np.nan,
        "gamma_Z": np.nan,
        "T_X": np.nan,
        "T_Z": np.nan,
        "bias": np.nan,
        "eta_blt": np.nan,
        "geo_lifetime": np.nan,
        "N_markov": np.inf,
        "leakage_proxy": np.inf,
        "contrast_x": 0.0,
        "contrast_z": 0.0,
        "e_bias": np.inf,
        "log_gamma_product": np.inf,
        "valid_numerics": False,
        "reason": reason,
    }


def gate1_cat_feasible(blt: dict, thresholds: GateThresholds, sim_cfg: TwoPointConfig) -> tuple[bool, str]:
    u = np.asarray(blt["u"], dtype=float)
    if not raw_bounds_ok(u):
        return False, "raw_bounds"
    abs_alpha = float(blt.get("abs_alpha", np.nan))
    abs_g2 = float(blt.get("abs_g2", np.nan))
    if not np.isfinite(abs_alpha) or abs_alpha < thresholds.min_abs_alpha or abs_alpha > thresholds.max_abs_alpha:
        return False, "alpha_bounds"
    if not np.isfinite(abs_g2) or abs_g2 / sim_cfg.kappa_b > thresholds.max_g2_over_kappa_b:
        return False, "g2_over_kappa_b"
    nbar = abs_alpha**2
    usable = max(1.0, float(sim_cfg.na) - sim_cfg.alpha_margin)
    if nbar / usable > thresholds.max_nbar_fraction:
        return False, "cat_too_large"
    return True, ""


def gate2_blt_valid(blt: dict, thresholds: GateThresholds) -> tuple[bool, str]:
    if not bool(blt.get("valid_numerics", False)):
        return False, str(blt.get("reason", "invalid_numerics"))
    for key in ("gamma_X", "gamma_Z", "T_X", "T_Z", "bias"):
        value = float(blt.get(key, np.nan))
        if not np.isfinite(value) or value <= 0.0:
            return False, f"invalid_{key}"
    if float(blt["N_markov"]) > thresholds.max_markov:
        return False, "markov_proxy"
    if float(blt["leakage_proxy"]) > thresholds.max_leakage_proxy:
        return False, "leakage_proxy"
    if min(float(blt["contrast_x"]), float(blt["contrast_z"])) < thresholds.min_contrast:
        return False, "low_contrast"
    return True, ""


def reward_blt(blt: dict, weights: BLTRewardConfig) -> dict:
    if not bool(blt.get("valid_numerics", False)):
        return {"reward": -1.0e6, "is_feasible": False, "bias_error": np.inf, "bias_rel_error": np.inf}
    e_bias = float(blt["e_bias"])
    log_gamma_product = float(blt["log_gamma_product"])
    markov = float(blt["N_markov"])
    leak = float(blt["leakage_proxy"])
    loss = (
        weights.w_eta * e_bias**2
        + weights.w_abs * log_gamma_product
        + weights.w_N * markov**2
        + weights.w_leak * leak**2
    )
    reward = -float(loss)
    bias = float(blt["bias"])
    bias_rel_error = abs(bias / weights.target_bias - 1.0)
    return {
        "reward": reward,
        "is_feasible": bool(bias_rel_error <= BIAS_TOL_REL),
        "bias_error": abs(e_bias),
        "bias_rel_error": float(bias_rel_error),
    }


def _blt_row(
    blt: dict,
    reward: dict,
    update_type: str,
    gate1_pass: bool,
    gate2_pass: bool,
    gate3_pass: bool,
    *,
    cost_units: float,
    cumulative_cost_units: float,
    trust_region_length: float,
    reason: str = "",
) -> dict:
    row = {
        "v": np.asarray(blt["v"], dtype=float),
        "x": np.asarray(blt["x"], dtype=float),
        "u": np.asarray(blt["u"], dtype=float),
        "reward": float(reward["reward"]),
        "reward_safe": float(reward["reward"]),
        "loss_to_minimize": -float(reward["reward"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_error": float(reward["bias_error"]),
        "bias_rel_error": float(reward["bias_rel_error"]),
        "T_X": float(blt["T_X"]),
        "T_Z": float(blt["T_Z"]),
        "bias": float(blt["bias"]),
        "geo_lifetime": float(blt["geo_lifetime"]),
        "update_type": update_type,
        "gate1_pass": int(bool(gate1_pass)),
        "gate2_pass": int(bool(gate2_pass)),
        "gate3_pass": int(bool(gate3_pass)),
        "N_markov": float(blt.get("N_markov", np.nan)),
        "leakage_proxy": float(blt.get("leakage_proxy", np.nan)),
        "contrast_x": float(blt.get("contrast_x", np.nan)),
        "contrast_z": float(blt.get("contrast_z", np.nan)),
        "gamma_X": float(blt.get("gamma_X", np.nan)),
        "gamma_Z": float(blt.get("gamma_Z", np.nan)),
        "cost_units": float(cost_units),
        "cumulative_cost_units": float(cumulative_cost_units),
        "trust_region_length": float(trust_region_length),
        "reason": reason,
    }
    return row


def _fallback_step4_row(
    v: np.ndarray,
    *,
    sim_cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    reward_cfg: ImprovedRewardConfig,
    bounds: PhysicalBounds,
    rng: np.random.Generator,
    use_alpha_correction: bool,
    cost_units: float,
    cumulative_cost_units: float,
    trust_region_length: float,
    reason: str,
) -> dict:
    step4 = evaluate_candidate_physical(
        v,
        sim_cfg=sim_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        bounds=bounds,
        rng=rng,
        use_alpha_correction=use_alpha_correction,
    )
    step4.update(
        {
            "update_type": BAYESIAN_UPDATE,
            "gate1_pass": 0,
            "gate2_pass": 0,
            "gate3_pass": 0,
            "N_markov": np.nan,
            "leakage_proxy": np.nan,
            "contrast_x": np.nan,
            "contrast_z": np.nan,
            "gamma_X": 1.0 / float(step4["T_X"]) if np.isfinite(float(step4["T_X"])) else np.nan,
            "gamma_Z": 1.0 / float(step4["T_Z"]) if np.isfinite(float(step4["T_Z"])) else np.nan,
            "cost_units": float(cost_units),
            "cumulative_cost_units": float(cumulative_cost_units),
            "trust_region_length": float(trust_region_length),
            "reason": reason,
        }
    )
    return step4


def _history_row(run_id: str, epoch: int, observed: dict, incumbent: dict) -> dict:
    row: Dict[str, object] = {
        "run_id": run_id,
        "epoch": int(epoch),
        "update_type": observed.get("update_type", ""),
        "gate1_pass": int(observed.get("gate1_pass", 0)),
        "gate2_pass": int(observed.get("gate2_pass", 0)),
        "gate3_pass": int(observed.get("gate3_pass", 0)),
        "cost_units": float(observed.get("cost_units", np.nan)),
        "cumulative_cost_units": float(observed.get("cumulative_cost_units", np.nan)),
        "trust_region_length": float(observed.get("trust_region_length", np.nan)),
        "N_markov": float(observed.get("N_markov", np.nan)),
        "leakage_proxy": float(observed.get("leakage_proxy", np.nan)),
        "reason": observed.get("reason", ""),
    }
    for prefix, item in (("observed", observed), ("incumbent", incumbent)):
        row[f"{prefix}_reward"] = float(item["reward"])
        row[f"{prefix}_T_X"] = float(item["T_X"])
        row[f"{prefix}_T_Z"] = float(item["T_Z"])
        row[f"{prefix}_bias"] = float(item["bias"])
        row[f"{prefix}_geo_lifetime"] = float(item["geo_lifetime"])
        row[f"{prefix}_is_feasible"] = int(bool(item["is_feasible"]))
        row[f"{prefix}_bias_error"] = float(item["bias_error"])
        u = np.asarray(item["x"], dtype=float)
        v = np.asarray(item["v"], dtype=float)
        row[f"{prefix}_g2_real"] = float(u[0])
        row[f"{prefix}_g2_imag"] = float(u[1])
        row[f"{prefix}_eps_d_real"] = float(u[2])
        row[f"{prefix}_eps_d_imag"] = float(u[3])
        row[f"{prefix}_log_kappa2"] = float(v[0])
        row[f"{prefix}_log_abs_alpha"] = float(v[1])
        row[f"{prefix}_theta_alpha"] = float(v[2])
        row[f"{prefix}_theta_g"] = float(v[3])
    return row


def _feature_vector(blt: dict) -> np.ndarray:
    return np.asarray(
        [
            float(blt["e_bias"]),
            float(blt["log_gamma_product"]),
            float(blt["N_markov"]),
            float(blt["leakage_proxy"]),
        ],
        dtype=float,
    )


def _evaluate_blt_reward_for_jac(
    v: np.ndarray,
    *,
    sim_cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    bounds: PhysicalBounds,
    rng: np.random.Generator,
    use_alpha_correction: bool,
    thresholds: GateThresholds,
    weights: BLTRewardConfig,
) -> tuple[dict, dict, bool]:
    blt = evaluate_blt_lite(
        v,
        sim_cfg=sim_cfg,
        noise_cfg=noise_cfg,
        bounds=bounds,
        rng=rng,
        use_alpha_correction=use_alpha_correction,
    )
    gate1, _ = gate1_cat_feasible(blt, thresholds, sim_cfg)
    gate2, _ = gate2_blt_valid(blt, thresholds)
    reward = reward_blt(blt, weights)
    # For finite-difference Jacobian estimation we only need locally finite
    # BLT features. Gate 2 is still enforced for accepting the final trial.
    return blt, reward, bool(gate1 and blt.get("valid_numerics", False))


def _prefer_jacobian_trial(
    trial_blt: dict,
    trial_reward: dict,
    best_blt: dict | None,
    best_reward: dict | None,
) -> bool:
    if best_blt is None or best_reward is None:
        return True
    trial_rel = float(trial_reward["bias_rel_error"])
    best_rel = float(best_reward["bias_rel_error"])
    trial_in_band = trial_rel <= BIAS_TOL_REL
    best_in_band = best_rel <= BIAS_TOL_REL
    if trial_in_band and not best_in_band:
        return True
    if best_in_band and not trial_in_band:
        return False
    if abs(trial_rel - best_rel) > 0.003:
        return trial_rel < best_rel
    return float(trial_reward["reward"]) > float(best_reward["reward"])


def _interpolate_unit(unit_a: np.ndarray, unit_b: np.ndarray, fraction: float) -> np.ndarray:
    out = np.asarray(unit_a, dtype=float).copy()
    diff = np.asarray(unit_b, dtype=float) - np.asarray(unit_a, dtype=float)
    diff[2:] = (diff[2:] + 0.5) % 1.0 - 0.5
    out[:2] = np.clip(out[:2] + fraction * diff[:2], 0.0, 1.0)
    out[2:] = (out[2:] + fraction * diff[2:]) % 1.0
    return out


def propose_blt_jacobian_update(
    v_current: np.ndarray,
    blt_current: dict,
    *,
    optimizer: TurboBayesianPhysicalOptimizer,
    sim_cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    bounds: PhysicalBounds,
    rng: np.random.Generator,
    use_alpha_correction: bool,
    thresholds: GateThresholds,
    weights: BLTRewardConfig,
    config: BLTConfig,
) -> tuple[np.ndarray | None, dict | None, dict | None, bool]:
    current_feature = _feature_vector(blt_current)
    unit_current = optimizer._to_unit(v_current)
    J = np.zeros((4, 4), dtype=float)
    valid_columns = 0
    for col in range(4):
        delta = np.zeros(4, dtype=float)
        delta[col] = config.jacobian_delta_unit
        unit_plus = unit_current.copy()
        unit_minus = unit_current.copy()
        if col < 2:
            unit_plus[col] = np.clip(unit_plus[col] + delta[col], 0.0, 1.0)
            unit_minus[col] = np.clip(unit_minus[col] - delta[col], 0.0, 1.0)
        else:
            unit_plus[col] = (unit_plus[col] + delta[col]) % 1.0
            unit_minus[col] = (unit_minus[col] - delta[col]) % 1.0
        blt_plus, _reward_plus, ok_plus = _evaluate_blt_reward_for_jac(
            optimizer._from_unit(unit_plus),
            sim_cfg=sim_cfg,
            noise_cfg=noise_cfg,
            bounds=bounds,
            rng=rng,
            use_alpha_correction=use_alpha_correction,
            thresholds=thresholds,
            weights=weights,
        )
        blt_minus, _reward_minus, ok_minus = _evaluate_blt_reward_for_jac(
            optimizer._from_unit(unit_minus),
            sim_cfg=sim_cfg,
            noise_cfg=noise_cfg,
            bounds=bounds,
            rng=rng,
            use_alpha_correction=use_alpha_correction,
            thresholds=thresholds,
            weights=weights,
        )
        if ok_plus and ok_minus:
            J[:, col] = (_feature_vector(blt_plus) - _feature_vector(blt_minus)) / (2.0 * config.jacobian_delta_unit)
            valid_columns += 1
        elif ok_plus:
            J[:, col] = (_feature_vector(blt_plus) - current_feature) / config.jacobian_delta_unit
            valid_columns += 1
        elif ok_minus:
            J[:, col] = (current_feature - _feature_vector(blt_minus)) / config.jacobian_delta_unit
            valid_columns += 1
        else:
            J[:, col] = 0.0

    if valid_columns == 0 or not np.all(np.isfinite(J)):
        return None, None, None, False
    try:
        condition = float(np.linalg.cond(J.T @ J + config.jacobian_damping * np.eye(4)))
    except Exception:
        condition = np.inf

    bias_grad = J[0]
    if not np.any(np.isfinite(bias_grad) & (np.abs(bias_grad) > 1.0e-5)):
        return None, None, None, False
    denom = float(bias_grad @ bias_grad + config.jacobian_damping)
    if not np.isfinite(denom) or denom <= 0.0:
        return None, None, None, False
    # Primary BLT local move: solve the linearized log-bias equation
    # e_bias(v + delta) ~= 0.  The remaining BLT diagnostics are enforced
    # by Gate 2 and by the line-search acceptance below.
    delta_unit = -config.jacobian_step_size * current_feature[0] * bias_grad / denom
    norm = float(np.linalg.norm(delta_unit))
    if not np.isfinite(norm) or norm <= 0.0:
        return None, None, None, False
    if norm > config.jacobian_max_norm:
        delta_unit *= config.jacobian_max_norm / norm

    current_reward = reward_blt(blt_current, weights)
    current_bias_error = abs(float(current_feature[0]))
    best_v: np.ndarray | None = None
    best_blt: dict | None = None
    best_reward: dict | None = None

    def consider_candidate(
        trial_v: np.ndarray,
        blt_trial: dict,
        reward_trial: dict,
        ok_trial: bool,
    ) -> None:
        nonlocal best_v, best_blt, best_reward
        gate2_trial, _ = gate2_blt_valid(blt_trial, thresholds)
        trial_bias_error = abs(float(blt_trial.get("e_bias", np.inf)))
        reward_improves = float(reward_trial["reward"]) > float(current_reward["reward"]) + config.jacobian_accept_margin
        bias_improves = trial_bias_error < current_bias_error * 0.82
        lands_in_band = float(reward_trial["bias_rel_error"]) <= BIAS_TOL_REL
        if ok_trial and gate2_trial and (reward_improves or bias_improves or lands_in_band):
            if _prefer_jacobian_trial(blt_trial, reward_trial, best_blt, best_reward):
                best_v = trial_v
                best_blt = blt_trial
                best_reward = reward_trial

    def consider_bracket(trial_unit: np.ndarray, blt_trial: dict) -> None:
        e0 = float(current_feature[0])
        e1 = float(blt_trial.get("e_bias", np.nan))
        if not np.isfinite(e0) or not np.isfinite(e1) or e0 * e1 >= 0.0:
            return
        fraction = abs(e0) / (abs(e0) + abs(e1) + EPS)
        for scale in (0.85, 1.0, 1.15):
            bracket_fraction = float(np.clip(scale * fraction, 0.05, 0.95))
            bracket_unit = _interpolate_unit(unit_current, trial_unit, bracket_fraction)
            bracket_v = optimizer._from_unit(bracket_unit)
            bracket_blt, bracket_reward, bracket_ok = _evaluate_blt_reward_for_jac(
                bracket_v,
                sim_cfg=sim_cfg,
                noise_cfg=noise_cfg,
                bounds=bounds,
                rng=rng,
                use_alpha_correction=use_alpha_correction,
                thresholds=thresholds,
                weights=weights,
            )
            consider_candidate(bracket_v, bracket_blt, bracket_reward, bracket_ok)

    for scale in (1.0, 0.65, 1.35, 1.8, 2.5, 3.5):
        trial_unit = unit_current.copy()
        trial_unit[:2] = np.clip(trial_unit[:2] + scale * delta_unit[:2], 0.0, 1.0)
        trial_unit[2:] = (trial_unit[2:] + scale * delta_unit[2:]) % 1.0
        trial_v = optimizer._from_unit(trial_unit)
        blt_trial, reward_trial, ok_trial = _evaluate_blt_reward_for_jac(
            trial_v,
            sim_cfg=sim_cfg,
            noise_cfg=noise_cfg,
            bounds=bounds,
            rng=rng,
            use_alpha_correction=use_alpha_correction,
            thresholds=thresholds,
            weights=weights,
        )
        consider_candidate(trial_v, blt_trial, reward_trial, ok_trial)
        consider_bracket(trial_unit, blt_trial)
    if best_reward is None:
        for col, gradient in enumerate(bias_grad):
            if not np.isfinite(gradient) or abs(float(gradient)) < 1.0e-5:
                continue
            signed_step = -config.jacobian_step_size * current_feature[0] / float(gradient)
            signed_step = float(np.clip(signed_step, -config.jacobian_max_norm, config.jacobian_max_norm))
            for scale in (0.35, 0.60, 0.85, 1.00, 1.25, 1.60):
                trial_unit = unit_current.copy()
                move = scale * signed_step
                if col < 2:
                    trial_unit[col] = np.clip(trial_unit[col] + move, 0.0, 1.0)
                else:
                    trial_unit[col] = (trial_unit[col] + move) % 1.0
                trial_v = optimizer._from_unit(trial_unit)
                blt_trial, reward_trial, ok_trial = _evaluate_blt_reward_for_jac(
                    trial_v,
                    sim_cfg=sim_cfg,
                    noise_cfg=noise_cfg,
                    bounds=bounds,
                    rng=rng,
                    use_alpha_correction=use_alpha_correction,
                    thresholds=thresholds,
                    weights=weights,
                )
                consider_candidate(trial_v, blt_trial, reward_trial, ok_trial)
                consider_bracket(trial_unit, blt_trial)
    return best_v, best_blt, best_reward, best_v is not None


def run_blt_hybrid_optimizer(
    *,
    verbose: bool = True,
    blt_cfg: BLTConfig | None = None,
    turbo_cfg: TurboBayesianOptConfig | None = None,
    reward_cfg: ImprovedRewardConfig | None = None,
    bounds: PhysicalBounds | None = None,
    thresholds: GateThresholds | None = None,
    weights: BLTRewardConfig | None = None,
) -> dict:
    blt_cfg = blt_cfg or BLTConfig()
    turbo_cfg = turbo_cfg or TurboBayesianOptConfig(
        max_epochs=blt_cfg.max_epochs,
        n_init=4,
        beta=0.4,
        n_pool=4096,
        global_fraction=0.12,
        initial_trust_region_length=0.58,
        length_min=0.04,
        length_max=0.90,
        success_tolerance=2,
        failure_tolerance=4,
        random_seed=blt_cfg.random_seed,
        noise_seed=blt_cfg.noise_seed,
        noise_sigma=blt_cfg.noise_sigma,
    )
    reward_cfg = reward_cfg or ImprovedRewardConfig()
    bounds = bounds or PhysicalBounds()
    thresholds = thresholds or GateThresholds()
    weights = weights or BLTRewardConfig()
    sim_cfg = TwoPointConfig()
    noise_cfg = NoiseConfig(sigma=blt_cfg.noise_sigma, seed=blt_cfg.noise_seed)
    clear_measure_cache()
    _BLT_EXACT_CACHE.clear()

    noise_rng = np.random.default_rng(blt_cfg.noise_seed)
    turbo_rng = np.random.default_rng(turbo_cfg.random_seed)
    v0 = raw_to_physical(
        OPTIMIZATION_START_X.copy(),
        sim_cfg,
        use_alpha_correction=blt_cfg.use_alpha_correction,
    )
    turbo = TurboBayesianPhysicalOptimizer(bounds=bounds, config=turbo_cfg, sim_cfg=sim_cfg, v0=v0, rng=turbo_rng)
    run_id = f"step05_blt_seed{blt_cfg.random_seed}"
    history: list[dict] = []
    cumulative_cost = 0.0

    start_eval = evaluate_candidate_physical(
        v0,
        sim_cfg=sim_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        bounds=bounds,
        rng=noise_rng,
        use_alpha_correction=blt_cfg.use_alpha_correction,
    )
    cumulative_cost += blt_cfg.bayesian_cost_units
    start_eval.update(
        {
            "update_type": BAYESIAN_UPDATE,
            "gate1_pass": 0,
            "gate2_pass": 0,
            "gate3_pass": 0,
            "N_markov": np.nan,
            "leakage_proxy": np.nan,
            "contrast_x": np.nan,
            "contrast_z": np.nan,
            "gamma_X": 1.0 / float(start_eval["T_X"]),
            "gamma_Z": 1.0 / float(start_eval["T_Z"]),
            "cost_units": blt_cfg.bayesian_cost_units,
            "cumulative_cost_units": cumulative_cost,
            "trust_region_length": turbo.trust_region_length,
            "reason": "initial_step4b_style_measurement",
        }
    )
    incumbent = dict(start_eval)
    turbo.tell(v0, start_eval)
    history.append(_history_row(run_id, 0, start_eval, incumbent))
    best_blt_state: dict | None = None
    start_time = time.time()

    for epoch in range(1, blt_cfg.max_epochs + 1):
        attempted_jacobian = False
        gate3_pass = False
        observed: dict

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
            attempted_jacobian = True
            candidate_v, jac_blt, jac_reward, ok_jac = propose_blt_jacobian_update(
                np.asarray(best_blt_state["v"], dtype=float),
                best_blt_state,
                optimizer=turbo,
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
                gate3_pass = True
                observed = _blt_row(
                    jac_blt,
                    jac_reward,
                    BLT_JACOBIAN_UPDATE,
                    True,
                    True,
                    True,
                    cost_units=blt_cfg.blt_cost_units + blt_cfg.jacobian_extra_cost_units,
                    cumulative_cost_units=cumulative_cost,
                    trust_region_length=turbo.trust_region_length,
                )
            else:
                attempted_jacobian = False

        if not attempted_jacobian or not gate3_pass:
            candidate_v = turbo.ask()
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
                    trust_region_length=turbo.trust_region_length,
                )
            else:
                observed = _fallback_step4_row(
                    candidate_v,
                    sim_cfg=sim_cfg,
                    noise_cfg=noise_cfg,
                    reward_cfg=reward_cfg,
                    bounds=bounds,
                    rng=noise_rng,
                    use_alpha_correction=blt_cfg.use_alpha_correction,
                    cost_units=blt_cfg.bayesian_cost_units,
                    cumulative_cost_units=cumulative_cost,
                    trust_region_length=turbo.trust_region_length,
                    reason=reason1 or reason2,
                )

        turbo.tell(np.asarray(observed["v"], dtype=float), observed)
        if observed["update_type"] in (BLT_REWARD_REGION, BLT_JACOBIAN_UPDATE):
            best_blt_state = {
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
        if select_better_physical(observed, incumbent):
            incumbent = dict(observed)
        row = _history_row(run_id, epoch, observed, incumbent)
        row["trust_region_length"] = float(turbo.trust_region_length)
        history.append(row)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == blt_cfg.max_epochs):
            elapsed = time.time() - start_time
            print(
                f"{run_id} epoch={epoch:03d}/{blt_cfg.max_epochs} "
                f"type={observed['update_type']} "
                f"incumbent bias={float(incumbent['bias']):.4g} "
                f"reward={float(incumbent['reward']):.3f} "
                f"tr={turbo.trust_region_length:.3f} "
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
        "turbo_config": asdict(turbo_cfg),
        "reward_config": asdict(weights),
        "gate_thresholds": asdict(thresholds),
        "physical_bounds": asdict(bounds),
    }
