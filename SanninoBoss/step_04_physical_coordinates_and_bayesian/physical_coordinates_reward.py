"""Step 4A: physical coordinates plus improved reward."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from cmaes import SepCMA

from two_points_with_noise import (
    BIAS_TOL_REL,
    BOUNDS,
    EPS,
    NoiseConfig,
    OPTIMIZATION_START_X,
    TARGET_BIAS,
    TwoPointConfig,
    cleanup_local_qutip_cache,
    clear_measure_cache,
    estimate_alpha,
    measure_lifetimes_two_point_noisy,
    params_to_complex,
)


@dataclass(frozen=True)
class PhysicalBounds:
    log_kappa2_min: float = math.log(0.08)
    log_kappa2_max: float = math.log(3.2)
    log_abs_alpha_min: float = math.log(0.65)
    log_abs_alpha_max: float = math.log(2.45)
    theta_alpha_min: float = -math.pi
    theta_alpha_max: float = math.pi
    theta_g_min: float = -math.pi
    theta_g_max: float = math.pi

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                [self.log_kappa2_min, self.log_kappa2_max],
                [self.log_abs_alpha_min, self.log_abs_alpha_max],
                [self.theta_alpha_min, self.theta_alpha_max],
                [self.theta_g_min, self.theta_g_max],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class ImprovedRewardConfig:
    target_bias: float = TARGET_BIAS
    bias_tol_rel: float = BIAS_TOL_REL
    w_eta: float = 260.0
    w_lifetime: float = 0.70
    w_fit: float = 1.0
    w_floor: float = 9.0
    min_tx: float = 0.05
    min_tz: float = 5.0
    eps_t: float = 1.0e-9
    invalid_reward: float = -1.0e6


@dataclass(frozen=True)
class PhysicalOptimizerConfig:
    epochs: int = 45
    population: int = 8
    sigma0: float = 0.32
    optimizer_seed: int = 4
    noise_seed: int = 11
    noise_sigma: float = 0.03
    use_alpha_correction: bool = True


def raw_to_complex(u: np.ndarray) -> tuple[complex, complex]:
    return params_to_complex(np.asarray(u, dtype=float))


def complex_to_raw(g2: complex, eps_d: complex) -> np.ndarray:
    return np.asarray([g2.real, g2.imag, eps_d.real, eps_d.imag], dtype=float)


def wrap_angle(theta: float | np.ndarray) -> float | np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def clip_physical(v: np.ndarray, bounds: PhysicalBounds) -> np.ndarray:
    clipped = np.minimum(np.maximum(np.asarray(v, dtype=float), bounds.as_array()[:, 0]), bounds.as_array()[:, 1])
    clipped[2] = float(wrap_angle(clipped[2]))
    clipped[3] = float(wrap_angle(clipped[3]))
    return clipped


def raw_bounds_ok(u: np.ndarray) -> bool:
    u = np.asarray(u, dtype=float)
    return bool(np.all(np.isfinite(u)) and np.all(u >= BOUNDS[:, 0]) and np.all(u <= BOUNDS[:, 1]))


def physical_to_raw(
    v: np.ndarray,
    sim_cfg: TwoPointConfig,
    *,
    use_alpha_correction: bool = True,
) -> np.ndarray:
    log_kappa2, log_abs_alpha, theta_alpha, theta_g = np.asarray(v, dtype=float)
    kappa2 = float(np.exp(log_kappa2))
    abs_alpha = float(np.exp(log_abs_alpha))
    alpha = abs_alpha * np.exp(1j * theta_alpha)
    abs_g2 = math.sqrt(sim_cfg.kappa_b * kappa2 / 4.0)
    g2 = abs_g2 * np.exp(1j * theta_g)
    eps_d = g2 * alpha**2
    if use_alpha_correction:
        eps_d = eps_d + sim_cfg.kappa_b * sim_cfg.kappa_a / (8.0 * g2)
    return complex_to_raw(g2, eps_d)


def raw_to_physical(
    u: np.ndarray,
    sim_cfg: TwoPointConfig,
    *,
    use_alpha_correction: bool = True,
) -> np.ndarray:
    g2, eps_d = raw_to_complex(u)
    abs_g2 = abs(g2)
    if abs_g2 <= EPS:
        raise ValueError("raw point has |g2| too close to zero")
    kappa2 = 4.0 * abs_g2**2 / sim_cfg.kappa_b
    if use_alpha_correction:
        alpha_sq = (eps_d - sim_cfg.kappa_b * sim_cfg.kappa_a / (8.0 * g2)) / g2
        alpha = complex(np.sqrt(alpha_sq + 0.0j))
    else:
        alpha = estimate_alpha(g2, eps_d, kappa_b=sim_cfg.kappa_b, kappa_a=sim_cfg.kappa_a)
    return np.asarray(
        [
            math.log(max(kappa2, EPS)),
            math.log(max(abs(alpha), EPS)),
            float(np.angle(alpha)),
            float(np.angle(g2)),
        ],
        dtype=float,
    )


def raw_to_physical_features(
    u: np.ndarray,
    sim_cfg: TwoPointConfig,
    *,
    use_alpha_correction: bool = True,
) -> Dict[str, float | bool]:
    try:
        g2, eps_d = raw_to_complex(u)
        v = raw_to_physical(u, sim_cfg, use_alpha_correction=use_alpha_correction)
        kappa2 = 4.0 * abs(g2) ** 2 / sim_cfg.kappa_b
        return {
            "valid_physical": True,
            "abs_g2": float(abs(g2)),
            "abs_eps_d": float(abs(eps_d)),
            "kappa2": float(kappa2),
            "abs_alpha": float(np.exp(v[1])),
            "theta_alpha": float(v[2]),
            "theta_g": float(v[3]),
        }
    except Exception:
        return {
            "valid_physical": False,
            "abs_g2": np.nan,
            "abs_eps_d": np.nan,
            "kappa2": np.nan,
            "abs_alpha": np.nan,
            "theta_alpha": np.nan,
            "theta_g": np.nan,
        }


def improved_reward(metrics: dict, cfg: ImprovedRewardConfig) -> dict:
    tx = float(metrics.get("T_X", np.nan))
    tz = float(metrics.get("T_Z", np.nan))
    bias = float(metrics.get("bias", np.nan))
    fit_penalty = float(metrics.get("fit_penalty", 1.0e3))
    valid = bool(metrics.get("valid", False))
    finite = bool(np.isfinite(tx) and np.isfinite(tz) and np.isfinite(bias))
    if not finite or tx <= 0.0 or tz <= 0.0 or bias <= 0.0 or not valid:
        return {
            "reward": float(cfg.invalid_reward),
            "reward_safe": float(cfg.invalid_reward),
            "is_feasible": False,
            "bias_error": np.inf,
            "bias_rel_error": np.inf,
            "floor_penalty": 1.0e6,
            "reason": "invalid_measurement",
        }

    log_bias_error = abs(math.log(bias / cfg.target_bias))
    log_band = math.log(1.0 + cfg.bias_tol_rel)
    bias_excess = max(0.0, log_bias_error - log_band)
    lifetime_score = 0.5 * (math.log(tx + cfg.eps_t) + math.log(tz + cfg.eps_t))
    floor_penalty = cfg.w_floor * (
        max(0.0, math.log(cfg.min_tx / tx)) ** 2
        + max(0.0, math.log(cfg.min_tz / tz)) ** 2
    )
    reward = (
        -cfg.w_eta * bias_excess**2
        + cfg.w_lifetime * lifetime_score
        - cfg.w_fit * fit_penalty
        - floor_penalty
    )
    bias_rel_error = abs(bias / cfg.target_bias - 1.0)
    is_feasible = bool(bias_rel_error <= cfg.bias_tol_rel)
    return {
        "reward": float(reward),
        "reward_safe": float(reward),
        "is_feasible": is_feasible,
        "bias_error": float(log_bias_error),
        "bias_rel_error": float(bias_rel_error),
        "floor_penalty": float(floor_penalty),
        "reason": "",
    }


def failed_result(v: np.ndarray, u: np.ndarray, reward_cfg: ImprovedRewardConfig, reason: str) -> dict:
    return {
        "v": np.asarray(v, dtype=float),
        "x": np.asarray(u, dtype=float),
        "u": np.asarray(u, dtype=float),
        "reward": float(reward_cfg.invalid_reward),
        "reward_safe": float(reward_cfg.invalid_reward),
        "loss_to_minimize": -float(reward_cfg.invalid_reward),
        "is_feasible": False,
        "bias_error": np.inf,
        "bias_rel_error": np.inf,
        "T_X": np.nan,
        "T_Z": np.nan,
        "bias": np.nan,
        "geo_lifetime": np.nan,
        "fit_penalty": 1.0e3,
        "fit_ok": False,
        "valid": False,
        "reason": reason,
    }


def evaluate_candidate_physical(
    v: np.ndarray,
    *,
    sim_cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    reward_cfg: ImprovedRewardConfig,
    bounds: PhysicalBounds,
    rng: np.random.Generator,
    use_alpha_correction: bool = True,
) -> dict:
    v = clip_physical(v, bounds)
    u = physical_to_raw(v, sim_cfg, use_alpha_correction=use_alpha_correction)
    if not raw_bounds_ok(u):
        return failed_result(v, u, reward_cfg, "raw_bounds")
    g2, eps_d = raw_to_complex(u)
    metrics = measure_lifetimes_two_point_noisy(g2, eps_d, sim_cfg, noise_cfg, rng)
    reward = improved_reward(metrics, reward_cfg)
    features = raw_to_physical_features(u, sim_cfg, use_alpha_correction=use_alpha_correction)
    row = {
        "v": v,
        "x": u,
        "u": u,
        "reward": float(reward["reward"]),
        "reward_safe": float(reward["reward_safe"]),
        "loss_to_minimize": -float(reward["reward_safe"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_error": float(reward["bias_error"]),
        "bias_rel_error": float(reward["bias_rel_error"]),
        "reason": str(reward["reason"] or metrics.get("reason", "")),
    }
    for key in ("T_X", "T_Z", "bias", "geo_lifetime", "fit_penalty", "fit_ok", "valid"):
        row[key] = metrics.get(key, np.nan)
    row.update(features)
    return row


def select_better_physical(candidate: dict, incumbent: dict) -> bool:
    c_feasible = bool(candidate["is_feasible"])
    i_feasible = bool(incumbent["is_feasible"])
    if c_feasible and not i_feasible:
        return True
    if i_feasible and not c_feasible:
        return False
    if c_feasible and i_feasible:
        c_error = float(candidate["bias_error"])
        i_error = float(incumbent["bias_error"])
        if abs(c_error - i_error) > math.log(1.006):
            return c_error < i_error
        return float(candidate["geo_lifetime"]) > float(incumbent["geo_lifetime"])
    return float(candidate["reward_safe"]) > float(incumbent["reward_safe"])


def history_row(run_id: str, epoch: int, observed: dict, incumbent: dict) -> dict:
    row: Dict[str, object] = {"run_id": run_id, "epoch": int(epoch)}
    for prefix, item in (("observed", observed), ("incumbent", incumbent)):
        row[f"{prefix}_reward"] = float(item["reward"])
        row[f"{prefix}_reward_safe"] = float(item["reward_safe"])
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


def run_physical_reward_optimizer(
    *,
    verbose: bool = True,
    opt_cfg: PhysicalOptimizerConfig | None = None,
    reward_cfg: ImprovedRewardConfig | None = None,
    bounds: PhysicalBounds | None = None,
) -> dict:
    opt_cfg = opt_cfg or PhysicalOptimizerConfig()
    reward_cfg = reward_cfg or ImprovedRewardConfig()
    bounds = bounds or PhysicalBounds()
    sim_cfg = TwoPointConfig()
    noise_cfg = NoiseConfig(sigma=opt_cfg.noise_sigma, seed=opt_cfg.noise_seed)
    rng = np.random.default_rng(opt_cfg.noise_seed)
    clear_measure_cache()

    v0 = clip_physical(
        raw_to_physical(
            OPTIMIZATION_START_X.copy(),
            sim_cfg,
            use_alpha_correction=opt_cfg.use_alpha_correction,
        ),
        bounds,
    )
    optimizer = SepCMA(
        mean=v0.copy(),
        sigma=float(opt_cfg.sigma0),
        bounds=bounds.as_array(),
        population_size=int(opt_cfg.population),
        seed=int(opt_cfg.optimizer_seed),
    )
    start_eval = evaluate_candidate_physical(
        v0,
        sim_cfg=sim_cfg,
        noise_cfg=noise_cfg,
        reward_cfg=reward_cfg,
        bounds=bounds,
        rng=rng,
        use_alpha_correction=opt_cfg.use_alpha_correction,
    )
    incumbent = dict(start_eval)
    run_id = f"step04A_physical_s{opt_cfg.sigma0:g}_seed{opt_cfg.optimizer_seed}"
    history = [history_row(run_id, 0, start_eval, incumbent)]
    start_time = time.time()
    for epoch in range(1, opt_cfg.epochs + 1):
        candidates = [np.asarray(optimizer.ask(), dtype=float) for _ in range(opt_cfg.population)]
        if len(candidates) >= 2:
            candidates[0] = np.asarray(incumbent["v"], dtype=float)
            candidates[1] = v0.copy()
        evaluated = [
            evaluate_candidate_physical(
                cand,
                sim_cfg=sim_cfg,
                noise_cfg=noise_cfg,
                reward_cfg=reward_cfg,
                bounds=bounds,
                rng=rng,
                use_alpha_correction=opt_cfg.use_alpha_correction,
            )
            for cand in candidates
        ]
        optimizer.tell([(np.asarray(item["v"], dtype=float), -float(item["reward_safe"])) for item in evaluated])
        observed = max(evaluated, key=lambda item: float(item["reward_safe"]))
        for item in evaluated:
            if select_better_physical(item, incumbent):
                incumbent = dict(item)
        history.append(history_row(run_id, epoch, observed, incumbent))
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == opt_cfg.epochs):
            elapsed = time.time() - start_time
            print(
                f"{run_id} epoch={epoch:03d}/{opt_cfg.epochs} "
                f"incumbent Tx={float(incumbent['T_X']):.4g} "
                f"Tz={float(incumbent['T_Z']):.4g} "
                f"bias={float(incumbent['bias']):.4g} "
                f"reward={float(incumbent['reward_safe']):.3f} "
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
        "reward_config": asdict(reward_cfg),
        "optimizer_config": asdict(opt_cfg),
        "physical_bounds": asdict(bounds),
    }
