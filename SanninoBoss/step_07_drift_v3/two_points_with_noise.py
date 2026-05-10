"""Normal noisy two-point cat-qubit bias optimization for step 4 comparisons.

This file is self-contained: it carries the physical model, the two-point
lifetime estimator, the reward, and the CMA-ES optimization loop used for
the noisy two-point run.  The noisy step still samples only
t=0 and one later time tau; it perturbs only the measured tau expectation.
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

sys.dont_write_bytecode = True

STEP_DIR = Path(__file__).resolve().parent
CACHE_ROOT = Path(tempfile.gettempdir()) / "sannino_step07_drift_physical_cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))

import qutip.settings as qutip_settings

qutip_settings.tmproot = str(CACHE_ROOT)
qutip_settings.coeffroot = str(CACHE_ROOT / "qutip_coeffs_1.3")

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from cmaes import SepCMA

warnings.filterwarnings(
    "ignore",
    message="A `SparseDIAQArray` has been converted to a `DenseQArray`.*",
)

EPS = 1e-12

TARGET_BIAS = 100.0
BIAS_TOL_REL = 0.03
CHALLENGE_BASELINE_X = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
OPTIMIZATION_START_X = np.array([1.0, 0.0, 2.5, 0.0], dtype=float)
BOUNDS = np.array(
    [
        [0.25, 3.0],
        [-1.0, 1.0],
        [0.50, 8.0],
        [-3.0, 3.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TwoPointConfig:
    na: int = 15
    nb: int = 5
    kappa_b: float = 10.0
    kappa_a: float = 1.0
    tau_x: float = 0.04067796841263771
    tau_z: float = 22.033899307250977
    alpha_margin: float = 2.5


@dataclass(frozen=True)
class RewardConfig:
    name: str = "exact_target_strict"
    variant: str = "exact_target"
    target_bias: float = TARGET_BIAS
    bias_tol_rel: float = BIAS_TOL_REL
    w_lifetime: float = 0.25
    w_bias_exact: float = 180.0
    w_fit: float = 2.0
    feasibility_bonus: float = 0.0
    min_tx: float = 0.05
    min_tz: float = 5.0
    floor_weight: float = 12.0


@dataclass(frozen=True)
class NoiseConfig:
    model: str = "additive_gaussian_tau_expectation"
    sigma: float = 0.02
    seed: int = 11


def params_to_complex(x: np.ndarray) -> Tuple[complex, complex]:
    x = np.asarray(x, dtype=float)
    if x.shape != (4,):
        raise ValueError("x must be [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]")
    return complex(x[0], x[1]), complex(x[2], x[3])


def estimate_alpha(
    g2: complex,
    eps_d: complex,
    *,
    kappa_b: float = 10.0,
    kappa_a: float = 1.0,
) -> complex:
    kappa_2 = 4.0 * abs(g2) ** 2 / kappa_b
    if kappa_2 <= EPS:
        return complex(np.nan, np.nan)
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    alpha_sq = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
    return complex(np.sqrt(alpha_sq + 0.0j))


def truncation_penalty(alpha_abs: float, na: int, *, margin: float = 2.5) -> float:
    if not np.isfinite(alpha_abs):
        return 1.0e3
    nbar = alpha_abs**2
    usable = max(1.0, float(na) - margin)
    if nbar <= usable:
        return 0.0
    return float(((nbar - usable) / usable) ** 2 * 100.0)


_MEASURE_CACHE: Dict[Tuple[float, ...], Dict[str, object]] = {}


def clear_measure_cache() -> None:
    _MEASURE_CACHE.clear()


def cleanup_local_qutip_cache() -> None:
    """Remove the empty qutip coefficient folder that qutip may create in cwd."""

    local_cache = Path.cwd() / "qutip_coeffs_1.3"
    if local_cache.exists() and local_cache.is_dir():
        shutil.rmtree(local_cache)


def _cache_key(g2: complex, eps_d: complex, cfg: TwoPointConfig) -> Tuple[float, ...]:
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


def two_point_tau(y0: float, y_tau: float, tau: float) -> Tuple[float, float, bool, str]:
    """Estimate T from y(t)=y(0) exp(-t/T) using exactly two samples."""

    if not np.isfinite(y0) or not np.isfinite(y_tau) or tau <= 0:
        return np.nan, np.inf, False, "non-finite two-point input"
    if abs(y0) <= EPS:
        return np.nan, np.inf, False, "zero initial expectation"
    ratio = abs(float(y_tau) / float(y0))
    if not np.isfinite(ratio) or ratio <= 0.0 or ratio >= 1.0:
        return np.nan, np.inf, False, f"invalid decay ratio {ratio:g}"
    T = -float(tau) / math.log(ratio)
    if not np.isfinite(T) or T <= 0:
        return np.nan, np.inf, False, "invalid lifetime"

    # Small, smooth diagnostic penalty: ratios very close to 0 or 1 are valid
    # but less stable as a two-point numerical measurement.
    penalty = 0.0
    if ratio < 0.03:
        penalty += (0.03 - ratio) * 20.0
    if ratio > 0.98:
        penalty += (ratio - 0.98) * 200.0
    return float(T), float(penalty), True, ""


def measure_lifetimes_two_point(
    g2: complex,
    eps_d: complex,
    cfg: TwoPointConfig,
    *,
    use_cache: bool = True,
    return_points: bool = False,
) -> Dict[str, object]:
    key = _cache_key(g2, eps_d, cfg)
    if use_cache and key in _MEASURE_CACHE:
        cached = dict(_MEASURE_CACHE[key])
        if return_points and "points" in cached:
            return cached
        if not return_points:
            return {k: v for k, v in cached.items() if k != "points"}

    alpha = estimate_alpha(g2, eps_d, kappa_b=cfg.kappa_b, kappa_a=cfg.kappa_a)
    alpha_abs = float(abs(alpha)) if np.isfinite(abs(alpha)) else np.nan
    alpha_penalty = truncation_penalty(alpha_abs, cfg.na, margin=cfg.alpha_margin)
    if not np.isfinite(alpha_abs) or alpha_abs < 0.20:
        return _invalid_result("invalid alpha", alpha_abs=alpha_abs, cfg=cfg)

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
    ts_x = jnp.asarray([0.0, float(cfg.tau_x)])
    ts_z = jnp.asarray([0.0, float(cfg.tau_z)])
    options = dq.Options(progress_meter=False)

    try:
        res_x = dq.mesolve(H, [loss_b, loss_a], psi_x, ts_x, options=options, exp_ops=[sx])
        res_z = dq.mesolve(H, [loss_b, loss_a], psi_z, ts_z, options=options, exp_ops=[sz])
    except Exception as exc:
        return _invalid_result(f"mesolve failed: {exc}", alpha_abs=alpha_abs, cfg=cfg)

    x_values = np.asarray(res_x.expects[0].real, dtype=float)
    z_values = np.asarray(res_z.expects[0].real, dtype=float)
    T_X, penalty_x, ok_x, reason_x = two_point_tau(x_values[0], x_values[1], cfg.tau_x)
    T_Z, penalty_z, ok_z, reason_z = two_point_tau(z_values[0], z_values[1], cfg.tau_z)

    fit_ok = bool(ok_x and ok_z)
    finite = bool(np.isfinite(T_X) and np.isfinite(T_Z) and T_X > 0 and T_Z > 0)
    bias = float(T_Z / T_X) if finite else np.nan
    fit_penalty = float(penalty_x + penalty_z + alpha_penalty)
    result: Dict[str, object] = {
        "g2_real": float(g2.real),
        "g2_imag": float(g2.imag),
        "eps_d_real": float(eps_d.real),
        "eps_d_imag": float(eps_d.imag),
        "T_X": float(T_X),
        "T_Z": float(T_Z),
        "bias": bias,
        "geo_lifetime": float(math.sqrt(T_X * T_Z)) if finite else np.nan,
        "alpha_abs": alpha_abs,
        "nbar": float(alpha_abs**2),
        "fit_ok": fit_ok,
        "fit_penalty": fit_penalty,
        "fit_x_r2": 1.0 if ok_x else np.nan,
        "fit_z_r2": 1.0 if ok_z else np.nan,
        "fit_x_rmse": 0.0 if ok_x else np.nan,
        "fit_z_rmse": 0.0 if ok_z else np.nan,
        "fit_x_hit_tau_bound": False,
        "fit_z_hit_tau_bound": False,
        "valid": bool(fit_ok and finite and fit_penalty < 10.0),
        "reason": "" if fit_ok else "; ".join(r for r in (reason_x, reason_z) if r),
    }
    if return_points:
        result["points"] = {
            "times_x": [0.0, float(cfg.tau_x)],
            "values_x": [float(x_values[0]), float(x_values[1])],
            "times_z": [0.0, float(cfg.tau_z)],
            "values_z": [float(z_values[0]), float(z_values[1])],
        }

    _MEASURE_CACHE[key] = dict(result)
    if return_points:
        return result
    return {k: v for k, v in result.items() if k != "points"}


def measure_lifetimes_two_point_noisy(
    g2: complex,
    eps_d: complex,
    cfg: TwoPointConfig,
    noise_cfg: NoiseConfig,
    rng: np.random.Generator,
    *,
    return_points: bool = False,
) -> Dict[str, object]:
    """Measure the two tau points, then add Gaussian readout noise at tau.

    The underlying Lindblad evolution is the same as the no-noise two-point
    routine.  Noise is introduced only as a measurement perturbation on the
    second sample of each observable.
    """

    exact = measure_lifetimes_two_point(
        g2,
        eps_d,
        cfg,
        use_cache=True,
        return_points=True,
    )
    if "points" not in exact:
        return exact

    points = exact["points"]
    x_values = np.asarray(points["values_x"], dtype=float).copy()
    z_values = np.asarray(points["values_z"], dtype=float).copy()

    noise_x = float(rng.normal(0.0, noise_cfg.sigma))
    noise_z = float(rng.normal(0.0, noise_cfg.sigma))
    x_values_noisy = x_values.copy()
    z_values_noisy = z_values.copy()
    x_values_noisy[1] = float(np.clip(x_values_noisy[1] + noise_x, -1.0, 1.0))
    z_values_noisy[1] = float(np.clip(z_values_noisy[1] + noise_z, -1.0, 1.0))

    T_X, penalty_x, ok_x, reason_x = two_point_tau(x_values_noisy[0], x_values_noisy[1], cfg.tau_x)
    T_Z, penalty_z, ok_z, reason_z = two_point_tau(z_values_noisy[0], z_values_noisy[1], cfg.tau_z)
    fit_ok = bool(ok_x and ok_z)
    finite = bool(np.isfinite(T_X) and np.isfinite(T_Z) and T_X > 0 and T_Z > 0)
    bias = float(T_Z / T_X) if finite else np.nan
    alpha_abs = float(exact.get("alpha_abs", np.nan))
    alpha_penalty = truncation_penalty(alpha_abs, cfg.na, margin=cfg.alpha_margin)
    fit_penalty = float(penalty_x + penalty_z + alpha_penalty)

    result = dict(exact)
    result.update(
        {
            "T_X": float(T_X),
            "T_Z": float(T_Z),
            "bias": bias,
            "geo_lifetime": float(math.sqrt(T_X * T_Z)) if finite else np.nan,
            "fit_ok": fit_ok,
            "fit_penalty": fit_penalty,
            "fit_x_r2": 1.0 if ok_x else np.nan,
            "fit_z_r2": 1.0 if ok_z else np.nan,
            "fit_x_rmse": float(abs(x_values_noisy[1] - x_values[1])),
            "fit_z_rmse": float(abs(z_values_noisy[1] - z_values[1])),
            "valid": bool(fit_ok and finite and fit_penalty < 10.0),
            "reason": "" if fit_ok else "; ".join(r for r in (reason_x, reason_z) if r),
            "noise_model": noise_cfg.model,
            "noise_sigma": float(noise_cfg.sigma),
            "noise_x_tau": noise_x,
            "noise_z_tau": noise_z,
        }
    )
    result["points"] = {
        "times_x": [0.0, float(cfg.tau_x)],
        "values_x_exact": [float(x_values[0]), float(x_values[1])],
        "values_x_noisy": [float(x_values_noisy[0]), float(x_values_noisy[1])],
        "times_z": [0.0, float(cfg.tau_z)],
        "values_z_exact": [float(z_values[0]), float(z_values[1])],
        "values_z_noisy": [float(z_values_noisy[0]), float(z_values_noisy[1])],
    }
    if return_points:
        return result
    return {k: v for k, v in result.items() if k != "points"}


def _invalid_result(reason: str, *, alpha_abs: float, cfg: TwoPointConfig) -> Dict[str, object]:
    penalty = 1.0e3 + truncation_penalty(alpha_abs, cfg.na, margin=cfg.alpha_margin)
    return {
        "g2_real": np.nan,
        "g2_imag": np.nan,
        "eps_d_real": np.nan,
        "eps_d_imag": np.nan,
        "T_X": np.nan,
        "T_Z": np.nan,
        "bias": np.nan,
        "geo_lifetime": np.nan,
        "alpha_abs": float(alpha_abs),
        "nbar": float(alpha_abs**2) if np.isfinite(alpha_abs) else np.nan,
        "fit_ok": False,
        "fit_penalty": float(penalty),
        "fit_x_r2": np.nan,
        "fit_z_r2": np.nan,
        "fit_x_rmse": np.nan,
        "fit_z_rmse": np.nan,
        "fit_x_hit_tau_bound": True,
        "fit_z_hit_tau_bound": True,
        "valid": False,
        "reason": reason,
    }


def compute_reward(metrics: Dict[str, object], cfg: RewardConfig) -> Dict[str, float | bool]:
    T_X = float(metrics.get("T_X", np.nan))
    T_Z = float(metrics.get("T_Z", np.nan))
    bias = float(metrics.get("bias", np.nan))
    fit_penalty = float(metrics.get("fit_penalty", 1.0e3))
    valid = bool(metrics.get("valid", False))
    finite = bool(np.isfinite(T_X) and np.isfinite(T_Z) and np.isfinite(bias))
    if not finite or T_X <= 0 or T_Z <= 0 or bias <= 0:
        return {
            "reward": -1.0e9,
            "loss_to_minimize": 1.0e9,
            "is_feasible": False,
            "bias_error": np.inf,
            "bias_rel_error": np.inf,
            "floor_penalty": 1.0e6,
        }

    log_eta = math.log(bias)
    log_target = math.log(cfg.target_bias)
    bias_error = abs(log_eta - log_target)
    bias_rel_error = abs(bias / cfg.target_bias - 1.0)
    lifetime_score = 0.5 * (math.log(T_X) + math.log(T_Z))
    is_within_band = bool(bias_rel_error <= cfg.bias_tol_rel)
    is_feasible = bool(valid and is_within_band)
    floor_penalty = cfg.floor_weight * (
        max(0.0, math.log(cfg.min_tx / T_X)) ** 2
        + max(0.0, math.log(cfg.min_tz / T_Z)) ** 2
    )
    reward = (
        cfg.feasibility_bonus * float(is_within_band)
        + cfg.w_lifetime * lifetime_score
        - cfg.w_bias_exact * bias_error**2
        - cfg.w_fit * fit_penalty
        - floor_penalty
    )
    return {
        "reward": float(reward),
        "loss_to_minimize": -float(reward),
        "is_feasible": is_feasible,
        "bias_error": float(bias_error),
        "bias_rel_error": float(bias_rel_error),
        "floor_penalty": float(floor_penalty),
    }


def optimizer_ask(opt: SepCMA) -> list[np.ndarray]:
    return [np.asarray(opt.ask(), dtype=float) for _ in range(opt.population_size)]


def select_better(candidate: dict, incumbent: dict) -> bool:
    c_feasible = bool(candidate["is_feasible"])
    i_feasible = bool(incumbent["is_feasible"])
    if c_feasible and not i_feasible:
        return True
    if i_feasible and not c_feasible:
        return False
    if c_feasible and i_feasible:
        c_error = float(candidate["bias_error"])
        i_error = float(incumbent["bias_error"])
        if abs(c_error - i_error) > math.log(1.01):
            return c_error < i_error
        return float(candidate["geo_lifetime"]) > float(incumbent["geo_lifetime"])
    return float(candidate["reward"]) > float(incumbent["reward"])


def evaluate_x(
    x: np.ndarray,
    sim_cfg: TwoPointConfig,
    reward_cfg: RewardConfig,
    *,
    noise_cfg: NoiseConfig | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    x = np.minimum(np.maximum(np.asarray(x, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
    g2, eps_d = params_to_complex(x)
    if noise_cfg is None or noise_cfg.sigma <= 0.0:
        metrics = measure_lifetimes_two_point(g2, eps_d, sim_cfg)
    else:
        if rng is None:
            raise ValueError("rng is required when noise_cfg is provided")
        metrics = measure_lifetimes_two_point_noisy(g2, eps_d, sim_cfg, noise_cfg, rng)
    reward = compute_reward(metrics, reward_cfg)
    row = {
        "x": x,
        "g2": g2,
        "eps_d": eps_d,
        "reward": float(reward["reward"]),
        "loss_to_minimize": float(reward["loss_to_minimize"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_error": float(reward["bias_error"]),
        "bias_rel_error": float(reward["bias_rel_error"]),
    }
    for key in (
        "T_X",
        "T_Z",
        "bias",
        "geo_lifetime",
        "alpha_abs",
        "nbar",
        "fit_penalty",
        "fit_x_r2",
        "fit_z_r2",
        "fit_ok",
        "valid",
    ):
        row[key] = metrics[key]
    return row


def _history_row(
    run_id: str,
    epoch: int,
    epoch_best: dict,
    mean_eval: dict,
    incumbent: dict,
    optimizer_mean: np.ndarray,
    reward_cfg: RewardConfig,
    sigma0: float,
    seed: int,
) -> dict:
    row: Dict[str, object] = {
        "run_id": run_id,
        "epoch": int(epoch),
        "reward_config": reward_cfg.name,
        "reward_variant": reward_cfg.variant,
        "sigma0": float(sigma0),
        "seed": int(seed),
    }
    for prefix, item in (("epoch_best", epoch_best), ("mean", mean_eval), ("incumbent", incumbent)):
        row[f"{prefix}_reward"] = float(item["reward"])
        row[f"{prefix}_loss"] = float(item["loss_to_minimize"])
        row[f"{prefix}_T_X"] = float(item["T_X"])
        row[f"{prefix}_T_Z"] = float(item["T_Z"])
        row[f"{prefix}_bias"] = float(item["bias"])
        row[f"{prefix}_geo_lifetime"] = float(item["geo_lifetime"])
        row[f"{prefix}_fit_penalty"] = float(item["fit_penalty"])
        row[f"{prefix}_is_feasible"] = int(bool(item["is_feasible"]))
        x = np.asarray(item["x"], dtype=float)
        row[f"{prefix}_g2_real"] = float(x[0])
        row[f"{prefix}_g2_imag"] = float(x[1])
        row[f"{prefix}_eps_d_real"] = float(x[2])
        row[f"{prefix}_eps_d_imag"] = float(x[3])
    for i, value in enumerate(np.asarray(optimizer_mean, dtype=float)):
        row[f"optimizer_mean_x{i}"] = float(value)
    return row


def run_one_optimization(
    *,
    run_id: str,
    sim_cfg: TwoPointConfig,
    reward_cfg: RewardConfig,
    epochs: int,
    population: int,
    sigma0: float,
    seed: int,
    noise_cfg: NoiseConfig | None = None,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    clear_measure_cache()
    noise_rng = np.random.default_rng(noise_cfg.seed if noise_cfg is not None else seed)
    optimizer = SepCMA(
        mean=OPTIMIZATION_START_X.copy(),
        sigma=float(sigma0),
        bounds=BOUNDS,
        population_size=int(population),
        seed=int(seed),
    )
    start_eval = evaluate_x(OPTIMIZATION_START_X.copy(), sim_cfg, reward_cfg, noise_cfg=noise_cfg, rng=noise_rng)
    incumbent = dict(start_eval)
    history: list[dict] = [
        _history_row(run_id, 0, start_eval, start_eval, incumbent, optimizer.mean, reward_cfg, sigma0, seed)
    ]
    start = time.time()
    for epoch in range(1, epochs + 1):
        xs = optimizer_ask(optimizer)
        if len(xs) >= 2:
            xs[0] = np.asarray(incumbent["x"], dtype=float)
            xs[1] = OPTIMIZATION_START_X.copy()

        evaluated = [evaluate_x(x, sim_cfg, reward_cfg, noise_cfg=noise_cfg, rng=noise_rng) for x in xs]
        optimizer.tell([(np.asarray(item["x"], dtype=float), float(item["loss_to_minimize"])) for item in evaluated])

        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        for item in evaluated:
            if select_better(item, incumbent):
                incumbent = dict(item)

        mean_eval = evaluate_x(
            np.asarray(optimizer.mean, dtype=float),
            sim_cfg,
            reward_cfg,
            noise_cfg=noise_cfg,
            rng=noise_rng,
        )
        row = _history_row(run_id, epoch, epoch_best, mean_eval, incumbent, optimizer.mean, reward_cfg, sigma0, seed)
        history.append(row)
        if verbose and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            elapsed = time.time() - start
            print(
                f"{run_id} epoch={epoch:03d}/{epochs} "
                f"incumbent Tx={row['incumbent_T_X']:.4g} Tz={row['incumbent_T_Z']:.4g} "
                f"bias={row['incumbent_bias']:.4g} reward={row['incumbent_reward']:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return history, incumbent


def run_step2(*, verbose: bool = True) -> dict:
    sim_cfg = TwoPointConfig()
    reward_cfg = RewardConfig()
    epochs = 34
    population = 8
    sigma0 = 0.45
    seed = 0
    run_id = f"step02_two_point_no_noise_s{sigma0:g}_seed{seed}"
    history, incumbent = run_one_optimization(
        run_id=run_id,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        epochs=epochs,
        population=population,
        sigma0=sigma0,
        seed=seed,
        verbose=verbose,
    )
    baseline_g2, baseline_eps = params_to_complex(CHALLENGE_BASELINE_X.copy())
    start_g2, start_eps = params_to_complex(OPTIMIZATION_START_X.copy())
    final_g2, final_eps = params_to_complex(np.asarray(incumbent["x"], dtype=float))
    baseline = measure_lifetimes_two_point(baseline_g2, baseline_eps, sim_cfg, return_points=True)
    start = measure_lifetimes_two_point(start_g2, start_eps, sim_cfg, return_points=True)
    final = measure_lifetimes_two_point(final_g2, final_eps, sim_cfg, return_points=True)
    cleanup_local_qutip_cache()
    return {
        "history": history,
        "incumbent": incumbent,
        "baseline": baseline,
        "start": start,
        "final": final,
        "sim_config": asdict(sim_cfg),
        "reward_config": asdict(reward_cfg),
        "run_config": {
            "run_id": run_id,
            "epochs": epochs,
            "population": population,
            "sigma0": sigma0,
            "seed": seed,
        },
    }


def run_two_points_no_noise(*, verbose: bool = True) -> dict:
    """Step-3 wrapper for the step-2 no-noise two-point run."""

    return run_step2(verbose=verbose)


def run_two_points_with_noise(
    *,
    verbose: bool = True,
    epochs: int = 60,
    population: int = 8,
    sigma0: float = 0.45,
    optimizer_seed: int = 0,
    noise_seed: int = 11,
    noise_sigma: float = 0.02,
) -> dict:
    sim_cfg = TwoPointConfig()
    reward_cfg = RewardConfig()
    noise_cfg = NoiseConfig(sigma=float(noise_sigma), seed=int(noise_seed))
    run_id = (
        f"step03_two_point_noise_sig{noise_sigma:g}_"
        f"optseed{optimizer_seed}_noiseseed{noise_seed}"
    )
    history, incumbent = run_one_optimization(
        run_id=run_id,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        epochs=epochs,
        population=population,
        sigma0=sigma0,
        seed=optimizer_seed,
        noise_cfg=noise_cfg,
        verbose=verbose,
    )
    baseline_g2, baseline_eps = params_to_complex(CHALLENGE_BASELINE_X.copy())
    start_g2, start_eps = params_to_complex(OPTIMIZATION_START_X.copy())
    final_g2, final_eps = params_to_complex(np.asarray(incumbent["x"], dtype=float))
    noise_rng = np.random.default_rng(noise_seed + 1_000_003)
    baseline = measure_lifetimes_two_point_noisy(baseline_g2, baseline_eps, sim_cfg, noise_cfg, noise_rng, return_points=True)
    start = measure_lifetimes_two_point_noisy(start_g2, start_eps, sim_cfg, noise_cfg, noise_rng, return_points=True)
    final = measure_lifetimes_two_point_noisy(final_g2, final_eps, sim_cfg, noise_cfg, noise_rng, return_points=True)
    cleanup_local_qutip_cache()
    return {
        "history": history,
        "incumbent": incumbent,
        "baseline": baseline,
        "start": start,
        "final": final,
        "sim_config": asdict(sim_cfg),
        "reward_config": asdict(reward_cfg),
        "noise_config": asdict(noise_cfg),
        "run_config": {
            "run_id": run_id,
            "epochs": epochs,
            "population": population,
            "sigma0": sigma0,
            "optimizer_seed": optimizer_seed,
            "noise_seed": noise_seed,
        },
    }


if __name__ == "__main__":
    payload = run_two_points_with_noise(verbose=True)
    final = payload["final"]
    print(
        "Final noisy two-point candidate: "
        f"T_X={float(final['T_X']):.6g} us, "
        f"T_Z={float(final['T_Z']):.6g} us, "
        f"bias={float(final['bias']):.6g}"
    )
