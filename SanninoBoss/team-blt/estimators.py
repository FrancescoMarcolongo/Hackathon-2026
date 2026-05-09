"""Reward estimators that use only the lab-like run_experiment primitive."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
from scipy.linalg import logm
from scipy.optimize import least_squares


RunExperiment = Callable[[np.ndarray, str, str, float, int | None], float]
BAD_REWARD = -1.0e6


@dataclass
class RewardConfig:
    eta_target: float = 30.0
    beta: float = 1.0
    chi: float = 0.05
    zeta: float = 0.25
    eps: float = 1e-9
    min_contrast: float = 1e-4
    contrast_target: float = 0.05
    bad_reward: float = BAD_REWARD


@dataclass
class EstimateResult:
    reward: float
    gamma_x: float = np.nan
    gamma_z: float = np.nan
    t_x: float = np.nan
    t_z: float = np.nan
    eta: float = np.nan
    offdiag_penalty: float = 0.0
    contrast_penalty: float = 0.0
    valid: bool = False
    settings: int = 0
    wait_time_cost: float = 0.0
    diagnostics: Dict[str, float | str] = field(default_factory=dict)


def _invalid(
    reason: str,
    *,
    settings: int,
    wait_time_cost: float,
    cfg: RewardConfig,
    diagnostics: Dict[str, float | str] | None = None,
) -> EstimateResult:
    diag = dict(diagnostics or {})
    diag["reason"] = reason
    return EstimateResult(
        reward=cfg.bad_reward,
        valid=False,
        settings=settings,
        wait_time_cost=wait_time_cost,
        diagnostics=diag,
    )


def spectral_reward(
    gamma_x: float,
    gamma_z: float,
    *,
    offdiag_penalty: float,
    contrast_penalty: float,
    cfg: RewardConfig,
) -> EstimateResult:
    vals = np.array([gamma_x, gamma_z, offdiag_penalty, contrast_penalty], dtype=float)
    if not np.all(np.isfinite(vals)):
        return _invalid("non-finite reward inputs", settings=0, wait_time_cost=0.0, cfg=cfg)
    if gamma_x <= 0 or gamma_z <= 0:
        return _invalid("non-positive logical rate", settings=0, wait_time_cost=0.0, cfg=cfg)

    gamma_x = float(max(gamma_x, cfg.eps))
    gamma_z = float(max(gamma_z, cfg.eps))
    eta = gamma_x / gamma_z
    if eta <= 0 or not np.isfinite(eta):
        return _invalid("unstable eta", settings=0, wait_time_cost=0.0, cfg=cfg)

    reward = (
        -np.log(gamma_x + cfg.eps)
        - np.log(gamma_z + cfg.eps)
        - cfg.beta * np.log((eta + cfg.eps) / cfg.eta_target) ** 2
        - cfg.chi * offdiag_penalty
        - cfg.zeta * contrast_penalty
    )
    reward = float(np.clip(reward, cfg.bad_reward, 1.0e6))
    return EstimateResult(
        reward=reward,
        gamma_x=gamma_x,
        gamma_z=gamma_z,
        t_x=1.0 / gamma_x,
        t_z=1.0 / gamma_z,
        eta=eta,
        offdiag_penalty=float(offdiag_penalty),
        contrast_penalty=float(contrast_penalty),
        valid=True,
    )


def _contrast_penalty(contrasts: List[float], cfg: RewardConfig) -> float:
    penalty = 0.0
    for c in contrasts:
        c_abs = abs(float(c))
        if not np.isfinite(c_abs):
            penalty += 1.0e3
        elif c_abs < cfg.contrast_target:
            penalty += ((cfg.contrast_target - c_abs) / cfg.contrast_target) ** 2
    return float(penalty)


def _two_time_gamma(c0: float, c1: float, dt: float, cfg: RewardConfig) -> tuple[float, str]:
    if not np.all(np.isfinite([c0, c1, dt])):
        return np.nan, "non-finite contrast"
    if dt <= 0:
        return np.nan, "non-positive time interval"
    if c0 <= cfg.min_contrast:
        return np.nan, "initial contrast too small or negative"
    if c1 <= cfg.min_contrast:
        return np.nan, "final contrast too small or negative"

    ratio = abs(c1 / c0)
    if ratio <= 0 or not np.isfinite(ratio):
        return np.nan, "unstable contrast ratio"
    if ratio > 1.25:
        return np.nan, "contrast grew after burn-in"
    gamma = -np.log(min(ratio, 1.0)) / dt
    return float(max(gamma, cfg.eps)), ""


def blt_lite_estimate(
    run_experiment: RunExperiment,
    knobs: np.ndarray,
    *,
    t0: float,
    t1: float,
    cfg: RewardConfig,
    n_shots: int | None = None,
) -> EstimateResult:
    """Two-time BLT-lite estimator using 8 lab-like settings."""

    vals: Dict[str, float] = {}
    for axis, prep_plus, prep_minus in (("x", "+x", "-x"), ("z", "+z", "-z")):
        for t_label, t in (("0", t0), ("1", t1)):
            vals[f"{axis}+{t_label}"] = run_experiment(knobs, prep_plus, axis, t, n_shots)
            vals[f"{axis}-{t_label}"] = run_experiment(knobs, prep_minus, axis, t, n_shots)

    cx0 = 0.5 * (vals["x+0"] - vals["x-0"])
    cx1 = 0.5 * (vals["x+1"] - vals["x-1"])
    cz0 = 0.5 * (vals["z+0"] - vals["z-0"])
    cz1 = 0.5 * (vals["z+1"] - vals["z-1"])
    wait_cost = 4.0 * (float(t0) + float(t1))
    diagnostics = {"C_x_t0": cx0, "C_x_t1": cx1, "C_z_t0": cz0, "C_z_t1": cz1}

    gamma_x, reason_x = _two_time_gamma(cx0, cx1, t1 - t0, cfg)
    gamma_z, reason_z = _two_time_gamma(cz0, cz1, t1 - t0, cfg)
    if reason_x or reason_z:
        return _invalid(
            reason_x or reason_z,
            settings=8,
            wait_time_cost=wait_cost,
            cfg=cfg,
            diagnostics=diagnostics,
        )

    result = spectral_reward(
        gamma_x,
        gamma_z,
        offdiag_penalty=0.0,
        contrast_penalty=_contrast_penalty([cx0, cx1, cz0, cz1], cfg),
        cfg=cfg,
    )
    result.settings = 8
    result.wait_time_cost = wait_cost
    result.diagnostics.update(diagnostics)
    result.diagnostics["estimator"] = "blt_lite"
    return result


def blt_full_estimate(
    run_experiment: RunExperiment,
    knobs: np.ndarray,
    *,
    t0: float,
    t1: float,
    cfg: RewardConfig,
    n_shots: int | None = None,
    rate_mode: str = "diag",
) -> EstimateResult:
    """Boundary Liouvillian Tracking with a 3x3 slow logical propagator."""

    axes = ("x", "y", "z")
    prep_pairs = (("+x", "-x"), ("+y", "-y"), ("+z", "-z"))
    matrices = []
    for t in (t0, t1):
        rmat = np.zeros((3, 3), dtype=float)
        for col, (prep_plus, prep_minus) in enumerate(prep_pairs):
            for row, meas_axis in enumerate(axes):
                plus = run_experiment(knobs, prep_plus, meas_axis, t, n_shots)
                minus = run_experiment(knobs, prep_minus, meas_axis, t, n_shots)
                rmat[row, col] = 0.5 * (plus - minus)
        matrices.append(rmat)
    r0, r1 = matrices
    wait_cost = 18.0 * (float(t0) + float(t1))

    if not np.all(np.isfinite(r0)) or not np.all(np.isfinite(r1)):
        return _invalid("non-finite R matrix", settings=36, wait_time_cost=wait_cost, cfg=cfg)
    col_norms = np.linalg.norm(r0, axis=0)
    if float(np.min(col_norms)) < cfg.min_contrast:
        return _invalid(
            "R(t0) contrast too small",
            settings=36,
            wait_time_cost=wait_cost,
            cfg=cfg,
            diagnostics={"min_R0_col_norm": float(np.min(col_norms))},
        )

    try:
        a_slow = r1 @ np.linalg.pinv(r0, rcond=1e-8)
        m_eff = logm(a_slow) / (float(t1) - float(t0))
    except Exception as exc:  # pragma: no cover - defensive around scipy logm
        return _invalid(
            f"logm failed: {exc}",
            settings=36,
            wait_time_cost=wait_cost,
            cfg=cfg,
        )

    if not np.all(np.isfinite(m_eff)):
        return _invalid("non-finite M_eff", settings=36, wait_time_cost=wait_cost, cfg=cfg)
    gamma_x_diag = float(max(cfg.eps, -np.real(m_eff[0, 0])))
    gamma_z_diag = float(max(cfg.eps, -np.real(m_eff[2, 2])))
    gamma_x_contrast, _ = _two_time_gamma(float(r0[0, 0]), float(r1[0, 0]), t1 - t0, cfg)
    gamma_z_contrast, _ = _two_time_gamma(float(r0[2, 2]), float(r1[2, 2]), t1 - t0, cfg)
    eig_rates = np.sort(np.maximum(cfg.eps, -np.real(np.linalg.eigvals(m_eff))))
    gamma_x_eig = float(eig_rates[-1]) if len(eig_rates) else gamma_x_diag
    gamma_z_eig = float(eig_rates[0]) if len(eig_rates) else gamma_z_diag
    if rate_mode == "diag":
        gamma_x, gamma_z = gamma_x_diag, gamma_z_diag
    elif rate_mode == "contrast":
        gamma_x, gamma_z = float(gamma_x_contrast), float(gamma_z_contrast)
    elif rate_mode == "eigen":
        gamma_x, gamma_z = gamma_x_eig, gamma_z_eig
    else:
        raise ValueError("rate_mode must be 'diag', 'contrast', or 'eigen'")
    offdiag = m_eff.copy()
    np.fill_diagonal(offdiag, 0.0)
    offdiag_penalty = float(np.real(np.sum(np.abs(offdiag) ** 2)))
    # Off-diagonal response entries are supposed to be small; they are handled
    # by offdiag_penalty.  Contrast health should only inspect the three logical
    # boundary columns/diagonal responses.
    contrast_penalty = _contrast_penalty(
        list(np.linalg.norm(r0, axis=0)) + list(np.linalg.norm(r1, axis=0)),
        cfg,
    )

    result = spectral_reward(
        gamma_x,
        gamma_z,
        offdiag_penalty=offdiag_penalty,
        contrast_penalty=contrast_penalty,
        cfg=cfg,
    )
    result.settings = 36
    result.wait_time_cost = wait_cost
    result.diagnostics.update(
        {
            "estimator": "blt_full",
            "rate_mode": rate_mode,
            "cond_R0": float(np.linalg.cond(r0)),
            "min_R0_col_norm": float(np.min(col_norms)),
            "gamma_x_diag": gamma_x_diag,
            "gamma_z_diag": gamma_z_diag,
            "gamma_x_contrast": float(gamma_x_contrast),
            "gamma_z_contrast": float(gamma_z_contrast),
            "gamma_x_eigen": gamma_x_eig,
            "gamma_z_eigen": gamma_z_eig,
        }
    )
    return result


def _exp_model(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    amp, tau, offset = params
    return amp * np.exp(-t / tau) + offset


def robust_exp_fit(t: np.ndarray, y: np.ndarray, *, max_tau: float = 1.0e5) -> Dict[str, np.ndarray]:
    """Soft-L1 exponential fit matching the notebook's baseline spirit."""

    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    amp0 = float(max(1e-3, np.max(y) - np.min(y)))
    offset0 = float(y[-1])
    tau0 = float(max(1e-3, 0.5 * (t[-1] - t[0])))

    def residual(params: np.ndarray) -> np.ndarray:
        return _exp_model(params, t) - y

    res = least_squares(
        residual,
        np.array([amp0, tau0, offset0]),
        bounds=([0.0, 1e-6, -1.5], [2.5, max_tau, 1.5]),
        loss="soft_l1",
        f_scale=0.1,
        max_nfev=400,
    )
    fit = _exp_model(res.x, t)
    resid = fit - y
    sse = float(np.sum(resid**2))
    centered = y - float(np.mean(y))
    sst = float(np.sum(centered**2))
    r2 = float(1.0 - sse / sst) if sst > 1e-14 else np.nan
    rmse = float(np.sqrt(np.mean(resid**2)))
    stderr = np.full(3, np.nan, dtype=float)
    if len(t) > 3 and res.jac is not None:
        try:
            dof = max(1, len(t) - 3)
            cov = np.linalg.pinv(res.jac.T @ res.jac) * (sse / dof)
            stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
        except np.linalg.LinAlgError:
            pass
    return {
        "params": res.x,
        "stderr": stderr,
        "fit": fit,
        "residuals": resid,
        "success": bool(res.success),
        "r2": r2,
        "rmse": rmse,
        "status": int(res.status),
        "message": str(res.message),
    }


def _collect_decay_axis(
    run_experiment: RunExperiment,
    knobs: np.ndarray,
    *,
    axis: str,
    times: np.ndarray,
    style: str,
    n_shots: int | None,
) -> tuple[np.ndarray, int, float]:
    if axis not in ("x", "z"):
        raise ValueError("gold fits are defined for x/z logical axes")
    if style == "challenge":
        prep = "+x" if axis == "x" else "+z"
        values = np.array(
            [run_experiment(knobs, prep, axis, float(t), n_shots) for t in times],
            dtype=float,
        )
        return values, len(times), float(np.sum(times))
    if style == "contrast":
        prep_plus = "+x" if axis == "x" else "+z"
        prep_minus = "-x" if axis == "x" else "-z"
        plus = np.array(
            [run_experiment(knobs, prep_plus, axis, float(t), n_shots) for t in times],
            dtype=float,
        )
        minus = np.array(
            [run_experiment(knobs, prep_minus, axis, float(t), n_shots) for t in times],
            dtype=float,
        )
        return 0.5 * (plus - minus), 2 * len(times), float(2.0 * np.sum(times))
    raise ValueError("style must be 'challenge' or 'contrast'")


def gold_full_fit_details(
    run_experiment: RunExperiment,
    knobs: np.ndarray,
    *,
    k_points: int,
    t_final_x: float,
    t_final_z: float,
    cfg: RewardConfig,
    style: str = "contrast",
    n_shots: int | None = None,
    adaptive: bool = False,
    max_extend: int = 2,
    min_fractional_drop: float = 0.04,
) -> tuple[EstimateResult, Dict[str, object]]:
    """Gold reference with challenge-style or contrast-style exponential fits."""

    total_settings = 0
    total_wait_cost = 0.0
    curves: Dict[str, object] = {"style": style}

    def collect_with_optional_extension(axis: str, t_final: float) -> tuple[np.ndarray, np.ndarray]:
        nonlocal total_settings, total_wait_cost
        current_final = float(t_final)
        values = np.array([], dtype=float)
        times = np.array([], dtype=float)
        for attempt in range(max(1, max_extend + 1 if adaptive else 1)):
            times = np.linspace(0.0, current_final, int(k_points))
            values, settings, wait_cost = _collect_decay_axis(
                run_experiment,
                knobs,
                axis=axis,
                times=times,
                style=style,
                n_shots=n_shots,
            )
            total_settings += int(settings)
            total_wait_cost += float(wait_cost)
            denom = max(abs(float(values[0])), cfg.eps)
            fractional_drop = abs(float(values[-1] - values[0])) / denom
            if not adaptive or fractional_drop >= min_fractional_drop:
                break
            current_final *= 2.0
        curves[f"{axis}_extended_t_final"] = current_final
        return times, values

    times_x, xs = collect_with_optional_extension("x", t_final_x)
    times_z, zs = collect_with_optional_extension("z", t_final_z)
    curves.update({"times_x": times_x, "values_x": xs, "times_z": times_z, "values_z": zs})
    diagnostics: Dict[str, float | str] = {
        "estimator": f"gold_{style}_full_fit",
        "x_start": float(xs[0]) if len(xs) else np.nan,
        "x_end": float(xs[-1]) if len(xs) else np.nan,
        "z_start": float(zs[0]) if len(zs) else np.nan,
        "z_end": float(zs[-1]) if len(zs) else np.nan,
    }

    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(zs)):
        result = _invalid(
            "non-finite decay curve",
            settings=total_settings,
            wait_time_cost=total_wait_cost,
            cfg=cfg,
            diagnostics=diagnostics,
        )
        return result, curves
    try:
        fit_x = robust_exp_fit(times_x, xs, max_tau=max(1.0e3, 50.0 * float(times_x[-1] + cfg.eps)))
        fit_z = robust_exp_fit(times_z, zs, max_tau=max(1.0e3, 50.0 * float(times_z[-1] + cfg.eps)))
    except Exception as exc:
        result = _invalid(
            f"fit failed: {exc}",
            settings=total_settings,
            wait_time_cost=total_wait_cost,
            cfg=cfg,
            diagnostics=diagnostics,
        )
        return result, curves

    curves.update({"fit_x": fit_x, "fit_z": fit_z})
    tau_x = float(fit_x["params"][1])
    tau_z = float(fit_z["params"][1])
    gamma_x = 1.0 / max(tau_x, cfg.eps)
    gamma_z = 1.0 / max(tau_z, cfg.eps)
    fit_ok = bool(fit_x["success"]) and bool(fit_z["success"])
    hit_bound = tau_x > 0.98 * max(1.0e3, 50.0 * float(times_x[-1] + cfg.eps)) or tau_z > 0.98 * max(
        1.0e3, 50.0 * float(times_z[-1] + cfg.eps)
    )
    contrast_penalty = _contrast_penalty(
        [xs[0], xs[-1], zs[0], zs[-1], fit_x["params"][0], fit_z["params"][0]], cfg
    )
    result = spectral_reward(
        gamma_x,
        gamma_z,
        offdiag_penalty=0.0,
        contrast_penalty=contrast_penalty,
        cfg=cfg,
    )
    result.settings = total_settings
    result.wait_time_cost = total_wait_cost
    result.valid = bool(result.valid and fit_ok and not hit_bound)
    if not result.valid and result.reward > cfg.bad_reward:
        result.diagnostics["reason"] = "gold fit flagged"
    result.diagnostics.update(diagnostics)
    result.diagnostics.update(
        {
            "style": style,
            "tau_x": tau_x,
            "tau_z": tau_z,
            "tau_x_stderr": float(fit_x["stderr"][1]),
            "tau_z_stderr": float(fit_z["stderr"][1]),
            "fit_x_success": str(fit_x["success"]),
            "fit_z_success": str(fit_z["success"]),
            "fit_x_r2": float(fit_x["r2"]),
            "fit_z_r2": float(fit_z["r2"]),
            "fit_x_rmse": float(fit_x["rmse"]),
            "fit_z_rmse": float(fit_z["rmse"]),
            "hit_tau_bound": str(hit_bound),
        }
    )
    return result, curves


def gold_full_fit_estimate(
    run_experiment: RunExperiment,
    knobs: np.ndarray,
    *,
    k_points: int,
    t_final_x: float,
    t_final_z: float,
    cfg: RewardConfig,
    style: str = "contrast",
    n_shots: int | None = None,
    adaptive: bool = False,
) -> EstimateResult:
    result, _ = gold_full_fit_details(
        run_experiment,
        knobs,
        k_points=k_points,
        t_final_x=t_final_x,
        t_final_z=t_final_z,
        cfg=cfg,
        style=style,
        n_shots=n_shots,
        adaptive=adaptive,
    )
    return result


def naive_full_fit_estimate(
    run_experiment: RunExperiment,
    knobs: np.ndarray,
    *,
    k_points: int,
    t_final_x: float,
    t_final_z: float,
    cfg: RewardConfig,
    n_shots: int | None = None,
) -> EstimateResult:
    """Naive/gold full decay-curve estimator using +x and +z preparations."""
    result = gold_full_fit_estimate(
        run_experiment,
        knobs,
        k_points=k_points,
        t_final_x=t_final_x,
        t_final_z=t_final_z,
        cfg=cfg,
        style="challenge",
        n_shots=n_shots,
        adaptive=False,
    )
    result.diagnostics["estimator"] = "naive_full_fit"
    return result
