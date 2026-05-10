"""Dynamiqs cat-qubit model and robust lifetime extraction.

The Hamiltonian, losses, logical states, and observables follow the challenge
notebook.  Rates are in MHz-like inverse microseconds, so fitted lifetimes are
reported in microseconds.
"""

from __future__ import annotations

import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

warnings.filterwarnings(
    "ignore",
    message="A `SparseDIAQArray` has been converted to a `DenseQArray`.*",
)

EPS = 1e-12


@dataclass(frozen=True)
class SimulationConfig:
    na: int = 15
    nb: int = 5
    kappa_b: float = 10.0
    kappa_a: float = 1.0
    t_final_x: float = 1.2
    t_final_z: float = 220.0
    n_points: int = 60
    max_tau_factor: float = 80.0
    alpha_margin: float = 2.5


def params_to_complex(x: np.ndarray) -> Tuple[complex, complex]:
    """Map four real optimizer knobs to complex g2 and epsilon_d."""

    x = np.asarray(x, dtype=float)
    if x.shape != (4,):
        raise ValueError("x must be [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]")
    return complex(x[0], x[1]), complex(x[2], x[3])


def complex_to_params(g2: complex, eps_d: complex) -> np.ndarray:
    return np.array([g2.real, g2.imag, eps_d.real, eps_d.imag], dtype=float)


def estimate_alpha(
    g2: complex,
    eps_d: complex,
    *,
    kappa_b: float = 10.0,
    kappa_a: float = 1.0,
) -> complex:
    """Adiabatic-elimination cat-size estimate used by the notebook."""

    kappa_2 = 4.0 * abs(g2) ** 2 / kappa_b
    if kappa_2 <= EPS:
        return complex(np.nan, np.nan)
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    alpha_sq = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
    return complex(np.sqrt(alpha_sq + 0.0j))


def truncation_penalty(alpha_abs: float, na: int, *, margin: float = 2.5) -> float:
    """Soft penalty when the coherent lobes approach the Fock cutoff."""

    if not np.isfinite(alpha_abs):
        return 1.0e3
    nbar = alpha_abs**2
    usable = max(1.0, float(na) - margin)
    if nbar <= usable:
        return 0.0
    return float(((nbar - usable) / usable) ** 2 * 100.0)


def _exp_model(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    amp, tau, offset = params
    return amp * np.exp(-t / tau) + offset


def robust_exp_fit(
    times: np.ndarray,
    values: np.ndarray,
    *,
    max_tau: float,
    f_scale: float = 0.05,
) -> Dict[str, object]:
    """Robustly fit y = A exp(-t / tau) + C using scipy least_squares."""

    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if len(t) < 4 or len(t) != len(y) or not np.all(np.isfinite(y)):
        return {
            "success": False,
            "reason": "bad input",
            "params": np.array([np.nan, np.nan, np.nan]),
            "fit": np.full_like(t, np.nan, dtype=float),
            "r2": np.nan,
            "rmse": np.nan,
            "hit_tau_bound": True,
        }

    amp0 = float(max(1e-4, y[0] - y[-1], np.max(y) - np.min(y)))
    tau0 = float(max(1e-3, 0.5 * (t[-1] - t[0])))
    offset0 = float(y[-1])

    def residual(params: np.ndarray) -> np.ndarray:
        return _exp_model(params, t) - y

    try:
        res = least_squares(
            residual,
            np.array([amp0, tau0, offset0], dtype=float),
            bounds=([0.0, 1e-6, -2.0], [3.0, max_tau, 2.0]),
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=600,
        )
    except Exception as exc:  # pragma: no cover - defensive around scipy.
        return {
            "success": False,
            "reason": f"least_squares failed: {exc}",
            "params": np.array([np.nan, np.nan, np.nan]),
            "fit": np.full_like(t, np.nan, dtype=float),
            "r2": np.nan,
            "rmse": np.nan,
            "hit_tau_bound": True,
        }

    fit = _exp_model(res.x, t)
    residuals = fit - y
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > EPS else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    hit_tau_bound = bool(res.x[1] > 0.98 * max_tau)
    success = bool(res.success and np.isfinite(res.x[1]) and res.x[1] > 0 and not hit_tau_bound)
    return {
        "success": success,
        "reason": "" if success else str(res.message),
        "params": np.asarray(res.x, dtype=float),
        "fit": np.asarray(fit, dtype=float),
        "r2": r2,
        "rmse": rmse,
        "hit_tau_bound": hit_tau_bound,
        "cost": float(res.cost),
    }


_MEASURE_CACHE: Dict[Tuple[float, ...], Dict[str, object]] = {}


def clear_measure_cache() -> None:
    _MEASURE_CACHE.clear()


def _cache_key(g2: complex, eps_d: complex, cfg: SimulationConfig) -> Tuple[float, ...]:
    return (
        round(float(g2.real), 10),
        round(float(g2.imag), 10),
        round(float(eps_d.real), 10),
        round(float(eps_d.imag), 10),
        float(cfg.na),
        float(cfg.nb),
        float(cfg.kappa_b),
        float(cfg.kappa_a),
        float(cfg.t_final_x),
        float(cfg.t_final_z),
        float(cfg.n_points),
    )


def measure_lifetimes(
    g2: complex,
    eps_d: complex,
    cfg: SimulationConfig,
    *,
    use_cache: bool = True,
    return_curves: bool = False,
) -> Dict[str, object]:
    """Return T_X, T_Z, bias, fit diagnostics, and optionally raw decay curves."""

    key = _cache_key(g2, eps_d, cfg)
    if use_cache and key in _MEASURE_CACHE:
        cached = dict(_MEASURE_CACHE[key])
        if return_curves and "curves" in cached:
            return cached
        if not return_curves:
            return {k: v for k, v in cached.items() if k != "curves"}

    alpha = estimate_alpha(g2, eps_d, kappa_b=cfg.kappa_b, kappa_a=cfg.kappa_a)
    alpha_abs = float(abs(alpha)) if np.isfinite(abs(alpha)) else np.nan
    alpha_penalty = truncation_penalty(alpha_abs, cfg.na, margin=cfg.alpha_margin)
    if not np.isfinite(alpha_abs) or alpha_abs < 0.20:
        return _invalid_result("invalid alpha", alpha_abs=alpha_abs, cfg=cfg)

    na, nb = int(cfg.na), int(cfg.nb)
    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    # Challenge notebook convention:
    # H = conj(g2) a^2 b^dag + g2 (a^dag)^2 b - eps_d b^dag - conj(eps_d) b.
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
    ts_x = jnp.linspace(0.0, float(cfg.t_final_x), int(cfg.n_points))
    ts_z = jnp.linspace(0.0, float(cfg.t_final_z), int(cfg.n_points))
    options = dq.Options(progress_meter=False)

    try:
        res_x = dq.mesolve(
            H,
            [loss_b, loss_a],
            psi_x,
            ts_x,
            options=options,
            exp_ops=[sx],
        )
        res_z = dq.mesolve(
            H,
            [loss_b, loss_a],
            psi_z,
            ts_z,
            options=options,
            exp_ops=[sz],
        )
    except Exception as exc:
        return _invalid_result(f"mesolve failed: {exc}", alpha_abs=alpha_abs, cfg=cfg)

    x_curve = np.asarray(res_x.expects[0].real, dtype=float)
    z_curve = np.asarray(res_z.expects[0].real, dtype=float)
    times_x = np.asarray(ts_x, dtype=float)
    times_z = np.asarray(ts_z, dtype=float)
    fit_x = robust_exp_fit(
        times_x,
        x_curve,
        max_tau=max(100.0, cfg.max_tau_factor * float(cfg.t_final_x)),
    )
    fit_z = robust_exp_fit(
        times_z,
        z_curve,
        max_tau=max(1000.0, cfg.max_tau_factor * float(cfg.t_final_z)),
    )

    T_X = float(fit_x["params"][1])
    T_Z = float(fit_z["params"][1])
    fit_ok = bool(fit_x["success"] and fit_z["success"])
    finite = bool(np.isfinite(T_X) and np.isfinite(T_Z) and T_X > 0 and T_Z > 0)
    bias = float(T_Z / T_X) if finite else np.nan
    fit_penalty = _fit_penalty(fit_x, fit_z) + alpha_penalty
    result: Dict[str, object] = {
        "g2_real": float(g2.real),
        "g2_imag": float(g2.imag),
        "eps_d_real": float(eps_d.real),
        "eps_d_imag": float(eps_d.imag),
        "T_X": T_X,
        "T_Z": T_Z,
        "bias": bias,
        "geo_lifetime": float(math.sqrt(T_X * T_Z)) if finite else np.nan,
        "alpha_abs": alpha_abs,
        "nbar": float(alpha_abs**2),
        "fit_ok": fit_ok,
        "fit_penalty": float(fit_penalty),
        "fit_x_r2": float(fit_x["r2"]),
        "fit_z_r2": float(fit_z["r2"]),
        "fit_x_rmse": float(fit_x["rmse"]),
        "fit_z_rmse": float(fit_z["rmse"]),
        "fit_x_hit_tau_bound": bool(fit_x["hit_tau_bound"]),
        "fit_z_hit_tau_bound": bool(fit_z["hit_tau_bound"]),
        "valid": bool(fit_ok and finite and fit_penalty < 10.0),
        "reason": "" if fit_ok else "fit failed",
    }
    if return_curves:
        result["curves"] = {
            "times_x": times_x,
            "values_x": x_curve,
            "fit_x": np.asarray(fit_x["fit"], dtype=float),
            "times_z": times_z,
            "values_z": z_curve,
            "fit_z": np.asarray(fit_z["fit"], dtype=float),
        }

    _MEASURE_CACHE[key] = dict(result)
    if return_curves:
        return result
    return {k: v for k, v in result.items() if k != "curves"}


def _fit_penalty(fit_x: Dict[str, object], fit_z: Dict[str, object]) -> float:
    penalty = 0.0
    for fit in (fit_x, fit_z):
        if not fit["success"]:
            penalty += 5.0
        r2 = float(fit["r2"]) if np.isfinite(float(fit["r2"])) else -np.inf
        if r2 < 0.995:
            penalty += (0.995 - max(r2, 0.0)) * 20.0
        rmse = float(fit["rmse"]) if np.isfinite(float(fit["rmse"])) else 1.0
        if rmse > 0.01:
            penalty += (rmse - 0.01) * 20.0
        if bool(fit["hit_tau_bound"]):
            penalty += 10.0
    return float(penalty)


def _invalid_result(reason: str, *, alpha_abs: float, cfg: SimulationConfig) -> Dict[str, object]:
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
