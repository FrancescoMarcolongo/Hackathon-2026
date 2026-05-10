"""Physical coordinate transforms for Version 1 experiments.

Version 0 searches directly in raw controls:

    u = [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]

Version 1 searches in physical-ish coordinates:

    v = [log(kappa_2), log(abs(alpha)), arg_alpha, phi_mismatch]

The simulator and reward still receive raw controls.  This module only changes
the optimizer's coordinate system.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


def sanitize_angle(theta: float | np.ndarray) -> float | np.ndarray:
    """Map angles to [-pi, pi)."""

    return (np.asarray(theta) + np.pi) % (2.0 * np.pi) - np.pi


def _raw_complex(u: np.ndarray) -> tuple[complex, complex]:
    u = np.asarray(u, dtype=float)
    if u.shape != (4,):
        raise ValueError("u must be [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]")
    return complex(u[0], u[1]), complex(u[2], u[3])


def raw_to_physical(u: np.ndarray, kappa_b: float = 10.0, eps: float = 1e-12) -> np.ndarray:
    """Convert raw controls ``u`` into physical optimizer coordinates ``v``."""

    g2, eps_d = _raw_complex(u)
    g2_safe = g2 if abs(g2) > eps else eps + 0.0j
    kappa_2 = 4.0 * abs(g2_safe) ** 2 / float(kappa_b)
    alpha_sq = 2.0 * eps_d / np.conj(g2_safe)
    alpha = np.sqrt(alpha_sq + 0.0j)

    # TODO: refine phi_mismatch convention after comparing phase conventions with the theoretical notes.
    phi_mismatch = np.angle(g2_safe)
    return np.array(
        [
            np.log(kappa_2 + eps),
            np.log(abs(alpha) + eps),
            sanitize_angle(np.angle(alpha)),
            sanitize_angle(phi_mismatch),
        ],
        dtype=float,
    )


def physical_to_raw(v: np.ndarray, kappa_b: float = 10.0) -> np.ndarray:
    """Convert physical optimizer coordinates ``v`` back to raw controls ``u``."""

    v = np.asarray(v, dtype=float)
    if v.shape != (4,):
        raise ValueError("v must be [log_kappa_2, log_abs_alpha, arg_alpha, phi_mismatch]")

    log_kappa_2, log_abs_alpha, arg_alpha, phi_mismatch = v
    kappa_2 = np.exp(log_kappa_2)
    abs_alpha = np.exp(log_abs_alpha)

    abs_g2 = 0.5 * np.sqrt(kappa_2 * float(kappa_b))
    g2 = abs_g2 * np.exp(1j * phi_mismatch)
    alpha = abs_alpha * np.exp(1j * arg_alpha)
    eps_d = 0.5 * np.conj(g2) * alpha**2

    return np.array([g2.real, g2.imag, eps_d.real, eps_d.imag], dtype=float)


def physical_diagnostics_from_raw(u: np.ndarray, kappa_b: float = 10.0, eps: float = 1e-12) -> dict[str, float]:
    """Return physical diagnostics implied by raw controls."""

    g2, eps_d = _raw_complex(u)
    g2_safe = g2 if abs(g2) > eps else eps + 0.0j
    kappa_2 = 4.0 * abs(g2_safe) ** 2 / float(kappa_b)
    alpha_sq = 2.0 * eps_d / np.conj(g2_safe)
    alpha = np.sqrt(alpha_sq + 0.0j)
    return {
        "g2_abs": float(abs(g2)),
        "g2_phase": float(sanitize_angle(np.angle(g2_safe))),
        "eps_d_abs": float(abs(eps_d)),
        "eps_d_phase": float(sanitize_angle(np.angle(eps_d))) if abs(eps_d) > eps else 0.0,
        "kappa_2": float(kappa_2),
        "alpha_abs": float(abs(alpha)),
        "alpha_phase": float(sanitize_angle(np.angle(alpha))),
    }


def physical_bounds_from_raw_reference(
    u0: np.ndarray,
    kappa_b: float = 10.0,
    spans: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    """Build conservative physical-coordinate bounds around a raw reference."""

    spans = {
        "log_kappa2": 2.0,
        "log_abs_alpha": 1.0,
        **(spans or {}),
    }
    v0 = raw_to_physical(u0, kappa_b=kappa_b)
    bounds = {
        "log_kappa2": [float(v0[0] - spans["log_kappa2"]), float(v0[0] + spans["log_kappa2"])],
        "log_abs_alpha": [float(v0[1] - spans["log_abs_alpha"]), float(v0[1] + spans["log_abs_alpha"])],
        "arg_alpha": [-math.pi, math.pi],
        "phi_mismatch": [-math.pi, math.pi],
    }
    lower = np.array([bounds[key][0] for key in bounds], dtype=float)
    upper = np.array([bounds[key][1] for key in bounds], dtype=float)
    return lower, upper, bounds


def evaluate_physical_candidate(
    v: np.ndarray,
    baseline_evaluator: Callable[..., Any],
    config: dict[str, Any] | None = None,
    kappa_b: float = 10.0,
) -> Any:
    """Convert ``v`` to raw ``u`` and call an existing baseline evaluator."""

    u = physical_to_raw(v, kappa_b=kappa_b)
    try:
        result = baseline_evaluator(u, config=config)
    except TypeError:
        result = baseline_evaluator(u)

    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
        loss, metrics = result
        metrics = dict(metrics)
        _attach_coordinate_metrics(metrics, v, u, kappa_b)
        return loss, metrics
    if isinstance(result, dict):
        result = dict(result)
        _attach_coordinate_metrics(result, v, u, kappa_b)
        return result
    return result


def _attach_coordinate_metrics(metrics: dict[str, Any], v: np.ndarray, u: np.ndarray, kappa_b: float) -> None:
    metrics.update(
        {
            "v0_log_kappa2": float(v[0]),
            "v1_log_abs_alpha": float(v[1]),
            "v2_arg_alpha": float(v[2]),
            "v3_phi_mismatch": float(v[3]),
            "u0_re_g2": float(u[0]),
            "u1_im_g2": float(u[1]),
            "u2_re_eps_d": float(u[2]),
            "u3_im_eps_d": float(u[3]),
        }
    )
    metrics.update(physical_diagnostics_from_raw(u, kappa_b=kappa_b))

