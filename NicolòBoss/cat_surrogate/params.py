"""Parameter packing and cheap derived physics features."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

EPS = 1e-12


def pack_params(g2: complex, eps_d: complex) -> np.ndarray:
    """Pack complex control parameters into the optimizer's four-real vector."""

    return np.asarray([g2.real, g2.imag, eps_d.real, eps_d.imag], dtype=float)


def unpack_params(theta: np.ndarray) -> tuple[complex, complex]:
    """Unpack ``[Re(g2), Im(g2), Re(eps_d), Im(eps_d)]`` into complex knobs."""

    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"theta must contain exactly 4 values, got shape {arr.shape}")
    g2 = complex(float(arr[0]), float(arr[1]))
    eps_d = complex(float(arr[2]), float(arr[3]))
    return g2, eps_d


def _safe_phase(z: complex) -> float:
    return 0.0 if abs(z) <= EPS else float(np.angle(z))


def derived_physics_features(
    g2: complex,
    eps_d: complex,
    kappa_b: float,
    na: int | None = None,
) -> dict[str, float]:
    """Compute inexpensive analytic features from the four optimization knobs."""

    safe_kappa = max(abs(float(kappa_b)), EPS)
    abs_g2 = abs(g2)
    abs_eps_d = abs(eps_d)
    kappa2_est = 4.0 * abs_g2**2 / safe_kappa
    eps2_est = 2.0 * g2 * eps_d / safe_kappa
    abs_eps2_est = abs(eps2_est)
    alpha2_est = 2.0 * abs_eps2_est / max(kappa2_est, EPS)
    alpha_est = math.sqrt(max(alpha2_est, 0.0))

    features: dict[str, float] = {
        "re_g2": float(g2.real),
        "im_g2": float(g2.imag),
        "abs_g2": float(abs_g2),
        "phase_g2": _safe_phase(g2),
        "re_eps_d": float(eps_d.real),
        "im_eps_d": float(eps_d.imag),
        "abs_eps_d": float(abs_eps_d),
        "phase_eps_d": _safe_phase(eps_d),
        "kappa2_est": float(kappa2_est),
        "re_eps2_est": float(eps2_est.real),
        "im_eps2_est": float(eps2_est.imag),
        "abs_eps2_est": float(abs_eps2_est),
        "alpha2_est": float(alpha2_est),
        "alpha_est": float(alpha_est),
    }
    if na is not None:
        features["truncation_margin"] = float(na - (alpha_est**2 + 4.0 * alpha_est))
    return features


def theta_feature_dict(theta: np.ndarray, kappa_b: float, na: int | None = None) -> dict[str, float]:
    """Return packed and derived features for a theta vector."""

    g2, eps_d = unpack_params(theta)
    features: dict[str, float] = {
        "theta_re_g2": float(g2.real),
        "theta_im_g2": float(g2.imag),
        "theta_re_eps_d": float(eps_d.real),
        "theta_im_eps_d": float(eps_d.imag),
    }
    features.update(derived_physics_features(g2, eps_d, kappa_b=kappa_b, na=na))
    return features


def ensure_numpy_theta(theta: Any) -> np.ndarray:
    """Convert user input to a finite four-vector."""

    arr = np.asarray(theta, dtype=float).reshape(-1)
    if arr.size != 4:
        raise ValueError(f"theta must contain exactly 4 values, got shape {arr.shape}")
    return arr
