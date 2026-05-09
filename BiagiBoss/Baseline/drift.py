"""Deterministic slow control drift helpers for the cat-qubit optimizer."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


DEFAULT_CONTROL_BOUNDS = np.array(
    [
        [0.25, 3.0],
        [-1.0, 1.0],
        [0.50, 8.0],
        [-3.0, 3.0],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class DriftConfig:
    kind: str = "slow_fourier_control"
    period_epochs: float = 240.0
    bandwidth: int = 2
    amplitude_scale: float = 1.0
    amplitudes: tuple[float, float, float, float] = (0.12, 0.06, 0.25, 0.18)
    phi1: tuple[float, float, float, float] = (0.0, 0.8, 1.6, 2.4)
    phi2: tuple[float, float, float, float] = (1.3, 2.1, 2.9, 3.7)
    x_reference: tuple[float, float, float, float] = (
        1.3285188,
        -0.7782763,
        3.0005451,
        1.7545150,
    )


def _as_control_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    return array


def drift_vector(epoch: int | float, drift_cfg: DriftConfig) -> np.ndarray:
    """Return the four-dimensional slow Fourier drift d(epoch)."""

    if drift_cfg.kind != "slow_fourier_control":
        raise ValueError(f"Unsupported drift kind: {drift_cfg.kind}")
    if drift_cfg.bandwidth not in (1, 2):
        raise ValueError("This benchmark implements bandwidth 1 or 2 only.")
    if drift_cfg.period_epochs <= 0:
        raise ValueError("period_epochs must be positive.")

    e = float(epoch)
    amplitudes = drift_cfg.amplitude_scale * _as_control_vector(
        drift_cfg.amplitudes, "amplitudes"
    )
    phi1 = _as_control_vector(drift_cfg.phi1, "phi1")
    phi2 = _as_control_vector(drift_cfg.phi2, "phi2")
    phase = 2.0 * np.pi * e / float(drift_cfg.period_epochs)
    signal = 0.70 * np.sin(phase + phi1)
    if drift_cfg.bandwidth >= 2:
        signal = signal + 0.30 * np.sin(2.0 * phase + phi2)
    return amplitudes * signal


def true_optimum_command(epoch: int | float, drift_cfg: DriftConfig) -> np.ndarray:
    """Known command that cancels the synthetic hardware drift."""

    return _as_control_vector(drift_cfg.x_reference, "x_reference") + drift_vector(
        epoch, drift_cfg
    )


def apply_control_drift(
    x_cmd: np.ndarray,
    epoch: int | float,
    drift_cfg: DriftConfig,
    bounds: np.ndarray = DEFAULT_CONTROL_BOUNDS,
) -> dict[str, np.ndarray]:
    """Apply command clipping and epoch-level drift before Hamiltonian evaluation."""

    command = _as_control_vector(x_cmd, "x_cmd")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")

    x_cmd_clipped = np.minimum(np.maximum(command, bounds[:, 0]), bounds[:, 1])
    drift = drift_vector(epoch, drift_cfg)
    x_eff = x_cmd_clipped - drift
    x_true_opt = true_optimum_command(epoch, drift_cfg)
    tracking_error = x_cmd_clipped - x_true_opt
    effective_error = x_eff - _as_control_vector(drift_cfg.x_reference, "x_reference")
    return {
        "x_cmd_clipped": x_cmd_clipped,
        "drift": drift,
        "x_eff": x_eff,
        "x_true_opt": x_true_opt,
        "tracking_error": tracking_error,
        "effective_error": effective_error,
    }


def verify_or_scale_true_path(
    drift_cfg: DriftConfig,
    bounds: np.ndarray,
    epochs: int,
    *,
    safety: float = 0.98,
) -> tuple[DriftConfig, dict[str, object]]:
    """Verify x_ref + d(epoch) remains in bounds, scaling amplitudes if needed."""

    bounds = np.asarray(bounds, dtype=float)
    x_ref = _as_control_vector(drift_cfg.x_reference, "x_reference")
    if np.any(x_ref < bounds[:, 0]) or np.any(x_ref > bounds[:, 1]):
        raise ValueError(f"x_reference is outside bounds: {x_ref.tolist()}")

    base_cfg = replace(drift_cfg, amplitude_scale=1.0)
    base_drift = np.array([drift_vector(e, base_cfg) for e in range(int(epochs) + 1)])
    allowed_scale = float("inf")
    limiting_component = None
    limiting_epoch = None
    for epoch, drift in enumerate(base_drift):
        for idx, value in enumerate(drift):
            if abs(value) < 1e-14:
                continue
            if value > 0:
                scale = (bounds[idx, 1] - x_ref[idx]) / value
            else:
                scale = (bounds[idx, 0] - x_ref[idx]) / value
            if scale < allowed_scale:
                allowed_scale = float(scale)
                limiting_component = int(idx)
                limiting_epoch = int(epoch)

    original_scale = float(drift_cfg.amplitude_scale)
    if not np.isfinite(allowed_scale):
        allowed_scale = original_scale
    adjusted_scale = original_scale
    was_scaled = False
    if original_scale > allowed_scale:
        adjusted_scale = max(0.0, safety * allowed_scale)
        was_scaled = True

    checked_cfg = replace(drift_cfg, amplitude_scale=adjusted_scale)
    true_path = np.array(
        [true_optimum_command(e, checked_cfg) for e in range(int(epochs) + 1)]
    )
    path_min = np.min(true_path, axis=0)
    path_max = np.max(true_path, axis=0)
    inside = bool(np.all(path_min >= bounds[:, 0] - 1e-12) and np.all(path_max <= bounds[:, 1] + 1e-12))
    if not inside:
        raise ValueError(
            "True optimum command path remains outside bounds after amplitude scaling: "
            f"min={path_min.tolist()}, max={path_max.tolist()}, bounds={bounds.tolist()}"
        )

    return checked_cfg, {
        "inside_bounds": inside,
        "original_amplitude_scale": original_scale,
        "amplitude_scale": adjusted_scale,
        "was_scaled": was_scaled,
        "allowed_scale": float(allowed_scale),
        "limiting_component": limiting_component,
        "limiting_epoch": limiting_epoch,
        "path_min": path_min.tolist(),
        "path_max": path_max.tolist(),
        "bounds": bounds.tolist(),
    }
