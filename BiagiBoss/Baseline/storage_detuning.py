"""Storage-detuning drift helpers with an explicit compensation knob."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


DEFAULT_STORAGE_DETUNING_BOUNDS = np.array(
    [
        [0.25, 3.0],
        [-1.0, 1.0],
        [0.50, 8.0],
        [-3.0, 3.0],
        [-0.18, 0.18],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class StorageDetuningConfig:
    kind: str = "storage_detuning_compensation"
    period_epochs: float = 96.0
    bandwidth: int = 3
    amplitude: float = 0.08
    weights: tuple[float, float, float] = (0.55, 0.30, 0.15)
    phases: tuple[float, float, float] = (0.20, 1.35, 2.80)
    x_reference: tuple[float, float, float, float] = (
        1.57798161,
        -0.22211626,
        3.25404420,
        1.31582660,
    )
    detuning_reference: float = 0.0


def _as_control_vector(values: Iterable[float], name: str, size: int) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def storage_detuning_drift(epoch: int | float, cfg: StorageDetuningConfig) -> float:
    """Return the synthetic hardware storage detuning at optimizer epoch e."""

    if cfg.kind != "storage_detuning_compensation":
        raise ValueError(f"Unsupported storage-detuning drift kind: {cfg.kind}")
    if cfg.bandwidth < 1 or cfg.bandwidth > len(cfg.weights):
        raise ValueError("bandwidth must be between 1 and len(weights).")
    if cfg.period_epochs <= 0:
        raise ValueError("period_epochs must be positive.")

    weights = _as_control_vector(cfg.weights, "weights", len(cfg.weights))
    phases = _as_control_vector(cfg.phases, "phases", len(cfg.phases))
    if len(phases) < cfg.bandwidth:
        raise ValueError("phases must include one phase per harmonic.")
    e = float(epoch)
    phase = 2.0 * np.pi * e / float(cfg.period_epochs)
    value = 0.0
    for harmonic in range(1, int(cfg.bandwidth) + 1):
        value += weights[harmonic - 1] * np.sin(harmonic * phase + phases[harmonic - 1])
    return float(cfg.amplitude * value)


def true_storage_detuning_optimum(
    epoch: int | float,
    cfg: StorageDetuningConfig,
) -> np.ndarray:
    """Known five-knob optimum: four stationary controls plus detuning feed-forward."""

    x_ref = _as_control_vector(cfg.x_reference, "x_reference", 4)
    delta_cmd = cfg.detuning_reference + storage_detuning_drift(epoch, cfg)
    return np.array([*x_ref, delta_cmd], dtype=float)


def apply_storage_detuning(
    x_cmd: np.ndarray,
    epoch: int | float,
    cfg: StorageDetuningConfig,
    bounds: np.ndarray = DEFAULT_STORAGE_DETUNING_BOUNDS,
) -> dict[str, np.ndarray | float]:
    """Convert commanded controls to physical controls under storage detuning drift."""

    command = _as_control_vector(x_cmd, "x_cmd", 5)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (5, 2):
        raise ValueError("bounds must have shape (5, 2)")

    x_cmd_clipped = np.minimum(np.maximum(command, bounds[:, 0]), bounds[:, 1])
    drift = storage_detuning_drift(epoch, cfg)
    residual_detuning = float(x_cmd_clipped[4] - drift)
    x_eff = np.array([*x_cmd_clipped[:4], residual_detuning], dtype=float)
    x_true_opt = true_storage_detuning_optimum(epoch, cfg)
    x_reference_eff = np.array(
        [*_as_control_vector(cfg.x_reference, "x_reference", 4), cfg.detuning_reference],
        dtype=float,
    )
    tracking_error = x_cmd_clipped - x_true_opt
    effective_error = x_eff - x_reference_eff
    return {
        "x_cmd_clipped": x_cmd_clipped,
        "drift": drift,
        "x_eff": x_eff,
        "x_true_opt": x_true_opt,
        "x_reference_eff": x_reference_eff,
        "tracking_error": tracking_error,
        "effective_error": effective_error,
        "residual_detuning": residual_detuning,
    }


def verify_storage_detuning_path(
    cfg: StorageDetuningConfig,
    bounds: np.ndarray,
    epochs: int,
) -> dict[str, object]:
    """Verify the known five-knob optimum is inside optimizer bounds."""

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (5, 2):
        raise ValueError("bounds must have shape (5, 2)")
    path = np.array([true_storage_detuning_optimum(e, cfg) for e in range(int(epochs) + 1)])
    path_min = np.min(path, axis=0)
    path_max = np.max(path, axis=0)
    inside = bool(np.all(path_min >= bounds[:, 0] - 1e-12) and np.all(path_max <= bounds[:, 1] + 1e-12))
    if not inside:
        raise ValueError(
            "Storage-detuning optimum path is outside bounds: "
            f"min={path_min.tolist()}, max={path_max.tolist()}, bounds={bounds.tolist()}"
        )
    return {
        "inside_bounds": inside,
        "path_min": path_min.tolist(),
        "path_max": path_max.tolist(),
        "bounds": bounds.tolist(),
    }
