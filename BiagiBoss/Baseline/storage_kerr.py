"""Storage-Kerr drift helpers for four-knob cat-control tracking."""

from __future__ import annotations

from dataclasses import dataclass
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
class StorageKerrConfig:
    kind: str = "storage_kerr_four_knob"
    period_epochs: float = 120.0
    bandwidth: int = 2
    amplitude: float = 0.020
    weights: tuple[float, float] = (0.68, 0.32)
    phases: tuple[float, float] = (0.0, 0.0)
    x_reference: tuple[float, float, float, float] = (
        1.6192075681,
        -0.3868491524,
        3.4954392642,
        -0.6602812715,
    )
    compensation_sensitivity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    kerr_reference: float = 0.0


def as_control_vector(values: Iterable[float], name: str, size: int = 4) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def storage_kerr_drift(epoch: int | float, cfg: StorageKerrConfig) -> float:
    """Return the epoch-level storage Kerr coefficient K(e)."""

    if cfg.kind != "storage_kerr_four_knob":
        raise ValueError(f"Unsupported storage-Kerr drift kind: {cfg.kind}")
    if cfg.bandwidth < 1 or cfg.bandwidth > len(cfg.weights):
        raise ValueError("bandwidth must be between 1 and len(weights).")
    if cfg.period_epochs <= 0:
        raise ValueError("period_epochs must be positive.")
    weights = np.asarray(cfg.weights, dtype=float)
    phases = np.asarray(cfg.phases, dtype=float)
    if len(phases) < cfg.bandwidth:
        raise ValueError("phases must include one phase per harmonic.")

    e = float(epoch)
    phase = 2.0 * np.pi * e / float(cfg.period_epochs)
    value = 0.0
    for harmonic in range(1, int(cfg.bandwidth) + 1):
        value += weights[harmonic - 1] * np.sin(harmonic * phase + phases[harmonic - 1])
    return float(cfg.amplitude * value)


def true_storage_kerr_optimum(epoch: int | float, cfg: StorageKerrConfig) -> np.ndarray:
    """Local four-knob Kerr-compensated target from stationary sensitivity calibration."""

    x_ref = as_control_vector(cfg.x_reference, "x_reference")
    sensitivity = as_control_vector(cfg.compensation_sensitivity, "compensation_sensitivity")
    delta_k = storage_kerr_drift(epoch, cfg) - float(cfg.kerr_reference)
    return x_ref + sensitivity * delta_k


def apply_storage_kerr(
    x_cmd: np.ndarray,
    epoch: int | float,
    cfg: StorageKerrConfig,
    bounds: np.ndarray = DEFAULT_CONTROL_BOUNDS,
) -> dict[str, np.ndarray | float]:
    """Clip four commanded cat controls and attach the epoch-level Kerr drift."""

    command = as_control_vector(x_cmd, "x_cmd")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")

    x_cmd_clipped = np.minimum(np.maximum(command, bounds[:, 0]), bounds[:, 1])
    kerr = storage_kerr_drift(epoch, cfg)
    x_true_opt = true_storage_kerr_optimum(epoch, cfg)
    x_reference = as_control_vector(cfg.x_reference, "x_reference")
    tracking_error = x_cmd_clipped - x_true_opt
    reference_error = x_cmd_clipped - x_reference
    return {
        "x_cmd_clipped": x_cmd_clipped,
        "x_eff": x_cmd_clipped.copy(),
        "x_true_opt": x_true_opt,
        "x_reference_eff": x_reference,
        "tracking_error": tracking_error,
        "effective_error": reference_error,
        "storage_kerr": kerr,
        "storage_kerr_delta": kerr - float(cfg.kerr_reference),
    }


def verify_storage_kerr_path(
    cfg: StorageKerrConfig,
    bounds: np.ndarray,
    epochs: int,
) -> dict[str, object]:
    """Verify the calibrated four-control target path remains inside bounds."""

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")
    path = np.array([true_storage_kerr_optimum(e, cfg) for e in range(int(epochs) + 1)])
    path_min = np.min(path, axis=0)
    path_max = np.max(path, axis=0)
    inside = bool(np.all(path_min >= bounds[:, 0] - 1e-12) and np.all(path_max <= bounds[:, 1] + 1e-12))
    if not inside:
        raise ValueError(
            "Storage-Kerr target path is outside bounds: "
            f"min={path_min.tolist()}, max={path_max.tolist()}, bounds={bounds.tolist()}"
        )
    return {
        "inside_bounds": inside,
        "path_min": path_min.tolist(),
        "path_max": path_max.tolist(),
        "bounds": bounds.tolist(),
    }
