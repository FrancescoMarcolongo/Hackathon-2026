"""Storage-Kerr drift helpers with an explicit Kerr compensation knob."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from storage_kerr import storage_kerr_drift


DEFAULT_STORAGE_KERR_BOUNDS = np.array(
    [
        [0.25, 3.0],
        [-1.0, 1.0],
        [0.50, 8.0],
        [-3.0, 3.0],
        [-0.45, 0.45],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class StorageKerrCompensationConfig:
    kind: str = "storage_kerr_compensation"
    period_epochs: float = 100.0
    bandwidth: int = 2
    amplitude: float = 0.30
    weights: tuple[float, float] = (0.68, 0.32)
    phases: tuple[float, float] = (0.0, 0.0)
    x_reference: tuple[float, float, float, float] = (
        1.6192075681,
        -0.3868491524,
        3.4954392642,
        -0.6602812715,
    )
    kerr_reference: float = 0.0


def _as_vector(values: Iterable[float], name: str, size: int) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def kerr_compensation_drift(epoch: int | float, cfg: StorageKerrCompensationConfig) -> float:
    if cfg.kind != "storage_kerr_compensation":
        raise ValueError(f"Unsupported storage-Kerr compensation kind: {cfg.kind}")
    proxy = _ProxyKerrConfig(
        period_epochs=cfg.period_epochs,
        bandwidth=cfg.bandwidth,
        amplitude=cfg.amplitude,
        weights=cfg.weights,
        phases=cfg.phases,
    )
    return storage_kerr_drift(epoch, proxy)


@dataclass(frozen=True)
class _ProxyKerrConfig:
    kind: str = "storage_kerr_four_knob"
    period_epochs: float = 100.0
    bandwidth: int = 2
    amplitude: float = 0.30
    weights: tuple[float, float] = (0.68, 0.32)
    phases: tuple[float, float] = (0.0, 0.0)
    x_reference: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    compensation_sensitivity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    kerr_reference: float = 0.0


def true_kerr_compensation_optimum(
    epoch: int | float,
    cfg: StorageKerrCompensationConfig,
) -> np.ndarray:
    x_ref = _as_vector(cfg.x_reference, "x_reference", 4)
    k_cmd = float(cfg.kerr_reference) - kerr_compensation_drift(epoch, cfg)
    return np.array([*x_ref, k_cmd], dtype=float)


def apply_storage_kerr_compensation(
    x_cmd: np.ndarray,
    epoch: int | float,
    cfg: StorageKerrCompensationConfig,
    bounds: np.ndarray = DEFAULT_STORAGE_KERR_BOUNDS,
) -> dict[str, np.ndarray | float]:
    command = _as_vector(x_cmd, "x_cmd", 5)
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (5, 2):
        raise ValueError("bounds must have shape (5, 2)")

    x_cmd_clipped = np.minimum(np.maximum(command, bounds[:, 0]), bounds[:, 1])
    k_drift = kerr_compensation_drift(epoch, cfg)
    residual_kerr = float(k_drift + x_cmd_clipped[4])
    x_eff = np.array([*x_cmd_clipped[:4], residual_kerr], dtype=float)
    x_true_opt = true_kerr_compensation_optimum(epoch, cfg)
    x_reference_eff = np.array(
        [*_as_vector(cfg.x_reference, "x_reference", 4), float(cfg.kerr_reference)],
        dtype=float,
    )
    tracking_error = x_cmd_clipped - x_true_opt
    effective_error = x_eff - x_reference_eff
    return {
        "x_cmd_clipped": x_cmd_clipped,
        "x_eff": x_eff,
        "x_true_opt": x_true_opt,
        "x_reference_eff": x_reference_eff,
        "tracking_error": tracking_error,
        "effective_error": effective_error,
        "kerr_drift": k_drift,
        "residual_kerr": residual_kerr,
    }


def verify_kerr_compensation_path(
    cfg: StorageKerrCompensationConfig,
    bounds: np.ndarray,
    epochs: int,
) -> dict[str, object]:
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (5, 2):
        raise ValueError("bounds must have shape (5, 2)")
    path = np.array([true_kerr_compensation_optimum(e, cfg) for e in range(int(epochs) + 1)])
    path_min = np.min(path, axis=0)
    path_max = np.max(path, axis=0)
    inside = bool(np.all(path_min >= bounds[:, 0] - 1e-12) and np.all(path_max <= bounds[:, 1] + 1e-12))
    if not inside:
        raise ValueError(
            "Storage-Kerr compensation path is outside bounds: "
            f"min={path_min.tolist()}, max={path_max.tolist()}, bounds={bounds.tolist()}"
        )
    return {
        "inside_bounds": inside,
        "path_min": path_min.tolist(),
        "path_max": path_max.tolist(),
        "bounds": bounds.tolist(),
    }
