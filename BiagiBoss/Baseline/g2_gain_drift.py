"""Complex g2-gain drift helpers with an exact command optimum."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from cat_model import complex_to_params, params_to_complex
from run_core_bias_optimization import BOUNDS


@dataclass(frozen=True)
class G2GainDriftConfig:
    kind: str = "complex_g2_gain"
    affect_epsilon_d: bool = True
    period_epochs: float = 72.0
    bandwidth: int = 2
    amplitude_scale: float = 1.0
    amplitude_modulation: float = 0.10
    phase_modulation: float = 0.18
    amplitude_weights: tuple[float, float] = (0.65, 0.35)
    phase_weights: tuple[float, float] = (0.60, 0.40)
    amplitude_phases: tuple[float, float] = (0.15, 1.40)
    phase_phases: tuple[float, float] = (1.10, 2.65)
    x_reference: tuple[float, float, float, float] = (
        1.6192075681484381,
        -0.38684915244951523,
        3.495439264194342,
        -0.6602812715117858,
    )


def _as_control_vector(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (4,):
        raise ValueError(f"{name} must have shape (4,)")
    return array


def _as_pair(values: Iterable[float], name: str) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=float)
    if array.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    return array


def _wave(epoch: int | float, period_epochs: float, weights: np.ndarray, phases: np.ndarray) -> float:
    phase = 2.0 * np.pi * float(epoch) / float(period_epochs)
    return float(weights[0] * np.sin(phase + phases[0]) + weights[1] * np.sin(2.0 * phase + phases[1]))


def g2_gain_components(epoch: int | float, cfg: G2GainDriftConfig) -> dict[str, float]:
    """Return amplitude and phase components of the complex g2 transfer drift."""

    if cfg.kind != "complex_g2_gain":
        raise ValueError(f"Unsupported g2-gain drift kind: {cfg.kind}")
    if cfg.bandwidth != 2:
        raise ValueError("The complex-gain benchmark uses exactly two harmonics.")
    if cfg.period_epochs <= 0:
        raise ValueError("period_epochs must be positive.")

    amp_weights = _as_pair(cfg.amplitude_weights, "amplitude_weights")
    phase_weights = _as_pair(cfg.phase_weights, "phase_weights")
    amp_phases = _as_pair(cfg.amplitude_phases, "amplitude_phases")
    phase_phases = _as_pair(cfg.phase_phases, "phase_phases")
    amplitude_delta = (
        float(cfg.amplitude_scale)
        * float(cfg.amplitude_modulation)
        * _wave(epoch, cfg.period_epochs, amp_weights, amp_phases)
    )
    phase_delta = (
        float(cfg.amplitude_scale)
        * float(cfg.phase_modulation)
        * _wave(epoch, cfg.period_epochs, phase_weights, phase_phases)
    )
    return {
        "gain_amplitude_delta": float(amplitude_delta),
        "gain_amplitude_factor": float(1.0 + amplitude_delta),
        "gain_phase_rad": float(phase_delta),
    }


def g2_gain_complex(epoch: int | float, cfg: G2GainDriftConfig) -> complex:
    """Return the complex transfer function multiplying commanded g2."""

    parts = g2_gain_components(epoch, cfg)
    amplitude = parts["gain_amplitude_factor"]
    if amplitude <= 0:
        raise ValueError(f"Non-positive g2 gain amplitude at epoch {epoch}: {amplitude}")
    return complex(amplitude * np.exp(1j * parts["gain_phase_rad"]))


def true_g2_gain_optimum(epoch: int | float, cfg: G2GainDriftConfig) -> np.ndarray:
    """Known command that cancels the complex gain drift."""

    x_ref = _as_control_vector(cfg.x_reference, "x_reference")
    g2_ref, eps_ref = params_to_complex(x_ref)
    gain = g2_gain_complex(epoch, cfg)
    eps_cmd = eps_ref / gain if cfg.affect_epsilon_d else eps_ref
    return complex_to_params(g2_ref / gain, eps_cmd)


def apply_g2_gain_drift(
    x_cmd: np.ndarray,
    epoch: int | float,
    cfg: G2GainDriftConfig,
    bounds: np.ndarray = BOUNDS,
) -> dict[str, np.ndarray | float]:
    """Map commanded controls to physical controls under complex g2 gain drift."""

    command = _as_control_vector(x_cmd, "x_cmd")
    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")

    x_cmd_clipped = np.minimum(np.maximum(command, bounds[:, 0]), bounds[:, 1])
    g2_cmd, eps_cmd = params_to_complex(x_cmd_clipped)
    gain = g2_gain_complex(epoch, cfg)
    g2_eff = gain * g2_cmd
    eps_eff = gain * eps_cmd if cfg.affect_epsilon_d else eps_cmd
    x_eff = complex_to_params(g2_eff, eps_eff)
    x_ref = _as_control_vector(cfg.x_reference, "x_reference")
    x_true_opt = true_g2_gain_optimum(epoch, cfg)
    tracking_error = x_cmd_clipped - x_true_opt
    effective_error = x_eff - x_ref
    command_offset = x_true_opt - x_ref
    parts = g2_gain_components(epoch, cfg)
    return {
        "x_cmd_clipped": x_cmd_clipped,
        "x_eff": x_eff,
        "x_true_opt": x_true_opt,
        "x_reference_eff": x_ref,
        "tracking_error": tracking_error,
        "effective_error": effective_error,
        "command_offset": command_offset,
        "gain_real": float(gain.real),
        "gain_imag": float(gain.imag),
        **parts,
    }


def verify_or_scale_g2_gain_path(
    cfg: G2GainDriftConfig,
    bounds: np.ndarray,
    epochs: int,
    *,
    safety: float = 0.98,
) -> tuple[G2GainDriftConfig, dict[str, object]]:
    """Verify the known command path is in bounds, scaling the drift if needed."""

    bounds = np.asarray(bounds, dtype=float)
    if bounds.shape != (4, 2):
        raise ValueError("bounds must have shape (4, 2)")
    x_ref = _as_control_vector(cfg.x_reference, "x_reference")
    if np.any(x_ref < bounds[:, 0]) or np.any(x_ref > bounds[:, 1]):
        raise ValueError(f"x_reference is outside bounds: {x_ref.tolist()}")

    original_scale = float(cfg.amplitude_scale)
    scale = original_scale
    was_scaled = False
    for _ in range(60):
        candidate = replace(cfg, amplitude_scale=scale)
        path = np.array([true_g2_gain_optimum(epoch, candidate) for epoch in range(int(epochs) + 1)])
        inside = bool(np.all(path >= bounds[:, 0] - 1e-12) and np.all(path <= bounds[:, 1] + 1e-12))
        if inside:
            path_min = np.min(path, axis=0)
            path_max = np.max(path, axis=0)
            return candidate, {
                "inside_bounds": True,
                "original_amplitude_scale": original_scale,
                "amplitude_scale": scale,
                "was_scaled": was_scaled,
                "path_min": path_min.tolist(),
                "path_max": path_max.tolist(),
                "bounds": bounds.tolist(),
            }
        scale *= float(safety)
        was_scaled = True

    raise ValueError("Could not scale complex g2-gain path into optimizer bounds.")
