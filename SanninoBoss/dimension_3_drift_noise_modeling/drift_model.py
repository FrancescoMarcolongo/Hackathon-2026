"""Detuning drift models for the final Drift and Noise Modeling block."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DetuningDriftConfig:
    epochs: int = 120
    seed: int = 7319
    sinusoid_amplitude: float = 0.030
    sinusoid_frequency: float = 0.016
    band_amplitude: float = 0.005
    band_f_max: float = 0.018


def band_limited_component(
    epochs: np.ndarray,
    *,
    amplitude: float,
    f_max: float,
    rng: np.random.Generator,
    n_modes: int = 5,
) -> np.ndarray:
    """Smooth random low-frequency component with bounded amplitude."""
    signal = np.zeros_like(epochs, dtype=float)
    for frequency in np.linspace(f_max / n_modes, f_max, n_modes):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        weight = rng.normal(0.0, 1.0)
        signal += weight * np.sin(2.0 * np.pi * frequency * epochs + phase)
    max_abs = float(np.max(np.abs(signal)))
    if max_abs > 0.0:
        signal = amplitude * signal / max_abs
    return signal


def generate_detuning_drift(config: DetuningDriftConfig) -> dict[str, np.ndarray | float]:
    """Generate Delta_env(epoch) from a random-phase sinusoid plus smooth drift."""
    rng = np.random.default_rng(config.seed)
    epochs = np.arange(config.epochs, dtype=float)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    sinusoid = config.sinusoid_amplitude * np.sin(2.0 * np.pi * config.sinusoid_frequency * epochs + phase)
    band = band_limited_component(
        epochs,
        amplitude=config.band_amplitude,
        f_max=config.band_f_max,
        rng=rng,
    )
    delta_env = sinusoid + band
    return {
        "epoch": epochs,
        "delta_env": delta_env,
        "delta_ideal": -delta_env,
        "phase": phase,
        "sinusoid": sinusoid,
        "band_limited": band,
    }
