"""Drift signals used in Step 7 of the presentation pipeline."""

from __future__ import annotations

import numpy as np


def sinusoidal_drift(
    epochs: np.ndarray,
    *,
    amplitude: float,
    frequency: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Return d(epoch) = A sin(2 pi f epoch + phase)."""
    epochs = np.asarray(epochs, dtype=float)
    return amplitude * np.sin(2.0 * np.pi * frequency * epochs + phase)


def band_limited_random_drift(
    n_epochs: int,
    *,
    amplitude: float,
    f_max: float,
    seed: int,
) -> np.ndarray:
    """Generate a smooth random signal with Fourier support below f_max."""
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(n_epochs, d=1.0)
    coefficients = np.zeros(len(frequencies), dtype=complex)
    for index, frequency in enumerate(frequencies):
        if index == 0 or frequency > f_max:
            continue
        phase = rng.uniform(0.0, 2.0 * np.pi)
        scale = 1.0 / np.sqrt(index)
        coefficients[index] = scale * np.exp(1j * phase)
    signal = np.fft.irfft(coefficients, n=n_epochs)
    signal -= float(np.mean(signal))
    max_abs = float(np.max(np.abs(signal)))
    if max_abs > 0.0:
        signal = amplitude * signal / max_abs
    return signal


def fourier_spectrum(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return one-sided frequency and amplitude spectrum."""
    signal = np.asarray(signal, dtype=float)
    frequencies = np.fft.rfftfreq(len(signal), d=1.0)
    spectrum = np.abs(np.fft.rfft(signal)) / len(signal)
    return frequencies, spectrum
