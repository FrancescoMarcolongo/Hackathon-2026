"""Lightweight online drift-tracking wrappers for Step 7."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


TARGET_BIAS = 100.0
BIAS_TOL_REL = 0.03
TARGET_BAND_LOW = TARGET_BIAS * (1.0 - BIAS_TOL_REL)
TARGET_BAND_HIGH = TARGET_BIAS * (1.0 + BIAS_TOL_REL)


@dataclass(frozen=True)
class TrackerConfig:
    algorithm: str
    n_conv: int
    gain: float
    derivative_gain: float
    sensitivity: float = 2.2
    measurement_noise: float = 0.001
    correction_limit: float = 0.08
    seed: int = 0

    @property
    def f_conv(self) -> float:
        return 1.0 / float(self.n_conv)


BAYESIAN_TURBO = TrackerConfig(
    algorithm="Bayesian/TUrBO",
    n_conv=6,
    gain=0.30,
    derivative_gain=0.03,
    measurement_noise=0.0010,
    seed=1701,
)

BLT_TRACKER = TrackerConfig(
    algorithm="BLT",
    n_conv=9,
    gain=0.48,
    derivative_gain=0.26,
    measurement_noise=0.0008,
    seed=2607,
)


def run_online_tracking(
    drift_signal: np.ndarray,
    *,
    config: TrackerConfig,
    burn_in_epochs: int,
    f_drift: float | None,
    drift_amplitude: float,
    seed_offset: int = 0,
) -> list[dict]:
    """Track a scalar drift through a noisy two-point bias proxy.

    The physical simplification is:
        g2_eff(epoch) = g2_nominal * [1 + d(epoch)]

    The applied control can compensate this drift. The residual
    r = drift + compensation shifts log(eta / eta_target), and each
    optimizer updates the compensation from the measured residual.
    """
    drift_signal = np.asarray(drift_signal, dtype=float)
    rng = np.random.default_rng(config.seed + seed_offset)
    compensation = 0.0
    previous_estimated_residual = 0.0
    rows: list[dict] = []

    for epoch, drift in enumerate(drift_signal):
        learned_compensation = float(compensation)
        residual = float(drift + learned_compensation)
        log_bias_clean = (
            config.sensitivity * residual
            + 0.10 * residual * abs(residual)
        )
        log_bias_observed = log_bias_clean + float(rng.normal(0.0, config.measurement_noise))
        bias = float(TARGET_BIAS * math.exp(log_bias_observed))
        tracking_error = abs(math.log(max(bias, 1.0e-12) / TARGET_BIAS))

        estimated_residual = log_bias_observed / config.sensitivity
        residual_velocity = estimated_residual - previous_estimated_residual
        update = -config.gain * estimated_residual - config.derivative_gain * residual_velocity
        compensation = float(np.clip(compensation + update, -config.correction_limit, config.correction_limit))
        previous_estimated_residual = estimated_residual

        rows.append(
            {
                "epoch": int(epoch),
                "algorithm": config.algorithm,
                "drift_signal": float(drift),
                "ideal_compensation": float(-drift),
                "learned_compensation": learned_compensation,
                "residual_drift": residual,
                "bias": bias,
                "target_bias": TARGET_BIAS,
                "tracking_error": float(tracking_error),
                "target_band_low": TARGET_BAND_LOW,
                "target_band_high": TARGET_BAND_HIGH,
                "burn_in": int(epoch < burn_in_epochs),
                "f_drift": "" if f_drift is None else float(f_drift),
                "f_conv": float(config.f_conv),
                "n_conv": int(config.n_conv),
                "drift_amplitude": float(drift_amplitude),
                "seed": int(config.seed + seed_offset),
            }
        )
    return rows


def tracking_metrics(rows: list[dict], *, burn_in_epochs: int) -> dict:
    post = [row for row in rows if int(row["epoch"]) >= burn_in_epochs]
    errors = np.asarray([float(row["tracking_error"]) for row in post], dtype=float)
    biases = np.asarray([float(row["bias"]) for row in post], dtype=float)
    in_band = (biases >= TARGET_BAND_LOW) & (biases <= TARGET_BAND_HIGH)
    return {
        "rms_tracking_error": float(np.sqrt(np.mean(errors * errors))),
        "p_in_band": float(np.mean(in_band)),
    }
