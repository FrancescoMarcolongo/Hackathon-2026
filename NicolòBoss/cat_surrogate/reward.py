"""Reward functions for true, proxy, and learned-surrogate optimization."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .features import compute_cheap_observables
from .model import predict_log_lifetimes
from .params import derived_physics_features, unpack_params


def _soft_square_excess(value: float, threshold: float) -> float:
    return max(0.0, value - threshold) ** 2


def physics_penalty(theta: np.ndarray, config: Any) -> float:
    """Soft quadratic penalties for parameters outside trusted physics ranges."""

    try:
        g2, eps_d = unpack_params(theta)
        derived = derived_physics_features(g2, eps_d, config.kappa_b, config.na)
        values = [*derived.values(), abs(g2), abs(eps_d), float(config.kappa_b)]
        if not all(math.isfinite(float(v)) for v in values):
            return 1e6

        penalty = 0.0
        g_ratio = abs(g2) / max(abs(float(config.kappa_b)), 1e-12)
        penalty += _soft_square_excess(g_ratio, float(config.g2_over_kappa_b_max))
        alpha = float(derived["alpha_est"])
        penalty += max(0.0, float(config.alpha_min) - alpha) ** 2
        penalty += _soft_square_excess(alpha, float(config.alpha_max))
        penalty += max(0.0, -float(derived.get("truncation_margin", 0.0))) ** 2
        return float(penalty)
    except Exception:
        return 1e6


def surrogate_reward(
    theta: np.ndarray,
    config: Any,
    surrogate_bundle: dict[str, Any],
    adapter: Any,
    return_info: bool = False,
) -> float | tuple[float, dict[str, float]]:
    """Compute reward from learned log-lifetime predictions."""

    features = compute_cheap_observables(theta, config, adapter)
    prediction = predict_log_lifetimes(features, surrogate_bundle)
    log_tx = prediction["mean_log_T_X"]
    log_tz = prediction["mean_log_T_Z"]
    log_eta = log_tz - log_tx
    log_eta_target = math.log(max(float(config.eta_target), 1e-300))
    uncertainty = prediction["std_log_T_X"] + prediction["std_log_T_Z"]
    penalty = physics_penalty(theta, config)
    reward = (
        float(config.w_lifetime) * (log_tx + log_tz)
        - float(config.w_bias) * (log_eta - log_eta_target) ** 2
        - float(config.w_uncertainty) * uncertainty
        - float(config.w_physics) * penalty
    )
    reward = float(reward) if math.isfinite(reward) else -1e12

    if not return_info:
        return reward
    info = {
        **prediction,
        "pred_log_T_X": log_tx,
        "pred_log_T_Z": log_tz,
        "pred_log_eta": log_eta,
        "uncertainty": float(uncertainty),
        "physics_penalty": float(penalty),
        "reward": reward,
    }
    return reward, info


def true_lifetime_reward(benchmark_result: dict[str, float], config: Any) -> float:
    """Reward using an expensive benchmark result instead of surrogate output."""

    log_tx = float(benchmark_result.get("log_T_X", math.log(max(benchmark_result["T_X"], 1e-300))))
    log_tz = float(benchmark_result.get("log_T_Z", math.log(max(benchmark_result["T_Z"], 1e-300))))
    log_eta = log_tz - log_tx
    log_eta_target = math.log(max(float(config.eta_target), 1e-300))
    return float(float(config.w_lifetime) * (log_tx + log_tz) - float(config.w_bias) * (log_eta - log_eta_target) ** 2)


def short_time_proxy_reward(theta: np.ndarray, config: Any, adapter: Any) -> float:
    """Hand-designed reward from short-time proxy observables."""

    obs = adapter.compute_short_time_observables(theta, config)
    tx = float(obs.get("T_X_short", 0.0))
    tz = float(obs.get("T_Z_short", 0.0))
    if tx <= 0 or tz <= 0 or not math.isfinite(tx + tz):
        return -1e12
    benchmark_like = {"log_T_X": math.log(tx), "log_T_Z": math.log(tz)}
    return true_lifetime_reward(benchmark_like, config) - float(config.w_physics) * physics_penalty(theta, config)
