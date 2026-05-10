"""Reward functions for the core bias optimization challenge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    name: str = "target_band_primary"
    variant: str = "target_band"
    target_bias: float = 100.0
    bias_tol_rel: float = 0.03
    w_lifetime: float = 0.5
    w_bias_under: float = 60.0
    w_bias_exact: float = 120.0
    w_fit: float = 2.0
    feasibility_bonus: float = 12.0
    min_tx: float = 0.05
    min_tz: float = 5.0
    floor_weight: float = 12.0


# ============================================================
# CORE CHALLENGE REWARD / LOSS FUNCTION
# SepCMA MINIMIZES, so optimizer.tell receives loss_to_minimize.
# ============================================================
def compute_reward(metrics: Dict[str, object], cfg: RewardConfig) -> Dict[str, float | bool]:
    """Compute reward and loss for one candidate.

    Lower-bound variant:
        reward = FEASIBILITY_BONUS * is_feasible
               + W_LIFETIME * 0.5 * (log(T_X) + log(T_Z))
               - W_BIAS_UNDER * max(0, log(target_bias) - log(bias))**2
               - W_FIT * fit_penalty
               - floor penalties.

    Exact-target / target-band variants:
        reward = FEASIBILITY_BONUS * is_within_band
               + W_LIFETIME * 0.5 * (log(T_X) + log(T_Z))
               - W_BIAS_EXACT * abs(log(bias) - log(target_bias))**2
               - W_FIT * fit_penalty
               - floor penalties.
    """

    T_X = float(metrics.get("T_X", np.nan))
    T_Z = float(metrics.get("T_Z", np.nan))
    bias = float(metrics.get("bias", np.nan))
    fit_penalty = float(metrics.get("fit_penalty", 1.0e3))
    valid = bool(metrics.get("valid", False))

    finite = bool(np.isfinite(T_X) and np.isfinite(T_Z) and np.isfinite(bias))
    if not finite or T_X <= 0 or T_Z <= 0 or bias <= 0:
        return {
            "reward": -1.0e9,
            "loss_to_minimize": 1.0e9,
            "is_feasible": False,
            "bias_shortfall": np.inf,
            "bias_error": np.inf,
            "lifetime_score": -np.inf,
            "floor_penalty": 1.0e6,
        }

    log_eta = math.log(bias)
    log_target = math.log(cfg.target_bias)
    bias_shortfall = max(0.0, log_target - log_eta)
    bias_error = abs(log_eta - log_target)
    bias_rel_error = abs(bias / cfg.target_bias - 1.0)
    lifetime_score = 0.5 * (math.log(T_X) + math.log(T_Z))
    is_within_band = bool(bias_rel_error <= cfg.bias_tol_rel)
    if cfg.variant == "lower_bound":
        is_feasible = bool(valid and bias >= cfg.target_bias * (1.0 - cfg.bias_tol_rel))
    else:
        is_feasible = bool(valid and is_within_band)

    floor_penalty = cfg.floor_weight * (
        max(0.0, math.log(cfg.min_tx / T_X)) ** 2
        + max(0.0, math.log(cfg.min_tz / T_Z)) ** 2
    )
    if cfg.variant == "lower_bound":
        reward = (
            cfg.feasibility_bonus * float(bias >= cfg.target_bias)
            + cfg.w_lifetime * lifetime_score
            - cfg.w_bias_under * bias_shortfall**2
            - cfg.w_fit * fit_penalty
            - floor_penalty
        )
    elif cfg.variant in ("exact_target", "target_band"):
        reward = (
            cfg.feasibility_bonus * float(is_within_band)
            + cfg.w_lifetime * lifetime_score
            - cfg.w_bias_exact * bias_error**2
            - cfg.w_fit * fit_penalty
            - floor_penalty
        )
    else:
        raise ValueError("variant must be 'lower_bound', 'exact_target', or 'target_band'")

    loss_to_minimize = -float(reward)
    return {
        "reward": float(reward),
        "loss_to_minimize": loss_to_minimize,
        "is_feasible": is_feasible,
        "bias_shortfall": float(bias_shortfall),
        "bias_error": float(bias_error),
        "bias_rel_error": float(bias_rel_error),
        "lifetime_score": float(lifetime_score),
        "floor_penalty": float(floor_penalty),
    }


def default_reward_sweep(target_bias: float, bias_tol_rel: float) -> list[RewardConfig]:
    """Small M1-friendly sweep over exact-target / target-band rewards."""

    return [
        RewardConfig(
            name="target_band_strict",
            variant="target_band",
            target_bias=target_bias,
            bias_tol_rel=bias_tol_rel,
            w_lifetime=0.35,
            w_bias_exact=160.0,
            feasibility_bonus=18.0,
            w_fit=2.0,
        ),
        RewardConfig(
            name="target_band_balanced",
            variant="target_band",
            target_bias=target_bias,
            bias_tol_rel=bias_tol_rel,
            w_lifetime=0.65,
            w_bias_exact=120.0,
            feasibility_bonus=16.0,
            w_fit=2.0,
        ),
        RewardConfig(
            name="exact_target_strict",
            variant="exact_target",
            target_bias=target_bias,
            bias_tol_rel=bias_tol_rel,
            w_lifetime=0.25,
            w_bias_exact=180.0,
            feasibility_bonus=0.0,
            w_fit=2.0,
        ),
        RewardConfig(
            name="target_band_lifetime",
            variant="target_band",
            target_bias=target_bias,
            bias_tol_rel=bias_tol_rel,
            w_lifetime=1.0,
            w_bias_exact=140.0,
            feasibility_bonus=14.0,
            w_fit=2.0,
        ),
    ]
