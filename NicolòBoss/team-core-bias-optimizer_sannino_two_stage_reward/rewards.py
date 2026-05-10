"""Two-stage reward for the core bias optimization challenge.

The original reward is dominated by bias matching.  This variant keeps that
pressure before the target band is reached, then increases lifetime pressure
inside the band so CMA-ES keeps searching for larger T_X and T_Z.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class TwoStageRewardConfig:
    name: str = "two_stage_lifetime_after_target"
    variant: str = "two_stage_target_band"
    target_bias: float = 100.0
    bias_tol_rel: float = 0.03
    w_lifetime_pre: float = 0.25
    w_lifetime_feasible: float = 2.0
    w_tx_feasible: float = 0.35
    w_tz_feasible: float = 0.35
    w_balanced_growth: float = 1.0
    w_bias_outside: float = 180.0
    w_bias_inside: float = 25.0
    w_fit: float = 2.0
    feasibility_bonus: float = 5.0
    min_tx: float = 0.05
    min_tz: float = 5.0
    floor_weight: float = 12.0
    reference_tx: float = 0.3062175295848265
    reference_tz: float = 12.356185559121926


def compute_reward(metrics: Dict[str, object], cfg: TwoStageRewardConfig) -> Dict[str, float | bool | str]:
    """Compute a two-stage reward and the loss consumed by SepCMA.

    Stage 1, outside the target band:
        strong exact-bias pressure, weak lifetime pressure.

    Stage 2, inside the target band:
        smaller exact-bias pressure plus stronger lifetime terms.  The
        additional balanced-growth term uses the weaker of Tx/ref_tx and
        Tz/ref_tz, which discourages improving only one lifetime.
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
            "stage": "invalid",
            "bias_error": np.inf,
            "bias_rel_error": np.inf,
            "lifetime_score": -np.inf,
            "post_target_lifetime_score": 0.0,
            "balanced_growth_score": -np.inf,
            "floor_penalty": 1.0e6,
        }

    log_eta = math.log(bias)
    log_target = math.log(cfg.target_bias)
    bias_error = abs(log_eta - log_target)
    bias_rel_error = abs(bias / cfg.target_bias - 1.0)
    is_within_band = bool(bias_rel_error <= cfg.bias_tol_rel)
    is_feasible = bool(valid and is_within_band)

    lifetime_score = 0.5 * (math.log(T_X) + math.log(T_Z))
    tx_growth = math.log(T_X / cfg.reference_tx) if cfg.reference_tx > 0 else 0.0
    tz_growth = math.log(T_Z / cfg.reference_tz) if cfg.reference_tz > 0 else 0.0
    balanced_growth_score = min(tx_growth, tz_growth)
    floor_penalty = cfg.floor_weight * (
        max(0.0, math.log(cfg.min_tx / T_X)) ** 2
        + max(0.0, math.log(cfg.min_tz / T_Z)) ** 2
    )

    bias_weight = cfg.w_bias_inside if is_within_band else cfg.w_bias_outside
    reward = (
        cfg.w_lifetime_pre * lifetime_score
        - bias_weight * bias_error**2
        - cfg.w_fit * fit_penalty
        - floor_penalty
    )
    post_target_lifetime_score = 0.0
    if is_within_band:
        post_target_lifetime_score = (
            cfg.w_lifetime_feasible * lifetime_score
            + cfg.w_tx_feasible * tx_growth
            + cfg.w_tz_feasible * tz_growth
            + cfg.w_balanced_growth * balanced_growth_score
        )
        reward += cfg.feasibility_bonus + post_target_lifetime_score

    return {
        "reward": float(reward),
        "loss_to_minimize": -float(reward),
        "is_feasible": is_feasible,
        "stage": "target_reached" if is_within_band else "seeking_target",
        "bias_error": float(bias_error),
        "bias_rel_error": float(bias_rel_error),
        "lifetime_score": float(lifetime_score),
        "post_target_lifetime_score": float(post_target_lifetime_score),
        "balanced_growth_score": float(balanced_growth_score),
        "floor_penalty": float(floor_penalty),
        "tx_growth": float(tx_growth),
        "tz_growth": float(tz_growth),
    }
