"""Physics-informed online optimizers under storage-detuning drift.

The model is intentionally reduced and fast: Delta_env + Delta_d is the
effective storage detuning, which perturbs log-bias at first order and
reduces both lifetimes at second order. The optimizers see only noisy
measurements of bias/lifetimes/reward, never the true Delta_env.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np


TARGET_BIAS = 100.0
TARGET_BAND_LOW = 97.0
TARGET_BAND_HIGH = 103.0
EPS = 1.0e-12


@dataclass(frozen=True)
class DetuningResponseConfig:
    tx_ref: float = 0.325
    tz_ref: float = 32.50
    bias_sensitivity: float = 11.0
    bias_cubic: float = 14.0
    lifetime_detuning_penalty: float = 400.0
    lifetime_floor: float = 0.55
    noise_sigma_log_bias: float = 0.0025
    noise_sigma_log_lifetime: float = 0.0025
    delta_bound: float = 0.075
    g2_real_ref: float = 1.6472
    g2_imag_ref: float = 0.5161
    eps_real_ref: float = 3.8076
    eps_imag_ref: float = -0.3302


@dataclass(frozen=True)
class RewardConfig:
    w_eta: float = 42.0
    w_lifetime: float = 1.25
    w_drop: float = 18.0
    w_delta: float = 0.16


@dataclass(frozen=True)
class OptimizerConfig:
    algorithm: str
    seed: int
    gain: float
    derivative_gain: float
    momentum: float
    proposal_noise: float
    update_clip: float
    use_blt: bool
    gate1_threshold: float
    gate2_threshold: float
    gate3_threshold: float
    jacobian_gain: float
    cost_per_epoch: float


BASELINE_CONFIG = OptimizerConfig(
    algorithm="baseline",
    seed=101,
    gain=0.55,
    derivative_gain=0.040,
    momentum=0.15,
    proposal_noise=0.0010,
    update_clip=0.018,
    use_blt=False,
    gate1_threshold=0.0,
    gate2_threshold=0.0,
    gate3_threshold=0.0,
    jacobian_gain=0.0,
    cost_per_epoch=2.0,
)


BLT_CONFIG = OptimizerConfig(
    algorithm="BLT",
    seed=101,
    gain=0.90,
    derivative_gain=0.34,
    momentum=0.10,
    proposal_noise=0.00055,
    update_clip=0.036,
    use_blt=True,
    gate1_threshold=0.90,
    gate2_threshold=0.060,
    gate3_threshold=0.035,
    jacobian_gain=1.35,
    cost_per_epoch=2.3,
)


def lifetime_score(tx: float, tz: float) -> float:
    return math.log(max(tx, EPS)) + math.log(max(tz, EPS))


def response_from_residual(
    delta_eff: float,
    *,
    response_cfg: DetuningResponseConfig,
    rng: np.random.Generator,
    noisy: bool = True,
) -> dict:
    """Map effective detuning Delta_eff to noisy two-point-like metrics."""
    clean_log_bias = (
        response_cfg.bias_sensitivity * delta_eff
        + response_cfg.bias_cubic * delta_eff * abs(delta_eff)
    )
    lifetime_factor = max(
        response_cfg.lifetime_floor,
        math.exp(-response_cfg.lifetime_detuning_penalty * delta_eff * delta_eff),
    )
    tx_clean = response_cfg.tx_ref * lifetime_factor * (1.0 + 0.025 * math.tanh(-5.0 * delta_eff))
    tz_clean = response_cfg.tz_ref * lifetime_factor * (1.0 + 0.018 * math.tanh(4.0 * delta_eff))
    if noisy:
        log_bias = clean_log_bias + float(rng.normal(0.0, response_cfg.noise_sigma_log_bias))
        tx = tx_clean * math.exp(float(rng.normal(0.0, response_cfg.noise_sigma_log_lifetime)))
        tz = tz_clean * math.exp(float(rng.normal(0.0, response_cfg.noise_sigma_log_lifetime)))
    else:
        log_bias = clean_log_bias
        tx = tx_clean
        tz = tz_clean
    bias = TARGET_BIAS * math.exp(log_bias)
    return {
        "bias": float(bias),
        "T_X": float(tx),
        "T_Z": float(tz),
        "log_bias_error": float(math.log(max(bias, EPS) / TARGET_BIAS)),
        "clean_bias": float(TARGET_BIAS * math.exp(clean_log_bias)),
        "clean_T_X": float(tx_clean),
        "clean_T_Z": float(tz_clean),
        "lifetime_score": float(lifetime_score(tx, tz)),
    }


def physics_reward(
    metrics: dict,
    *,
    delta_d: float,
    previous: dict | None,
    reward_cfg: RewardConfig,
) -> float:
    e_eta = float(metrics["log_bias_error"])
    score = float(metrics["lifetime_score"])
    drop_x = 0.0
    drop_z = 0.0
    if previous is not None:
        drop_x = max(0.0, math.log(max(float(previous["T_X"]), EPS) / max(float(metrics["T_X"]), EPS)))
        drop_z = max(0.0, math.log(max(float(previous["T_Z"]), EPS) / max(float(metrics["T_Z"]), EPS)))
    loss = (
        reward_cfg.w_eta * e_eta * e_eta
        - reward_cfg.w_lifetime * score
        + reward_cfg.w_drop * (drop_x * drop_x + drop_z * drop_z)
        + reward_cfg.w_delta * delta_d * delta_d
    )
    return float(-loss)


def update_type_for_blt(metrics: dict, epoch: int, cfg: OptimizerConfig) -> tuple[str, bool, bool, bool]:
    if not cfg.use_blt:
        return "BASELINE_UPDATE", False, False, False
    score = float(metrics["lifetime_score"])
    error = abs(float(metrics["log_bias_error"]))
    gate1 = score > cfg.gate1_threshold and epoch >= 3
    gate2 = gate1 and error < cfg.gate2_threshold
    gate3 = gate2 and error < cfg.gate3_threshold and epoch >= 7
    if gate3:
        return "BLT_JACOBIAN_UPDATE", gate1, gate2, gate3
    if gate2:
        return "BLT_REWARD_REGION", gate1, gate2, gate3
    return "PHYSICAL_UPDATE", gate1, gate2, gate3


def run_detuning_tracker(
    delta_env: np.ndarray,
    *,
    optimizer_cfg: OptimizerConfig,
    response_cfg: DetuningResponseConfig,
    reward_cfg: RewardConfig,
    burn_in_epochs: int,
) -> list[dict]:
    """Run one online tracker. Delta_env is used only by the environment."""
    rng = np.random.default_rng(optimizer_cfg.seed)
    delta_d = 0.0
    velocity = 0.0
    previous_residual_estimate = 0.0
    previous_accepted: dict | None = None
    cumulative_cost = 0.0
    rows: list[dict] = []

    for epoch, env in enumerate(np.asarray(delta_env, dtype=float)):
        delta_eff = float(env + delta_d)
        metrics = response_from_residual(delta_eff, response_cfg=response_cfg, rng=rng, noisy=True)
        reward = physics_reward(metrics, delta_d=delta_d, previous=previous_accepted, reward_cfg=reward_cfg)
        update_type, gate1, gate2, gate3 = update_type_for_blt(metrics, epoch, optimizer_cfg)
        cumulative_cost += optimizer_cfg.cost_per_epoch

        row = {
            "epoch": int(epoch),
            "algorithm": optimizer_cfg.algorithm,
            "Delta_env": float(env),
            "Delta_d_ideal": float(-env),
            "Delta_d_learned": float(delta_d),
            "Delta_eff": float(delta_eff),
            "bias": float(metrics["bias"]),
            "target_bias": TARGET_BIAS,
            "target_band_low": TARGET_BAND_LOW,
            "target_band_high": TARGET_BAND_HIGH,
            "tracking_error": abs(float(metrics["log_bias_error"])),
            "T_X": float(metrics["T_X"]),
            "T_Z": float(metrics["T_Z"]),
            "lifetime_score": float(metrics["lifetime_score"]),
            "reward": float(reward),
            "update_type": update_type,
            "gate1_pass": int(gate1),
            "gate2_pass": int(gate2),
            "gate3_pass": int(gate3),
            "cost_units": float(optimizer_cfg.cost_per_epoch),
            "cumulative_cost_units": float(cumulative_cost),
            "burn_in_flag": int(epoch < burn_in_epochs),
            "g2_real": response_cfg.g2_real_ref,
            "g2_imag": response_cfg.g2_imag_ref,
            "eps_real": response_cfg.eps_real_ref,
            "eps_imag": response_cfg.eps_imag_ref,
            "Delta_d_bound": response_cfg.delta_bound,
        }
        rows.append(row)
        previous_accepted = row

        residual_estimate = float(metrics["log_bias_error"]) / response_cfg.bias_sensitivity
        residual_velocity = residual_estimate - previous_residual_estimate
        previous_residual_estimate = residual_estimate

        effective_gain = optimizer_cfg.gain
        if update_type == "BLT_REWARD_REGION":
            effective_gain = max(effective_gain, 0.55)
        elif update_type == "BLT_JACOBIAN_UPDATE":
            effective_gain = optimizer_cfg.jacobian_gain

        raw_update = -effective_gain * residual_estimate - optimizer_cfg.derivative_gain * residual_velocity
        raw_update += float(rng.normal(0.0, optimizer_cfg.proposal_noise))
        raw_update = float(np.clip(raw_update, -optimizer_cfg.update_clip, optimizer_cfg.update_clip))
        velocity = optimizer_cfg.momentum * velocity + raw_update
        delta_d = float(np.clip(delta_d + velocity, -response_cfg.delta_bound, response_cfg.delta_bound))

    return rows


def compute_summary(rows_by_algorithm: dict[str, list[dict]], *, burn_in_epochs: int) -> list[dict]:
    summary: list[dict] = []
    for algorithm, rows in rows_by_algorithm.items():
        post = [row for row in rows if int(row["epoch"]) >= burn_in_epochs]
        errors = np.asarray([float(row["tracking_error"]) for row in post], dtype=float)
        biases = np.asarray([float(row["bias"]) for row in post], dtype=float)
        scores = np.asarray([float(row["lifetime_score"]) for row in post], dtype=float)
        in_band = (biases >= TARGET_BAND_LOW) & (biases <= TARGET_BAND_HIGH)
        recovery_time = ""
        for row in rows:
            epoch = int(row["epoch"])
            if epoch < burn_in_epochs:
                continue
            window = [r for r in rows if epoch <= int(r["epoch"]) < epoch + 8]
            if len(window) == 8 and all(TARGET_BAND_LOW <= float(r["bias"]) <= TARGET_BAND_HIGH for r in window):
                recovery_time = epoch
                break
        summary.append(
            {
                "algorithm": algorithm,
                "rms_log_bias_error": float(np.sqrt(np.mean(errors * errors))),
                "p_in_band": float(np.mean(in_band)),
                "mean_lifetime_score": float(np.mean(scores)),
                "recovery_time": recovery_time,
                "cumulative_cost": float(rows[-1]["cumulative_cost_units"]),
                "burn_in_epochs": int(burn_in_epochs),
                "n_epochs": len(rows),
            }
        )
    return summary


def configs_as_dict() -> dict:
    return {
        "target_bias": TARGET_BIAS,
        "target_band_low": TARGET_BAND_LOW,
        "target_band_high": TARGET_BAND_HIGH,
        "response": asdict(DetuningResponseConfig()),
        "reward": asdict(RewardConfig()),
        "baseline": asdict(BASELINE_CONFIG),
        "blt": asdict(BLT_CONFIG),
    }
