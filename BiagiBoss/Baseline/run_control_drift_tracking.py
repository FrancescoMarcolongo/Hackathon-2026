"""Run a deterministic slow control-drift tracking experiment."""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from cmaes import SepCMA

from cat_model import (
    SimulationConfig,
    clear_measure_cache,
    measure_lifetimes,
    params_to_complex,
)
from drift import (
    DriftConfig,
    apply_control_drift,
    drift_vector,
    true_optimum_command,
    verify_or_scale_true_path,
)
from drift_plotting import (
    plot_drift_bias_vs_epoch,
    plot_drift_effective_parameters_vs_epoch,
    plot_drift_lifetimes_vs_epoch,
    plot_drift_parameters_vs_epoch,
    plot_drift_reward_vs_epoch,
    plot_drift_signal_vs_epoch,
    plot_drift_tracking_error_vs_epoch,
)
from plotting import plot_decay_fit
from rewards import RewardConfig, compute_reward
from run_core_bias_optimization import BIAS_TOL_REL, BOUNDS, OPTIMIZATION_START_X, TARGET_BIAS
from validation import write_csv, write_json

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CONTROL_SUFFIXES = ("g2_real", "g2_imag", "eps_d_real", "eps_d_imag")
WARMUP_EPOCH = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a small code smoke test.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of drift epochs.")
    parser.add_argument("--population", type=int, default=None, help="SepCMA population size.")
    parser.add_argument("--sigma0", type=float, default=0.45, help="Initial SepCMA sigma.")
    parser.add_argument("--tracking-sigma-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-bias", type=float, default=TARGET_BIAS)
    parser.add_argument(
        "--reward-preset",
        choices=(
            "exact_target_strict",
            "target_band_strict",
            "target_band_lifetime",
            "target_band_tracking",
            "reference_signature_tracking",
        ),
        default="target_band_lifetime",
        help="Reward preset built from the baseline reward formula.",
    )
    parser.add_argument(
        "--reference-signature-weight",
        type=float,
        default=80.0,
        help="Penalty weight for matching stationary reference T_X/T_Z in reference_signature_tracking.",
    )
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--period-epochs", type=float, default=240.0)
    parser.add_argument(
        "--x-reference",
        type=float,
        nargs=4,
        default=None,
        metavar=("G2_RE", "G2_IM", "EPS_RE", "EPS_IM"),
        help="Optional stationary reference optimum for the selected simulation preset.",
    )
    parser.add_argument("--warmup-epoch", type=int, default=WARMUP_EPOCH)
    parser.add_argument(
        "--sim-preset",
        choices=("quick", "proxy", "medium", "final"),
        default=None,
        help="Simulation preset. Defaults to quick with --quick and medium otherwise.",
    )
    parser.add_argument(
        "--no-decay-snapshots",
        action="store_true",
        help="Skip optional epoch 0/mid/final decay-fit snapshots.",
    )
    return parser.parse_args()


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("dynamiqs", "jax", "cmaes", "scipy", "matplotlib", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return versions


def build_simulation_config(args: argparse.Namespace) -> SimulationConfig:
    preset = args.sim_preset or ("quick" if args.quick else "medium")
    if preset == "quick":
        return SimulationConfig(na=8, nb=3, t_final_x=1.0, t_final_z=90.0, n_points=24)
    if preset == "proxy":
        return SimulationConfig(na=10, nb=3, t_final_x=1.0, t_final_z=130.0, n_points=28)
    if preset == "medium":
        return SimulationConfig(na=12, nb=4, t_final_x=1.2, t_final_z=220.0, n_points=36)
    return SimulationConfig(na=15, nb=5, t_final_x=1.2, t_final_z=260.0, n_points=60)


def build_reward_config(target_bias: float, preset: str) -> RewardConfig:
    if preset == "exact_target_strict":
        return RewardConfig(
            name="exact_target_strict",
            variant="exact_target",
            target_bias=float(target_bias),
            bias_tol_rel=BIAS_TOL_REL,
            w_lifetime=0.25,
            w_bias_exact=180.0,
            feasibility_bonus=0.0,
            w_fit=2.0,
        )
    if preset == "target_band_strict":
        return RewardConfig(
            name="target_band_strict",
            variant="target_band",
            target_bias=float(target_bias),
            bias_tol_rel=BIAS_TOL_REL,
            w_lifetime=0.35,
            w_bias_exact=160.0,
            feasibility_bonus=18.0,
            w_fit=2.0,
        )
    if preset == "target_band_lifetime":
        return RewardConfig(
            name="target_band_lifetime",
            variant="target_band",
            target_bias=float(target_bias),
            bias_tol_rel=BIAS_TOL_REL,
            w_lifetime=1.0,
            w_bias_exact=140.0,
            feasibility_bonus=14.0,
            w_fit=2.0,
        )
    if preset == "target_band_tracking":
        return RewardConfig(
            name="target_band_tracking",
            variant="target_band",
            target_bias=float(target_bias),
            bias_tol_rel=BIAS_TOL_REL,
            w_lifetime=2.0,
            w_bias_exact=160.0,
            feasibility_bonus=18.0,
            w_fit=2.0,
        )
    if preset == "reference_signature_tracking":
        return RewardConfig(
            name="reference_signature_tracking",
            variant="target_band",
            target_bias=float(target_bias),
            bias_tol_rel=BIAS_TOL_REL,
            w_lifetime=0.35,
            w_bias_exact=160.0,
            feasibility_bonus=18.0,
            w_fit=2.0,
        )
    raise ValueError(f"Unsupported reward preset: {preset}")


def build_drift_config(args: argparse.Namespace) -> DriftConfig:
    kwargs = {
        "period_epochs": float(args.period_epochs),
        "amplitude_scale": float(args.amplitude_scale),
    }
    if args.x_reference is not None:
        kwargs["x_reference"] = tuple(float(v) for v in args.x_reference)
    return DriftConfig(
        **kwargs,
    )


def optimizer_ask(opt: SepCMA) -> list[np.ndarray]:
    return [np.asarray(opt.ask(), dtype=float) for _ in range(opt.population_size)]


def optimizer_effective_sigma(opt: SepCMA) -> float:
    try:
        opt._eigen_decomposition()
    except Exception:
        pass
    sigma = float(getattr(opt, "_sigma", np.nan))
    diagonal_raw = getattr(opt, "_D", None)
    if diagonal_raw is None:
        return sigma
    diagonal = np.asarray(diagonal_raw, dtype=float)
    if diagonal.size == 0 or not np.all(np.isfinite(diagonal)):
        return sigma
    return float(sigma * np.min(diagonal))


def maybe_reheat_optimizer(
    opt: SepCMA,
    *,
    population: int,
    seed: int,
    epoch: int,
    sigma_floor: float,
) -> tuple[SepCMA, bool]:
    if sigma_floor <= 0:
        return opt, False
    effective_sigma = optimizer_effective_sigma(opt)
    if np.isfinite(effective_sigma) and effective_sigma >= sigma_floor and not opt.should_stop():
        return opt, False
    mean = np.minimum(np.maximum(np.asarray(opt.mean, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
    reheated = SepCMA(
        mean=mean,
        sigma=float(sigma_floor),
        bounds=BOUNDS,
        population_size=int(population),
        seed=int(seed + 100_000 + epoch),
    )
    return reheated, True


def evaluate_x_drift(
    x_cmd: np.ndarray,
    *,
    epoch: int,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    drift_cfg: DriftConfig,
    reference_metrics: dict[str, object] | None = None,
    reference_signature_weight: float = 0.0,
) -> dict:
    drift_state = apply_control_drift(x_cmd, epoch, drift_cfg, BOUNDS)
    g2_eff, eps_eff = params_to_complex(drift_state["x_eff"])
    metrics = measure_lifetimes(g2_eff, eps_eff, sim_cfg)
    reward = compute_reward(metrics, reward_cfg)
    signature_penalty = 0.0
    if reward_cfg.name == "reference_signature_tracking":
        if reference_metrics is None:
            raise ValueError("reference_signature_tracking requires reference_metrics.")
        signature_penalty = reference_signature_penalty(
            metrics,
            reference_metrics,
            weight=float(reference_signature_weight),
        )
        reward = dict(reward)
        reward["reward"] = float(reward["reward"]) - signature_penalty
        reward["loss_to_minimize"] = -float(reward["reward"])
    row = {
        "x": drift_state["x_cmd_clipped"],
        "x_cmd": drift_state["x_cmd_clipped"],
        "x_eff": drift_state["x_eff"],
        "drift": drift_state["drift"],
        "x_true_opt": drift_state["x_true_opt"],
        "tracking_error": drift_state["tracking_error"],
        "effective_error": drift_state["effective_error"],
        "g2": g2_eff,
        "eps_d": eps_eff,
        "reward": float(reward["reward"]),
        "loss_to_minimize": float(reward["loss_to_minimize"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_shortfall": float(reward["bias_shortfall"]),
        "bias_error": float(reward["bias_error"]),
        "bias_rel_error": float(reward["bias_rel_error"]),
        "reference_signature_penalty": float(signature_penalty),
    }
    for key in (
        "T_X",
        "T_Z",
        "bias",
        "geo_lifetime",
        "alpha_abs",
        "nbar",
        "fit_penalty",
        "fit_x_r2",
        "fit_z_r2",
        "fit_ok",
        "valid",
    ):
        row[key] = metrics[key]
    return row


def run_drift_tracking(
    *,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    drift_cfg: DriftConfig,
    epochs: int,
    population: int,
    sigma0: float,
    seed: int,
    tracking_sigma_floor: float,
    reference_metrics: dict[str, object] | None,
    reference_signature_weight: float,
) -> tuple[list[dict], list[dict], dict]:
    clear_measure_cache()
    mean0 = OPTIMIZATION_START_X.copy() + drift_vector(0, drift_cfg)
    mean0 = np.minimum(np.maximum(mean0, BOUNDS[:, 0]), BOUNDS[:, 1])
    optimizer = SepCMA(
        mean=mean0,
        sigma=float(sigma0),
        bounds=BOUNDS,
        population_size=int(population),
        seed=int(seed),
    )

    history: list[dict] = []
    samples: list[dict] = []
    previous_epoch_best_x = mean0.copy()
    reheats = 0
    start = time.time()

    start_eval = evaluate_x_drift(
        mean0,
        epoch=0,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        drift_cfg=drift_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=reference_signature_weight,
    )
    start_stats = _population_stats([start_eval])
    history.append(
        _history_row(
            0,
            mean_eval=start_eval,
            epoch_best=start_eval,
            population_stats=start_stats,
            optimizer=optimizer,
            sigma0=sigma0,
            seed=seed,
            reheated=False,
        )
    )
    samples.append(_sample_row(0, 0, "initial_mean", start_eval))

    for epoch in range(1, int(epochs) + 1):
        xs = optimizer_ask(optimizer)
        roles = ["sample"] * len(xs)
        if xs:
            xs[0] = np.asarray(optimizer.mean, dtype=float)
            roles[0] = "optimizer_mean"
        if len(xs) >= 2 and previous_epoch_best_x is not None:
            xs[1] = np.asarray(previous_epoch_best_x, dtype=float)
            roles[1] = "previous_epoch_best"

        evaluated = [
            evaluate_x_drift(
                x,
                epoch=epoch,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                drift_cfg=drift_cfg,
                reference_metrics=reference_metrics,
                reference_signature_weight=reference_signature_weight,
            )
            for x in xs
        ]
        for idx, (role, item) in enumerate(zip(roles, evaluated)):
            samples.append(_sample_row(epoch, idx, role, item))

        solutions = [
            (np.asarray(item["x"], dtype=float), float(item["loss_to_minimize"]))
            for item in evaluated
        ]
        optimizer.tell(solutions)
        optimizer, reheated = maybe_reheat_optimizer(
            optimizer,
            population=population,
            seed=seed,
            epoch=epoch,
            sigma_floor=tracking_sigma_floor,
        )
        if reheated:
            reheats += 1

        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        previous_epoch_best_x = np.asarray(epoch_best["x"], dtype=float)
        mean_eval = evaluate_x_drift(
            np.asarray(optimizer.mean, dtype=float),
            epoch=epoch,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            drift_cfg=drift_cfg,
            reference_metrics=reference_metrics,
            reference_signature_weight=reference_signature_weight,
        )
        population_stats = _population_stats(evaluated)
        history.append(
            _history_row(
                epoch,
                mean_eval=mean_eval,
                epoch_best=epoch_best,
                population_stats=population_stats,
                optimizer=optimizer,
                sigma0=sigma0,
                seed=seed,
                reheated=reheated,
            )
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = history[-1]
            elapsed = time.time() - start
            print(
                f"drift epoch={epoch:03d}/{epochs} "
                f"mean bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"track_l2={row['tracking_error_l2']:.4g} "
                f"sigma_eff={row['optimizer_effective_sigma']:.4g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    metadata = {
        "initial_mean": mean0.tolist(),
        "reheat_count": reheats,
        "tracking_sigma_floor": float(tracking_sigma_floor),
    }
    return history, samples, metadata


def _population_stats(evaluated: Iterable[dict]) -> dict:
    data = list(evaluated)
    reward = np.array([float(item["reward"]) for item in data], dtype=float)
    bias = np.array([float(item["bias"]) for item in data], dtype=float)
    return {
        "population_reward_mean": _nan_stat(np.nanmean, reward),
        "population_reward_std": _nan_stat(np.nanstd, reward),
        "population_bias_mean": _nan_stat(np.nanmean, bias),
        "population_bias_std": _nan_stat(np.nanstd, bias),
    }


def _nan_stat(func, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(func(finite))


def _flatten_vector(row: dict, prefix: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    for suffix, value in zip(CONTROL_SUFFIXES, values):
        row[f"{prefix}_{suffix}"] = float(value)


def _history_row(
    epoch: int,
    *,
    mean_eval: dict,
    epoch_best: dict,
    population_stats: dict,
    optimizer: SepCMA,
    sigma0: float,
    seed: int,
    reheated: bool,
) -> dict:
    row: dict[str, object] = {
        "epoch": int(epoch),
        "reward": float(mean_eval["reward"]),
        "loss": float(mean_eval["loss_to_minimize"]),
        "T_X": float(mean_eval["T_X"]),
        "T_Z": float(mean_eval["T_Z"]),
        "bias": float(mean_eval["bias"]),
        "geo_lifetime": float(mean_eval["geo_lifetime"]),
        "is_feasible": int(bool(mean_eval["is_feasible"])),
        "bias_error": float(mean_eval["bias_error"]),
        "bias_rel_error": float(mean_eval["bias_rel_error"]),
        "fit_penalty": float(mean_eval["fit_penalty"]),
        "fit_x_r2": float(mean_eval["fit_x_r2"]),
        "fit_z_r2": float(mean_eval["fit_z_r2"]),
        "reference_signature_penalty": float(mean_eval["reference_signature_penalty"]),
        "epoch_best_reward": float(epoch_best["reward"]),
        "epoch_best_loss": float(epoch_best["loss_to_minimize"]),
        "epoch_best_T_X": float(epoch_best["T_X"]),
        "epoch_best_T_Z": float(epoch_best["T_Z"]),
        "epoch_best_bias": float(epoch_best["bias"]),
        "epoch_best_is_feasible": int(bool(epoch_best["is_feasible"])),
        "population_reward_mean": float(population_stats["population_reward_mean"]),
        "population_reward_std": float(population_stats["population_reward_std"]),
        "population_bias_mean": float(population_stats["population_bias_mean"]),
        "population_bias_std": float(population_stats["population_bias_std"]),
        "sigma0": float(sigma0),
        "seed": int(seed),
        "optimizer_generation": int(optimizer.generation),
        "optimizer_effective_sigma": optimizer_effective_sigma(optimizer),
        "reheated": int(bool(reheated)),
    }
    _flatten_vector(row, "command", np.asarray(mean_eval["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(mean_eval["x_eff"], dtype=float))
    _flatten_vector(row, "drift", np.asarray(mean_eval["drift"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(mean_eval["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(mean_eval["tracking_error"], dtype=float))
    _flatten_vector(row, "effective_error", np.asarray(mean_eval["effective_error"], dtype=float))
    _flatten_vector(row, "reference", np.asarray(mean_eval["x_true_opt"], dtype=float) - np.asarray(mean_eval["drift"], dtype=float))
    _flatten_vector(row, "epoch_best_command", np.asarray(epoch_best["x_cmd"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(mean_eval["tracking_error"], dtype=float)))
    row["effective_error_l2"] = float(np.linalg.norm(np.asarray(mean_eval["effective_error"], dtype=float)))
    return row


def _sample_row(epoch: int, candidate_id: int, role: str, item: dict) -> dict:
    row: dict[str, object] = {
        "epoch": int(epoch),
        "candidate_id": int(candidate_id),
        "role": role,
        "reward": float(item["reward"]),
        "loss": float(item["loss_to_minimize"]),
        "T_X": float(item["T_X"]),
        "T_Z": float(item["T_Z"]),
        "bias": float(item["bias"]),
        "is_feasible": int(bool(item["is_feasible"])),
        "fit_penalty": float(item["fit_penalty"]),
        "fit_x_r2": float(item["fit_x_r2"]),
        "fit_z_r2": float(item["fit_z_r2"]),
        "reference_signature_penalty": float(item["reference_signature_penalty"]),
    }
    _flatten_vector(row, "command", np.asarray(item["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(item["x_eff"], dtype=float))
    _flatten_vector(row, "drift", np.asarray(item["drift"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(item["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(item["tracking_error"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(item["tracking_error"], dtype=float)))
    return row


def write_decay_snapshots(
    history: list[dict],
    sim_cfg: SimulationConfig,
    *,
    skip_optional: bool,
) -> dict[str, str]:
    figures: dict[str, str] = {}
    if not history:
        return figures

    final = history[-1]
    final_result = _row_decay_result(final, sim_cfg)
    path = FIGURES / "drift_optimized_decay_fits_final.png"
    plot_decay_fit(final_result, path, "Drift-tracked final epoch decay fits")
    figures["drift_optimized_decay_fits_final"] = str(path)

    if skip_optional:
        return figures

    selected = [
        ("epoch_000", history[0]),
        ("epoch_mid", history[len(history) // 2]),
        ("epoch_final", final),
    ]
    for label, row in selected:
        result = _row_decay_result(row, sim_cfg)
        snap_path = FIGURES / f"drift_optimized_decay_fits_{label}.png"
        plot_decay_fit(result, snap_path, f"Drift decay fits {label}")
        figures[f"drift_optimized_decay_fits_{label}"] = str(snap_path)
    return figures


def _row_decay_result(row: dict, sim_cfg: SimulationConfig) -> dict:
    x_eff = np.array(
        [
            row["effective_g2_real"],
            row["effective_g2_imag"],
            row["effective_eps_d_real"],
            row["effective_eps_d_imag"],
        ],
        dtype=float,
    )
    g2, eps_d = params_to_complex(x_eff)
    return measure_lifetimes(g2, eps_d, sim_cfg, return_curves=True)


def compute_summary_metrics(
    history: list[dict],
    *,
    warmup_epoch: int,
    target_bias: float,
    bias_tol_rel: float,
) -> dict:
    post = [row for row in history if int(row["epoch"]) >= int(warmup_epoch)]
    if not post:
        post = list(history)
    bias = np.array([float(row["bias"]) for row in post], dtype=float)
    tx = np.array([float(row["T_X"]) for row in post], dtype=float)
    tz = np.array([float(row["T_Z"]) for row in post], dtype=float)
    tracking = np.array([float(row["tracking_error_l2"]) for row in post], dtype=float)
    fit_penalty = np.array([float(row["fit_penalty"]) for row in post], dtype=float)
    fit_x = np.array([float(row["fit_x_r2"]) for row in post], dtype=float)
    fit_z = np.array([float(row["fit_z_r2"]) for row in post], dtype=float)
    lower = target_bias * (1.0 - bias_tol_rel)
    upper = target_bias * (1.0 + bias_tol_rel)
    in_band = (bias >= lower) & (bias <= upper)
    final = history[-1]
    return {
        "warmup_epoch": int(warmup_epoch),
        "final_bias": float(final["bias"]),
        "median_bias_after_warmup": _finite_median(bias),
        "fraction_post_warmup_in_success_band": float(np.mean(in_band[np.isfinite(bias)]))
        if np.any(np.isfinite(bias))
        else float("nan"),
        "success_band": [float(lower), float(upper)],
        "median_tracking_error_l2_after_warmup": _finite_median(tracking),
        "final_tracking_error_l2": float(final["tracking_error_l2"]),
        "median_T_X_after_warmup": _finite_median(tx),
        "median_T_Z_after_warmup": _finite_median(tz),
        "median_fit_penalty_after_warmup": _finite_median(fit_penalty),
        "median_fit_x_r2_after_warmup": _finite_median(fit_x),
        "median_fit_z_r2_after_warmup": _finite_median(fit_z),
        "final_fit_penalty": float(final["fit_penalty"]),
        "final_fit_x_r2": float(final["fit_x_r2"]),
        "final_fit_z_r2": float(final["fit_z_r2"]),
        "tracking_good": bool(
            np.isfinite(_finite_median(tracking))
            and _finite_median(tracking) < 0.35
            and (
                float(np.mean(in_band[np.isfinite(bias)]))
                if np.any(np.isfinite(bias))
                else 0.0
            )
            >= 0.70
        ),
    }


def reference_signature_penalty(
    metrics: dict[str, object],
    reference_metrics: dict[str, object],
    *,
    weight: float,
) -> float:
    if weight <= 0:
        return 0.0
    penalty = 0.0
    for key in ("T_X", "T_Z"):
        value = float(metrics.get(key, np.nan))
        reference = float(reference_metrics.get(key, np.nan))
        if not np.isfinite(value) or not np.isfinite(reference) or value <= 0 or reference <= 0:
            return 1.0e6
        penalty += math.log(value / reference) ** 2
    return float(weight * penalty)


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.median(finite))


def write_report(
    path: Path,
    *,
    command: str,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    drift_cfg: DriftConfig,
    bounds_check: dict,
    summary: dict,
    optimizer_metadata: dict,
    figures: dict[str, str],
) -> None:
    x_ref = np.asarray(drift_cfg.x_reference, dtype=float)
    default_x_ref = np.asarray(DriftConfig().x_reference, dtype=float)
    amplitudes = np.asarray(drift_cfg.amplitudes, dtype=float) * float(drift_cfg.amplitude_scale)
    highest_period = float(drift_cfg.period_epochs) / float(max(1, drift_cfg.bandwidth))
    good_text = (
        "Yes. The post-warm-up command path tracks the dashed true optimum closely enough "
        "to keep the effective controls near the stationary reference and maintain the "
        "bias in the target band for most post-warm-up epochs."
        if summary["tracking_good"]
        else "Not fully. The run completed and produced real simulation outputs, but the "
        "post-warm-up tracking or target-band fraction is below the acceptance threshold."
    )

    lines = [
        "# Slow Fourier control-drift tracking validation",
        "",
        "## Reproduction command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Package versions",
        "",
    ]
    for name, version in package_versions().items():
        lines.append(f"- {name}: {version}")
    lines += [
        "",
        "## 1. Drift chosen",
        "",
        "The experiment uses deterministic, epoch-level, four-dimensional control drift:",
        "",
        "```text",
        "d_i(e) = A_i [0.70 sin(2*pi*e/P + phi1_i) + 0.30 sin(4*pi*e/P + phi2_i)]",
        "x_eff(e) = x_cmd(e) - d(e)",
        "```",
        "",
        f"- P: `{drift_cfg.period_epochs:g}` epochs",
        f"- bandwidth: `{drift_cfg.bandwidth}` harmonics",
        f"- amplitude scale: `{drift_cfg.amplitude_scale:g}`",
        f"- effective amplitudes A: `{_fmt_vec(amplitudes)}`",
        f"- phi1: `{_fmt_vec(drift_cfg.phi1)}`",
        f"- phi2: `{_fmt_vec(drift_cfg.phi2)}`",
        f"- x_ref: `{_fmt_vec(x_ref)}`",
        "",
        "The components are ordered as Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d).",
        "",
        *(
            [
                "Reference note: the `DriftConfig` default preserves the supplied "
                f"full-truncation baseline point `{_fmt_vec(default_x_ref)}`. This "
                "reproducible medium-preset run passes an explicit calibrated stationary "
                "reference with `--x-reference`, because the medium truncation has a "
                "different control representative with the same target-band lifetime "
                "signature.",
                "",
            ]
            if not np.allclose(x_ref, default_x_ref)
            else []
        ),
        "Bounds check over the planned epoch range:",
        "",
        "```json",
        _json_dumps(bounds_check),
        "```",
        "",
        "## 2. Why this drift is slow",
        "",
        "The no-drift convergence scale is about 15 optimizer epochs. The fundamental "
        f"drift period is `{drift_cfg.period_epochs:g}` epochs and the highest-harmonic "
        f"period is `{highest_period:g}` epochs. Both are much larger than 15 epochs, "
        "so the drift is adiabatic relative to the optimizer convergence scale.",
        "",
        "## 3. Why this model",
        "",
        "This is a generic low-bandwidth control drift on the four physical calibration "
        "knobs, directly analogous to the pi-pulse example where the measured knob is "
        "the commanded knob minus drift. In this cat-qubit setting it represents slow "
        "amplitude and phase offsets in the complex g2 and epsilon_d controls.",
        "",
        "More physical drifts such as Kerr drift, storage detuning, TLS coupling, or SNR "
        "degradation are useful next benchmarks, but they do not provide an analytically "
        "obvious four-dimensional optimal command path. This control-drift benchmark is "
        "chosen first because the correct moving optimum is known exactly.",
        "",
        "## 4. True optimal functional form",
        "",
        "The true optimal command is:",
        "",
        "```text",
        "x_opt_true(e) = x_ref + d(e)",
        "```",
        "",
        "If the optimizer commands this path, the physical Hamiltonian receives "
        "`x_eff = x_ref`, so the lifetimes and bias should match the stationary no-drift "
        "optimized behavior.",
        "",
        "## 5. What the optimizer sees",
        "",
        "The optimizer sees only measured rewards, T_X, T_Z, and bias under the current "
        "epoch's drift. The drift vector and true optimum are used by the simulator and "
        "for plotting/reporting only; they are not used to choose candidates.",
        "",
        "For `reference_signature_tracking`, the scalar reward also penalizes measured "
        "T_X/T_Z deviations from the stationary no-drift reference signature. This uses "
        "only measured lifetime outputs and a fixed no-drift calibration target; it does "
        "not reveal d(e) or x_ref + d(e) to the optimizer.",
        "",
        "## 6. Plots produced",
        "",
    ]
    for label, fig_path in figures.items():
        lines.append(f"- {label}: `{fig_path}`")
    lines += [
        "",
        "The parameter plot is the key diagnostic: solid lines are optimizer commands "
        "and dashed lines are the analytically known true command optimum x_ref + d(e).",
        "",
        "## 7. Is the result good?",
        "",
        good_text,
        "",
        "Quantitative post-warm-up metrics:",
        "",
        f"- warm-up epoch: `{summary['warmup_epoch']}`",
        f"- final bias: `{summary['final_bias']:.6g}`",
        f"- median bias after warm-up: `{summary['median_bias_after_warmup']:.6g}`",
        "- fraction of post-warm-up epochs with 97 <= eta <= 103: "
        f"`{summary['fraction_post_warmup_in_success_band']:.4g}`",
        "- median tracking error L2 after warm-up: "
        f"`{summary['median_tracking_error_l2_after_warmup']:.6g}`",
        f"- final tracking error L2: `{summary['final_tracking_error_l2']:.6g}`",
        f"- median T_X after warm-up: `{summary['median_T_X_after_warmup']:.6g}` us",
        f"- median T_Z after warm-up: `{summary['median_T_Z_after_warmup']:.6g}` us",
        f"- median fit penalty after warm-up: `{summary['median_fit_penalty_after_warmup']:.6g}`",
        f"- median fit R2 after warm-up: X=`{summary['median_fit_x_r2_after_warmup']:.6g}`, "
        f"Z=`{summary['median_fit_z_r2_after_warmup']:.6g}`",
        "",
        "The decay fits are considered well conditioned when fit penalty remains near "
        "zero and both X/Z R2 values are close to one.",
        "",
        "Optimizer tracking details:",
        "",
        "```json",
        _json_dumps(optimizer_metadata),
        "```",
        "",
        "## 8. What to see in the graphs",
        "",
        "Expected good behavior is that the command curves converge to and follow the "
        "dashed true optimum curves, the drift signal remains smooth and slow, the "
        "effective physical parameters stay close to x_ref after warm-up, the bias stays "
        "near eta = 100 after warm-up, T_X and T_Z stay near the no-drift optimized "
        "values with small variations, and reward stabilizes after initial convergence.",
        "",
        "## 9. Limitations",
        "",
        "This is a deterministic slow-drift control benchmark. It is not yet a stochastic "
        "measurement-noise benchmark, Hamiltonian-detuning benchmark, Kerr benchmark, or "
        "SNR-degradation benchmark.",
        "",
        "## 10. Next steps",
        "",
        "- Add storage detuning drift with an explicit detuning compensation knob.",
        "- Add Kerr drift and compare whether the four-command manifold is sufficient.",
        "- Add measurement SNR degradation and uncertainty-aware reward/proxy logic.",
        "- Compare this online SepCMA tracker against PPO or other online optimizers.",
        "",
        "## Run configuration",
        "",
        "```json",
        _json_dumps(
            {
                "simulation": asdict(sim_cfg),
                "reward": asdict(reward_cfg),
                "drift": asdict(drift_cfg),
            }
        ),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_vec(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8g}" for v in values) + "]"


def _json_dumps(payload: dict) -> str:
    import json

    return json.dumps(payload, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return str(obj)


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs if args.epochs is not None else (30 if args.quick else 160))
    population = int(args.population if args.population is not None else (4 if args.quick else 8))
    sim_cfg = build_simulation_config(args)
    reward_cfg = build_reward_config(float(args.target_bias), str(args.reward_preset))
    drift_cfg_raw = build_drift_config(args)
    drift_cfg, bounds_check = verify_or_scale_true_path(drift_cfg_raw, BOUNDS, epochs)

    print(f"Drift simulation config: {asdict(sim_cfg)}")
    print(f"Reward config: {asdict(reward_cfg)}")
    print(f"Drift config: {asdict(drift_cfg)}")
    print(f"Bounds check: {bounds_check}")
    print(f"epochs={epochs}, population={population}, sigma0={args.sigma0}")
    if args.quick:
        print("Quick mode is a code smoke test; final scientific plots should use medium or final.")

    reference_metrics = None
    if reward_cfg.name == "reference_signature_tracking":
        reference_g2, reference_eps = params_to_complex(np.asarray(drift_cfg.x_reference, dtype=float))
        reference_metrics = measure_lifetimes(reference_g2, reference_eps, sim_cfg)
        print(
            "Reference signature: "
            f"Tx={float(reference_metrics['T_X']):.5g} "
            f"Tz={float(reference_metrics['T_Z']):.5g} "
            f"bias={float(reference_metrics['bias']):.5g}"
        )

    history, samples, optimizer_metadata = run_drift_tracking(
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        drift_cfg=drift_cfg,
        epochs=epochs,
        population=population,
        sigma0=float(args.sigma0),
        seed=int(args.seed),
        tracking_sigma_floor=float(args.tracking_sigma_floor),
        reference_metrics=reference_metrics,
        reference_signature_weight=float(args.reference_signature_weight),
    )

    summary = compute_summary_metrics(
        history,
        warmup_epoch=int(args.warmup_epoch),
        target_bias=float(args.target_bias),
        bias_tol_rel=BIAS_TOL_REL,
    )
    summary.update(
        {
            "target_bias": float(args.target_bias),
            "period_epochs": float(drift_cfg.period_epochs),
            "highest_harmonic_period_epochs": float(drift_cfg.period_epochs)
            / float(max(1, drift_cfg.bandwidth)),
            "epochs": epochs,
            "population": population,
            "simulation": asdict(sim_cfg),
            "reward_config": asdict(reward_cfg),
            "drift_config": asdict(drift_cfg),
            "bounds_check": bounds_check,
            "optimizer_metadata": optimizer_metadata,
            "reference_metrics": reference_metrics or {},
            "reference_signature_weight": float(args.reference_signature_weight)
            if reward_cfg.name == "reference_signature_tracking"
            else 0.0,
        }
    )

    write_csv(RESULTS / "drift_optimization_history.csv", history)
    write_csv(RESULTS / "drift_epoch_samples.csv", samples)
    write_json(RESULTS / "drift_summary_metrics.json", summary)
    write_json(
        RESULTS / "drift_final_candidate.json",
        {
            "final_mean": history[-1],
            "true_optimum_final": true_optimum_command(epochs, drift_cfg).tolist(),
            "drift_final": drift_vector(epochs, drift_cfg).tolist(),
        },
    )

    figures = {
        "drift_bias_vs_epoch": str(FIGURES / "drift_bias_vs_epoch.png"),
        "drift_lifetimes_vs_epoch": str(FIGURES / "drift_lifetimes_vs_epoch.png"),
        "drift_reward_or_loss_vs_epoch": str(FIGURES / "drift_reward_or_loss_vs_epoch.png"),
        "drift_parameters_vs_epoch": str(FIGURES / "drift_parameters_vs_epoch.png"),
        "drift_signal_vs_epoch": str(FIGURES / "drift_signal_vs_epoch.png"),
        "drift_tracking_error_vs_epoch": str(FIGURES / "drift_tracking_error_vs_epoch.png"),
        "drift_effective_parameters_vs_epoch": str(FIGURES / "drift_effective_parameters_vs_epoch.png"),
    }
    plot_drift_bias_vs_epoch(
        history,
        float(args.target_bias),
        BIAS_TOL_REL,
        int(args.warmup_epoch),
        FIGURES / "drift_bias_vs_epoch.png",
    )
    plot_drift_lifetimes_vs_epoch(history, FIGURES / "drift_lifetimes_vs_epoch.png")
    plot_drift_reward_vs_epoch(history, FIGURES / "drift_reward_or_loss_vs_epoch.png")
    plot_drift_parameters_vs_epoch(history, FIGURES / "drift_parameters_vs_epoch.png")
    plot_drift_signal_vs_epoch(history, FIGURES / "drift_signal_vs_epoch.png")
    plot_drift_tracking_error_vs_epoch(history, FIGURES / "drift_tracking_error_vs_epoch.png")
    plot_drift_effective_parameters_vs_epoch(
        history, FIGURES / "drift_effective_parameters_vs_epoch.png"
    )
    figures.update(
        write_decay_snapshots(
            history,
            sim_cfg,
            skip_optional=bool(args.no_decay_snapshots),
        )
    )

    command = "python run_control_drift_tracking.py"
    if args.quick:
        command += " --quick"
    if args.epochs is not None:
        command += f" --epochs {args.epochs}"
    if args.population is not None:
        command += f" --population {args.population}"
    if args.sim_preset is not None:
        command += f" --sim-preset {args.sim_preset}"
    if args.reward_preset != "target_band_lifetime":
        command += f" --reward-preset {args.reward_preset}"
    if args.reference_signature_weight != 80.0:
        command += f" --reference-signature-weight {args.reference_signature_weight}"
    if args.sigma0 != 0.45:
        command += f" --sigma0 {args.sigma0}"
    if args.tracking_sigma_floor != 0.05:
        command += f" --tracking-sigma-floor {args.tracking_sigma_floor}"
    if args.amplitude_scale != 1.0:
        command += f" --amplitude-scale {args.amplitude_scale}"
    if args.period_epochs != 240.0:
        command += f" --period-epochs {args.period_epochs}"
    if args.x_reference is not None:
        command += " --x-reference " + " ".join(f"{float(v):.8g}" for v in args.x_reference)

    write_report(
        ROOT / "drift_validation_report.md",
        command=command,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        drift_cfg=drift_cfg,
        bounds_check=bounds_check,
        summary=summary,
        optimizer_metadata=optimizer_metadata,
        figures=figures,
    )

    print("\nDrift tracking summary")
    print(f"final bias={summary['final_bias']:.6g}")
    print(f"median post-warmup bias={summary['median_bias_after_warmup']:.6g}")
    print(
        "post-warmup success-band fraction="
        f"{summary['fraction_post_warmup_in_success_band']:.4g}"
    )
    print(
        "median post-warmup tracking L2="
        f"{summary['median_tracking_error_l2_after_warmup']:.6g}"
    )
    print(f"final tracking L2={summary['final_tracking_error_l2']:.6g}")
    print(f"Saved drift report to {ROOT / 'drift_validation_report.md'}")
    print(f"Saved drift figures to {FIGURES}")
    print(f"Saved drift results to {RESULTS}")


if __name__ == "__main__":
    main()
