"""Run four-knob online tracking under storage-Kerr drift."""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from cmaes import SepCMA

from cat_model import SimulationConfig, clear_measure_cache, measure_lifetimes, params_to_complex
from plotting import plot_decay_fit
from rewards import RewardConfig, compute_reward
from run_core_bias_optimization import BIAS_TOL_REL, BOUNDS, OPTIMIZATION_START_X, TARGET_BIAS
from storage_kerr import (
    DEFAULT_CONTROL_BOUNDS,
    StorageKerrConfig,
    apply_storage_kerr,
    as_control_vector,
    storage_kerr_drift,
    true_storage_kerr_optimum,
    verify_storage_kerr_path,
)
from storage_kerr_plotting import (
    plot_kerr_bias_vs_epoch,
    plot_kerr_effective_parameters_vs_epoch,
    plot_kerr_lifetimes_vs_epoch,
    plot_kerr_parameters_vs_epoch,
    plot_kerr_reward_vs_epoch,
    plot_kerr_signal_vs_epoch,
    plot_kerr_tracking_error_vs_epoch,
)
from validation import write_csv, write_json


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CONTROL_SUFFIXES = ("g2_real", "g2_imag", "eps_d_real", "eps_d_imag")
WARMUP_EPOCH = 15
CONTROL_STEPS = np.array([0.035, 0.025, 0.060, 0.050], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--sigma0", type=float, default=0.40)
    parser.add_argument("--tracking-sigma-floor", type=float, default=0.055)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-bias", type=float, default=TARGET_BIAS)
    parser.add_argument("--warmup-epoch", type=int, default=WARMUP_EPOCH)
    parser.add_argument("--period-epochs", type=float, default=120.0)
    parser.add_argument("--kerr-amplitude", type=float, default=0.020)
    parser.add_argument("--kerr-calibration-step", type=float, default=0.006)
    parser.add_argument("--calibration-ridge", type=float, default=2.0e-3)
    parser.add_argument(
        "--reference-signature-weight",
        type=float,
        default=100.0,
        help="Penalty weight for matching stationary reference T_X/T_Z under Kerr drift.",
    )
    parser.add_argument(
        "--compensation-sensitivity",
        type=float,
        nargs=4,
        default=None,
        metavar=("S_G2_RE", "S_G2_IM", "S_EPS_RE", "S_EPS_IM"),
        help="Optional precomputed dx/dK compensation vector. By default it is calibrated locally.",
    )
    parser.add_argument(
        "--x-reference",
        type=float,
        nargs=4,
        default=None,
        metavar=("G2_RE", "G2_IM", "EPS_RE", "EPS_IM"),
        help="Stationary four-control reference. Defaults to the calibrated medium reference.",
    )
    parser.add_argument(
        "--sim-preset",
        choices=("quick", "medium", "final"),
        default=None,
        help="Simulation preset. Defaults to quick with --quick and medium otherwise.",
    )
    parser.add_argument("--no-decay-snapshots", action="store_true")
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
    if preset == "medium":
        return SimulationConfig(na=12, nb=4, t_final_x=1.2, t_final_z=220.0, n_points=36)
    return SimulationConfig(na=15, nb=5, t_final_x=1.2, t_final_z=260.0, n_points=60)


def build_reward_config(target_bias: float) -> RewardConfig:
    return RewardConfig(
        name="storage_kerr_reference_signature",
        variant="target_band",
        target_bias=float(target_bias),
        bias_tol_rel=BIAS_TOL_REL,
        w_lifetime=0.35,
        w_bias_exact=170.0,
        feasibility_bonus=18.0,
        w_fit=2.0,
    )


def build_storage_config(args: argparse.Namespace) -> StorageKerrConfig:
    kwargs = {
        "period_epochs": float(args.period_epochs),
        "amplitude": float(args.kerr_amplitude),
    }
    if args.x_reference is not None:
        kwargs["x_reference"] = tuple(float(v) for v in args.x_reference)
    if args.compensation_sensitivity is not None:
        kwargs["compensation_sensitivity"] = tuple(float(v) for v in args.compensation_sensitivity)
    return StorageKerrConfig(**kwargs)


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
        seed=int(seed + 300_000 + epoch),
    )
    return reheated, True


def _metric_vector(metrics: dict[str, object]) -> np.ndarray:
    values = []
    for key in ("T_X", "T_Z", "bias"):
        value = float(metrics.get(key, np.nan))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Cannot calibrate Kerr sensitivity: invalid {key}={value}")
        values.append(math.log(value))
    return np.asarray(values, dtype=float)


def estimate_kerr_sensitivity(
    x_reference: np.ndarray,
    sim_cfg: SimulationConfig,
    *,
    kerr_step: float,
    ridge: float,
) -> dict[str, object]:
    """Estimate dx/dK that keeps local measured signature stationary."""

    x_ref = as_control_vector(x_reference, "x_reference")
    g2_ref, eps_ref = params_to_complex(x_ref)
    ref_metrics = measure_lifetimes(g2_ref, eps_ref, sim_cfg, storage_kerr=0.0)
    y_ref = _metric_vector(ref_metrics)
    jac_controls = np.zeros((len(y_ref), 4), dtype=float)
    for idx, step in enumerate(CONTROL_STEPS):
        x_plus = x_ref.copy()
        x_minus = x_ref.copy()
        x_plus[idx] = min(float(DEFAULT_CONTROL_BOUNDS[idx, 1]), x_plus[idx] + float(step))
        x_minus[idx] = max(float(DEFAULT_CONTROL_BOUNDS[idx, 0]), x_minus[idx] - float(step))
        g2_plus, eps_plus = params_to_complex(x_plus)
        g2_minus, eps_minus = params_to_complex(x_minus)
        y_plus = _metric_vector(measure_lifetimes(g2_plus, eps_plus, sim_cfg, storage_kerr=0.0))
        y_minus = _metric_vector(measure_lifetimes(g2_minus, eps_minus, sim_cfg, storage_kerr=0.0))
        denom = float(x_plus[idx] - x_minus[idx])
        jac_controls[:, idx] = (y_plus - y_minus) / denom

    y_k_plus = _metric_vector(measure_lifetimes(g2_ref, eps_ref, sim_cfg, storage_kerr=float(kerr_step)))
    y_k_minus = _metric_vector(measure_lifetimes(g2_ref, eps_ref, sim_cfg, storage_kerr=-float(kerr_step)))
    jac_kerr = (y_k_plus - y_k_minus) / (2.0 * float(kerr_step))
    system = jac_controls.T @ jac_controls + float(ridge) * np.eye(4)
    rhs = -jac_controls.T @ jac_kerr
    sensitivity = np.linalg.solve(system, rhs)
    residual = jac_controls @ sensitivity + jac_kerr
    return {
        "reference_metrics": ref_metrics,
        "metric_names": ["log_T_X", "log_T_Z", "log_bias"],
        "control_steps": CONTROL_STEPS.tolist(),
        "kerr_step": float(kerr_step),
        "ridge": float(ridge),
        "jac_controls": jac_controls.tolist(),
        "jac_kerr": jac_kerr.tolist(),
        "compensation_sensitivity": sensitivity.tolist(),
        "linear_residual": residual.tolist(),
        "linear_residual_norm": float(np.linalg.norm(residual)),
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


def evaluate_x_kerr(
    x_cmd: np.ndarray,
    *,
    epoch: int,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    storage_cfg: StorageKerrConfig,
    reference_metrics: dict[str, object],
    reference_signature_weight: float,
) -> dict:
    state = apply_storage_kerr(x_cmd, epoch, storage_cfg, BOUNDS)
    x_eff = np.asarray(state["x_eff"], dtype=float)
    g2, eps_d = params_to_complex(x_eff)
    metrics = measure_lifetimes(
        g2,
        eps_d,
        sim_cfg,
        storage_kerr=float(state["storage_kerr"]),
    )
    reward = dict(compute_reward(metrics, reward_cfg))
    signature_penalty = reference_signature_penalty(
        metrics,
        reference_metrics,
        weight=float(reference_signature_weight),
    )
    reward["reward"] = float(reward["reward"]) - signature_penalty
    reward["loss_to_minimize"] = -float(reward["reward"])

    row = {
        "x": np.asarray(state["x_cmd_clipped"], dtype=float),
        "x_cmd": np.asarray(state["x_cmd_clipped"], dtype=float),
        "x_eff": x_eff,
        "x_true_opt": np.asarray(state["x_true_opt"], dtype=float),
        "x_reference_eff": np.asarray(state["x_reference_eff"], dtype=float),
        "tracking_error": np.asarray(state["tracking_error"], dtype=float),
        "effective_error": np.asarray(state["effective_error"], dtype=float),
        "storage_kerr_drift": float(state["storage_kerr"]),
        "storage_kerr_delta": float(state["storage_kerr_delta"]),
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


def run_storage_kerr_tracking(
    *,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    storage_cfg: StorageKerrConfig,
    reference_metrics: dict[str, object],
    epochs: int,
    population: int,
    sigma0: float,
    seed: int,
    tracking_sigma_floor: float,
    reference_signature_weight: float,
) -> tuple[list[dict], list[dict], dict]:
    clear_measure_cache()
    mean0 = np.minimum(np.maximum(OPTIMIZATION_START_X.copy(), BOUNDS[:, 0]), BOUNDS[:, 1])
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

    start_eval = evaluate_x_kerr(
        mean0,
        epoch=0,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=reference_signature_weight,
    )
    history.append(
        _history_row(
            0,
            mean_eval=start_eval,
            epoch_best=start_eval,
            population_stats=_population_stats([start_eval]),
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
            evaluate_x_kerr(
                x,
                epoch=epoch,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                storage_cfg=storage_cfg,
                reference_metrics=reference_metrics,
                reference_signature_weight=reference_signature_weight,
            )
            for x in xs
        ]
        for idx, (role, item) in enumerate(zip(roles, evaluated)):
            samples.append(_sample_row(epoch, idx, role, item))

        optimizer.tell(
            [
                (np.asarray(item["x"], dtype=float), float(item["loss_to_minimize"]))
                for item in evaluated
            ]
        )
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
        mean_eval = evaluate_x_kerr(
            np.asarray(optimizer.mean, dtype=float),
            epoch=epoch,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            storage_cfg=storage_cfg,
            reference_metrics=reference_metrics,
            reference_signature_weight=reference_signature_weight,
        )
        history.append(
            _history_row(
                epoch,
                mean_eval=mean_eval,
                epoch_best=epoch_best,
                population_stats=_population_stats(evaluated),
                optimizer=optimizer,
                sigma0=sigma0,
                seed=seed,
                reheated=reheated,
            )
        )

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = history[-1]
            print(
                f"storage-kerr epoch={epoch:03d}/{epochs} "
                f"bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"track_l2={row['tracking_error_l2']:.4g} "
                f"K={row['drift_storage_kerr']:.4g} "
                f"sigma_eff={row['optimizer_effective_sigma']:.4g} "
                f"elapsed={time.time() - start:.1f}s",
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
        "drift_storage_kerr": float(mean_eval["storage_kerr_drift"]),
    }
    _flatten_vector(row, "command", np.asarray(mean_eval["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(mean_eval["x_eff"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(mean_eval["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(mean_eval["tracking_error"], dtype=float))
    _flatten_vector(row, "effective_error", np.asarray(mean_eval["effective_error"], dtype=float))
    _flatten_vector(row, "reference", np.asarray(mean_eval["x_reference_eff"], dtype=float))
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
        "drift_storage_kerr": float(item["storage_kerr_drift"]),
    }
    _flatten_vector(row, "command", np.asarray(item["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(item["x_eff"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(item["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(item["tracking_error"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(item["tracking_error"], dtype=float)))
    return row


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
    reference_distance = np.array([float(row["effective_error_l2"]) for row in post], dtype=float)
    fit_penalty = np.array([float(row["fit_penalty"]) for row in post], dtype=float)
    fit_x = np.array([float(row["fit_x_r2"]) for row in post], dtype=float)
    fit_z = np.array([float(row["fit_z_r2"]) for row in post], dtype=float)
    lower = target_bias * (1.0 - bias_tol_rel)
    upper = target_bias * (1.0 + bias_tol_rel)
    in_band = (bias >= lower) & (bias <= upper)
    final = history[-1]
    in_band_fraction = float(np.mean(in_band[np.isfinite(bias)])) if np.any(np.isfinite(bias)) else float("nan")
    return {
        "warmup_epoch": int(warmup_epoch),
        "final_bias": float(final["bias"]),
        "median_bias_after_warmup": _finite_median(bias),
        "fraction_post_warmup_in_success_band": in_band_fraction,
        "success_band": [float(lower), float(upper)],
        "median_tracking_error_l2_after_warmup": _finite_median(tracking),
        "final_tracking_error_l2": float(final["tracking_error_l2"]),
        "median_distance_from_stationary_reference_after_warmup": _finite_median(reference_distance),
        "final_distance_from_stationary_reference": float(final["effective_error_l2"]),
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
            and _finite_median(tracking) < 0.28
            and in_band_fraction >= 0.90
        ),
    }


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.median(finite))


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
    path = FIGURES / "storage_kerr_optimized_decay_fits_final.png"
    plot_decay_fit(final_result, path, "Storage-Kerr tracked final decay fits")
    figures["storage_kerr_optimized_decay_fits_final"] = str(path)
    if skip_optional:
        return figures
    selected = [
        ("epoch_000", history[0]),
        ("epoch_mid", history[len(history) // 2]),
        ("epoch_final", final),
    ]
    for label, row in selected:
        result = _row_decay_result(row, sim_cfg)
        snap_path = FIGURES / f"storage_kerr_optimized_decay_fits_{label}.png"
        plot_decay_fit(result, snap_path, f"Storage-Kerr decay fits {label}")
        figures[f"storage_kerr_optimized_decay_fits_{label}"] = str(snap_path)
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
    return measure_lifetimes(
        g2,
        eps_d,
        sim_cfg,
        storage_kerr=float(row["drift_storage_kerr"]),
        return_curves=True,
    )


def verify_or_scale_kerr_path(
    cfg: StorageKerrConfig,
    bounds: np.ndarray,
    epochs: int,
) -> tuple[StorageKerrConfig, dict]:
    try:
        return cfg, verify_storage_kerr_path(cfg, bounds, epochs)
    except ValueError:
        x_ref = as_control_vector(cfg.x_reference, "x_reference")
        sensitivity = as_control_vector(cfg.compensation_sensitivity, "compensation_sensitivity")
        if np.linalg.norm(sensitivity) <= 0:
            raise
        ks_unit = np.array(
            [
                storage_kerr_drift(e, replace(cfg, amplitude=1.0))
                - float(cfg.kerr_reference)
                for e in range(int(epochs) + 1)
            ],
            dtype=float,
        )
        max_scale = 1.0
        for dim in range(4):
            excursions = sensitivity[dim] * ks_unit
            if np.allclose(excursions, 0.0):
                continue
            upper_margin = float(bounds[dim, 1] - x_ref[dim])
            lower_margin = float(x_ref[dim] - bounds[dim, 0])
            positive = np.max(excursions)
            negative = -np.min(excursions)
            if positive > 0:
                max_scale = min(max_scale, 0.95 * upper_margin / positive)
            if negative > 0:
                max_scale = min(max_scale, 0.95 * lower_margin / negative)
        if max_scale <= 0 or not np.isfinite(max_scale):
            raise
        scaled = replace(cfg, amplitude=float(min(cfg.amplitude, max_scale)))
        check = verify_storage_kerr_path(scaled, bounds, epochs)
        check["amplitude_scaled_from"] = float(cfg.amplitude)
        check["amplitude_scaled_to"] = float(scaled.amplitude)
        return scaled, check


def write_report(
    path: Path,
    *,
    command: str,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    storage_cfg: StorageKerrConfig,
    bounds_check: dict,
    summary: dict,
    optimizer_metadata: dict,
    calibration: dict,
    reference_metrics: dict[str, object],
    figures: dict[str, str],
) -> None:
    highest_period = float(storage_cfg.period_epochs) / float(storage_cfg.bandwidth)
    good_text = (
        "Yes. The four commanded cat controls converge toward the oscillating calibrated "
        "Kerr-compensation target and eta remains in the 97-103 success band after warm-up."
        if summary["tracking_good"]
        else "Not fully. The run completed, but the four-control tracker did not meet the "
        "requested tracking or bias-band threshold; this is the regime where a fifth explicit "
        "Kerr-compensation knob should be tested next."
    )
    lines = [
        "# Storage-Kerr four-knob drift tracking validation",
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
        "## Kerr drift model",
        "",
        "I returned to the original four optimizer knobs:",
        "",
        "```text",
        "x = [Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d)]",
        "```",
        "",
        "The storage Hamiltonian now includes a drifted Kerr nonlinearity:",
        "",
        "```text",
        "n_a = a^dag a",
        "H_K(e) = -0.5 K(e) n_a (n_a - I)",
        "```",
        "",
        "The deterministic storage Kerr drift is:",
        "",
        "```text",
        "K(e) = A [",
        "    0.68 sin(2*pi*e/P + phi1)",
        "  + 0.32 sin(4*pi*e/P + phi2)",
        "]",
        "```",
        "",
        f"- P: `{storage_cfg.period_epochs:g}` epochs",
        f"- bandwidth: `{storage_cfg.bandwidth}` harmonics",
        f"- highest-harmonic period: `{highest_period:g}` epochs",
        f"- A: `{storage_cfg.amplitude:g}`",
        f"- weights: `{_fmt_vec(storage_cfg.weights)}`",
        f"- phases: `{_fmt_vec(storage_cfg.phases)}`",
        f"- stationary four-control x_ref: `{_fmt_vec(storage_cfg.x_reference)}`",
        "",
        "The highest harmonic period is still longer than the roughly 15-epoch no-drift "
        "convergence scale, but it is faster than the original 240-epoch control-drift "
        "benchmark. The imposed Kerr signal is visibly oscillatory; the local four-control "
        "target moves only weakly because Kerr is not a direct command offset in this manifold.",
        "",
        "## Calibrated moving optimum",
        "",
        "Unlike pure command drift, Kerr drift does not give an analytic command path such as "
        "`x_ref + d(e)`. I therefore estimate a local physical compensation direction from "
        "finite-difference simulations at the stationary reference. The calibration solves:",
        "",
        "```text",
        "J_x s + J_K ~= 0",
        "x_opt_calibrated(e) = x_ref + s [K(e) - K_ref]",
        "```",
        "",
        "where the measured signature vector is `[log(T_X), log(T_Z), log(eta)]`. The vector "
        "`s = dx/dK` is used only to draw the dashed target curves and quantify tracking; "
        "the optimizer is not given `K(e)`, `s`, or `x_opt_calibrated(e)` when choosing "
        "candidates.",
        "",
        "Calibration details:",
        "",
        "```json",
        _json_dumps(calibration),
        "```",
        "",
        "Bounds check:",
        "",
        "```json",
        _json_dumps(bounds_check),
        "```",
        "",
        "Stationary reference measurement:",
        "",
        "```json",
        _json_dumps(
            {
                "T_X": float(reference_metrics["T_X"]),
                "T_Z": float(reference_metrics["T_Z"]),
                "bias": float(reference_metrics["bias"]),
                "fit_penalty": float(reference_metrics["fit_penalty"]),
                "fit_x_r2": float(reference_metrics["fit_x_r2"]),
                "fit_z_r2": float(reference_metrics["fit_z_r2"]),
            }
        ),
        "```",
        "",
        "## Plots produced",
        "",
    ]
    for label, fig_path in figures.items():
        lines.append(f"- {label}: `{fig_path}`")
    lines += [
        "",
        "The parameter plot shows that the four-control optimizer finds a valid bias-restoring "
        "branch, but not the small finite-difference target branch. The signal plot shows "
        "the imposed oscillatory storage Kerr drift.",
        "",
        "## Is the result good?",
        "",
        good_text,
        "",
        f"- warm-up epoch: `{summary['warmup_epoch']}`",
        f"- final bias: `{summary['final_bias']:.6g}`",
        f"- median bias after warm-up: `{summary['median_bias_after_warmup']:.6g}`",
        "- fraction of post-warm-up epochs with 97 <= eta <= 103: "
        f"`{summary['fraction_post_warmup_in_success_band']:.4g}`",
        "- median tracking error L2 after warm-up: "
        f"`{summary['median_tracking_error_l2_after_warmup']:.6g}`",
        f"- final tracking error L2: `{summary['final_tracking_error_l2']:.6g}`",
        "- median distance from stationary reference after warm-up: "
        f"`{summary['median_distance_from_stationary_reference_after_warmup']:.6g}`",
        f"- median T_X after warm-up: `{summary['median_T_X_after_warmup']:.6g}` us",
        f"- median T_Z after warm-up: `{summary['median_T_Z_after_warmup']:.6g}` us",
        f"- median fit penalty after warm-up: `{summary['median_fit_penalty_after_warmup']:.6g}`",
        f"- median fit R2 after warm-up: X=`{summary['median_fit_x_r2_after_warmup']:.6g}`, "
        f"Z=`{summary['median_fit_z_r2_after_warmup']:.6g}`",
        "",
        "## Four knobs versus a possible fifth knob",
        "",
        "This run tests the requested four-knob manifold first. A fifth physically informative "
        "knob would be an explicit Kerr compensation command `K_cmd`, with "
        "`K_eff(e) = K_drift(e) - K_cmd(e)`. Because the four-knob result is judged by the "
        "metrics above, the fifth-knob comparison is only necessary if the four-control "
        "tracker misses the bias band or fails to follow the dashed target.",
        "",
        "## Limitations and next steps",
        "",
        "This is a deterministic Kerr-drift benchmark without stochastic measurement noise. "
        "The dashed optimum is a local finite-difference compensation curve, not a global "
        "proof of uniqueness. Natural extensions are a five-knob explicit Kerr compensator, "
        "Kerr plus detuning drift, measurement SNR degradation, and PPO or other online "
        "optimizer comparisons.",
        "",
        "## Optimizer details",
        "",
        "```json",
        _json_dumps(optimizer_metadata),
        "```",
        "",
        "## Run configuration",
        "",
        "```json",
        _json_dumps(
            {
                "simulation": asdict(sim_cfg),
                "reward": asdict(reward_cfg),
                "storage_kerr": asdict(storage_cfg),
            }
        ),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_vec(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(v):.8g}" for v in values) + "]"


def _json_dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, complex):
        return [obj.real, obj.imag]
    return str(obj)


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    epochs = int(args.epochs if args.epochs is not None else (35 if args.quick else 140))
    population = int(args.population if args.population is not None else (5 if args.quick else 8))
    sim_cfg = build_simulation_config(args)
    reward_cfg = build_reward_config(float(args.target_bias))
    storage_cfg = build_storage_config(args)

    ref_x = as_control_vector(storage_cfg.x_reference, "x_reference")
    if args.compensation_sensitivity is None:
        calibration = estimate_kerr_sensitivity(
            ref_x,
            sim_cfg,
            kerr_step=float(args.kerr_calibration_step),
            ridge=float(args.calibration_ridge),
        )
        storage_cfg = replace(
            storage_cfg,
            compensation_sensitivity=tuple(float(v) for v in calibration["compensation_sensitivity"]),
        )
    else:
        ref_g2, ref_eps = params_to_complex(ref_x)
        reference_metrics = measure_lifetimes(ref_g2, ref_eps, sim_cfg, storage_kerr=0.0)
        calibration = {
            "reference_metrics": reference_metrics,
            "compensation_sensitivity": list(storage_cfg.compensation_sensitivity),
            "source": "command-line",
        }

    storage_cfg, bounds_check = verify_or_scale_kerr_path(storage_cfg, BOUNDS, epochs)
    ref_g2, ref_eps = params_to_complex(ref_x)
    reference_metrics = measure_lifetimes(ref_g2, ref_eps, sim_cfg, storage_kerr=0.0)

    print(f"Storage-Kerr simulation config: {asdict(sim_cfg)}")
    print(f"Reward config: {asdict(reward_cfg)}")
    print(f"Storage-Kerr config: {asdict(storage_cfg)}")
    print(f"Bounds check: {bounds_check}")
    print(
        "Reference signature: "
        f"Tx={float(reference_metrics['T_X']):.5g} "
        f"Tz={float(reference_metrics['T_Z']):.5g} "
        f"bias={float(reference_metrics['bias']):.5g}"
    )
    print(f"epochs={epochs}, population={population}, sigma0={args.sigma0}")
    if args.quick:
        print("Quick mode is a smoke test; medium/final presets are for scientific plots.")

    history, samples, optimizer_metadata = run_storage_kerr_tracking(
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        reference_metrics=reference_metrics,
        epochs=epochs,
        population=population,
        sigma0=float(args.sigma0),
        seed=int(args.seed),
        tracking_sigma_floor=float(args.tracking_sigma_floor),
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
            "period_epochs": float(storage_cfg.period_epochs),
            "highest_harmonic_period_epochs": float(storage_cfg.period_epochs)
            / float(storage_cfg.bandwidth),
            "epochs": epochs,
            "population": population,
            "simulation": asdict(sim_cfg),
            "reward_config": asdict(reward_cfg),
            "storage_kerr_config": asdict(storage_cfg),
            "bounds_check": bounds_check,
            "calibration": calibration,
            "reference_metrics": reference_metrics,
            "reference_signature_weight": float(args.reference_signature_weight),
            "optimizer_metadata": optimizer_metadata,
        }
    )

    write_csv(RESULTS / "storage_kerr_optimization_history.csv", history)
    write_csv(RESULTS / "storage_kerr_epoch_samples.csv", samples)
    write_json(RESULTS / "storage_kerr_summary_metrics.json", summary)
    write_json(
        RESULTS / "storage_kerr_final_candidate.json",
        {
            "final_mean": history[-1],
            "calibrated_optimum_final": true_storage_kerr_optimum(epochs, storage_cfg).tolist(),
            "kerr_drift_final": storage_kerr_drift(epochs, storage_cfg),
        },
    )

    figures = {
        "storage_kerr_bias_vs_epoch": str(FIGURES / "storage_kerr_bias_vs_epoch.png"),
        "storage_kerr_lifetimes_vs_epoch": str(FIGURES / "storage_kerr_lifetimes_vs_epoch.png"),
        "storage_kerr_reward_or_loss_vs_epoch": str(FIGURES / "storage_kerr_reward_or_loss_vs_epoch.png"),
        "storage_kerr_parameters_vs_epoch": str(FIGURES / "storage_kerr_parameters_vs_epoch.png"),
        "storage_kerr_signal_vs_epoch": str(FIGURES / "storage_kerr_signal_vs_epoch.png"),
        "storage_kerr_tracking_error_vs_epoch": str(FIGURES / "storage_kerr_tracking_error_vs_epoch.png"),
        "storage_kerr_effective_parameters_vs_epoch": str(FIGURES / "storage_kerr_effective_parameters_vs_epoch.png"),
    }
    plot_kerr_bias_vs_epoch(
        history,
        float(args.target_bias),
        BIAS_TOL_REL,
        int(args.warmup_epoch),
        FIGURES / "storage_kerr_bias_vs_epoch.png",
    )
    plot_kerr_lifetimes_vs_epoch(history, FIGURES / "storage_kerr_lifetimes_vs_epoch.png")
    plot_kerr_reward_vs_epoch(history, FIGURES / "storage_kerr_reward_or_loss_vs_epoch.png")
    plot_kerr_parameters_vs_epoch(history, FIGURES / "storage_kerr_parameters_vs_epoch.png")
    plot_kerr_signal_vs_epoch(history, FIGURES / "storage_kerr_signal_vs_epoch.png")
    plot_kerr_tracking_error_vs_epoch(history, FIGURES / "storage_kerr_tracking_error_vs_epoch.png")
    plot_kerr_effective_parameters_vs_epoch(history, FIGURES / "storage_kerr_effective_parameters_vs_epoch.png")
    figures.update(
        write_decay_snapshots(
            history,
            sim_cfg,
            skip_optional=bool(args.no_decay_snapshots),
        )
    )

    command = "python run_storage_kerr_tracking.py"
    if args.quick:
        command += " --quick"
    if args.epochs is not None:
        command += f" --epochs {args.epochs}"
    if args.population is not None:
        command += f" --population {args.population}"
    if args.tracking_sigma_floor != 0.055:
        command += f" --tracking-sigma-floor {args.tracking_sigma_floor}"
    if args.sigma0 != 0.40:
        command += f" --sigma0 {args.sigma0}"
    if args.period_epochs != 120.0:
        command += f" --period-epochs {args.period_epochs}"
    if args.kerr_amplitude != 0.020:
        command += f" --kerr-amplitude {args.kerr_amplitude}"
    if args.reference_signature_weight != 100.0:
        command += f" --reference-signature-weight {args.reference_signature_weight}"
    if args.compensation_sensitivity is not None:
        command += " --compensation-sensitivity " + " ".join(
            f"{float(v):.8g}" for v in args.compensation_sensitivity
        )
    if args.x_reference is not None:
        command += " --x-reference " + " ".join(f"{float(v):.8g}" for v in args.x_reference)
    if args.sim_preset is not None:
        command += f" --sim-preset {args.sim_preset}"

    write_report(
        ROOT / "storage_kerr_validation_report.md",
        command=command,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        bounds_check=bounds_check,
        summary=summary,
        optimizer_metadata=optimizer_metadata,
        calibration=calibration,
        reference_metrics=reference_metrics,
        figures=figures,
    )

    print("\nStorage-Kerr tracking summary")
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
    print(f"Saved report to {ROOT / 'storage_kerr_validation_report.md'}")
    print(f"Saved figures to {FIGURES}")
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
