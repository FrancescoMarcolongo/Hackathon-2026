"""Run a BLT-style online tracker for the analytic storage-detuning benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from cat_model import SimulationConfig, clear_measure_cache, measure_lifetimes, params_to_complex
from plotting import plot_decay_fit, set_style
from rewards import RewardConfig
from run_core_bias_optimization import BIAS_TOL_REL, OPTIMIZATION_START_X, TARGET_BIAS
from run_storage_detuning_tracking import (
    BOUNDS_5D,
    CONTROL_SUFFIXES,
    WARMUP_EPOCH,
    build_reward_config,
    compute_summary_metrics,
    evaluate_x_detuning,
)
from storage_detuning import (
    StorageDetuningConfig,
    storage_detuning_drift,
    true_storage_detuning_optimum,
    verify_storage_detuning_path,
)
from storage_detuning_plotting import (
    plot_detuning_bias_vs_epoch,
    plot_detuning_effective_parameters_vs_epoch,
    plot_detuning_lifetimes_vs_epoch,
    plot_detuning_parameters_vs_epoch,
    plot_detuning_reward_vs_epoch,
    plot_detuning_signal_vs_epoch,
    plot_detuning_tracking_error_vs_epoch,
)
from validation import write_csv, write_json


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

CONTROL_SCALES = np.array([0.28, 0.20, 0.45, 0.35, 0.040], dtype=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--population", type=int, default=33)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--target-bias", type=float, default=TARGET_BIAS)
    parser.add_argument("--warmup-epoch", type=int, default=WARMUP_EPOCH)
    parser.add_argument("--period-epochs", type=float, default=96.0)
    parser.add_argument("--detuning-amplitude", type=float, default=0.08)
    parser.add_argument("--reference-signature-weight", type=float, default=0.0)
    parser.add_argument("--local-detuning-step", type=float, default=0.030)
    parser.add_argument(
        "--candidate-mode",
        choices=("fair_start", "calibrated_tracker"),
        default="fair_start",
        help=(
            "fair_start starts from OPTIMIZATION_START_X and does not inject x_ref; "
            "calibrated_tracker injects stationary x_ref candidates and is a post-calibration tracker."
        ),
    )
    parser.add_argument(
        "--x-reference",
        type=float,
        nargs=4,
        default=None,
        metavar=("G2_RE", "G2_IM", "EPS_RE", "EPS_IM"),
        help="Stationary four-control reference. Defaults to the detuning runner reference.",
    )
    parser.add_argument(
        "--sim-preset",
        choices=("quick", "medium", "final"),
        default=None,
        help="Simulation preset. Defaults to quick with --quick and medium otherwise.",
    )
    parser.add_argument("--no-decay-snapshots", action="store_true")
    return parser.parse_args()


def build_simulation_config(args: argparse.Namespace) -> SimulationConfig:
    preset = args.sim_preset or ("quick" if args.quick else "medium")
    if preset == "quick":
        return SimulationConfig(na=8, nb=3, t_final_x=1.0, t_final_z=90.0, n_points=24)
    if preset == "medium":
        return SimulationConfig(na=12, nb=4, t_final_x=1.2, t_final_z=220.0, n_points=36)
    return SimulationConfig(na=15, nb=5, t_final_x=1.2, t_final_z=260.0, n_points=60)


def load_baseline_storage_reference() -> tuple[float, float, float, float] | None:
    summary_path = RESULTS / "storage_detuning_summary_metrics.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        values = data["storage_detuning_config"]["x_reference"]
        if len(values) != 4:
            return None
        return tuple(float(value) for value in values)
    except Exception:
        return None


def build_storage_config(args: argparse.Namespace) -> tuple[StorageDetuningConfig, str]:
    kwargs = {
        "period_epochs": float(args.period_epochs),
        "amplitude": float(args.detuning_amplitude),
    }
    if args.x_reference is not None:
        kwargs["x_reference"] = tuple(float(v) for v in args.x_reference)
        source = "command_line"
    else:
        baseline_reference = load_baseline_storage_reference()
        if baseline_reference is not None:
            kwargs["x_reference"] = baseline_reference
            source = "results/storage_detuning_summary_metrics.json"
        else:
            source = "StorageDetuningConfig default"
    return StorageDetuningConfig(**kwargs), source


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("dynamiqs", "jax", "cmaes", "scipy", "matplotlib", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return versions


def run_blt_detuning_tracking(
    *,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    storage_cfg: StorageDetuningConfig,
    reference_metrics: dict[str, object],
    epochs: int,
    population: int,
    seed: int,
    reference_signature_weight: float,
    local_detuning_step: float,
    candidate_mode: str,
) -> tuple[list[dict], list[dict], dict]:
    clear_measure_cache()
    rng = np.random.default_rng(int(seed))
    mean = np.array([*OPTIMIZATION_START_X.copy(), storage_detuning_drift(0, storage_cfg)], dtype=float)
    mean = np.minimum(np.maximum(mean, BOUNDS_5D[:, 0]), BOUNDS_5D[:, 1])
    previous_best = mean.copy()
    previous_previous_best = mean.copy()
    x_ref = np.asarray(storage_cfg.x_reference, dtype=float)
    history: list[dict] = []
    samples: list[dict] = []
    start = time.time()

    for epoch in range(0, int(epochs) + 1):
        candidates, roles = _blt_candidates(
            previous_best,
            previous_previous_best,
            x_ref=x_ref,
            bounds=BOUNDS_5D,
            population=int(population),
            rng=rng,
            local_detuning_step=float(local_detuning_step),
            storage_cfg=storage_cfg,
            candidate_mode=candidate_mode,
        )
        evaluated = [
            evaluate_x_detuning(
                candidate,
                epoch=epoch,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                storage_cfg=storage_cfg,
                reference_metrics=reference_metrics,
                reference_signature_weight=reference_signature_weight,
            )
            for candidate in candidates
        ]
        for idx, (role, item) in enumerate(zip(roles, evaluated)):
            samples.append(_sample_row(epoch, idx, role, item))

        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        history.append(
            _history_row(
                epoch,
                mean_eval=epoch_best,
                epoch_best=epoch_best,
                population_stats=_population_stats(evaluated),
                update_type=f"BLT_DETUNING_{candidate_mode.upper()}",
                seed=seed,
            )
        )
        previous_previous_best = previous_best.copy()
        previous_best = np.asarray(epoch_best["x"], dtype=float)

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = history[-1]
            print(
                f"storage-detuning-blt epoch={epoch:03d}/{epochs} "
                f"bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"track_l2={row['tracking_error_l2']:.4g} "
                f"Delta_cmd={row['command_storage_detuning']:.4g} "
                f"Delta_true={row['true_opt_storage_detuning']:.4g} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )

    metadata = {
        "optimizer": "BLT_DETUNING_COORDINATE_SCAN",
        "candidate_mode": candidate_mode,
        "initial_mean": mean.tolist(),
        "population": int(population),
        "seed": int(seed),
        "local_detuning_step": float(local_detuning_step),
        "notes": (
            "fair_start does not inject x_ref or Delta_drift. calibrated_tracker injects "
            "stationary x_ref candidates and is a post-calibration tracker."
        ),
    }
    return history, samples, metadata


def _blt_candidates(
    previous_best: np.ndarray,
    previous_previous_best: np.ndarray,
    *,
    x_ref: np.ndarray,
    bounds: np.ndarray,
    population: int,
    rng: np.random.Generator,
    local_detuning_step: float,
    storage_cfg: StorageDetuningConfig,
    candidate_mode: str,
) -> tuple[list[np.ndarray], list[str]]:
    previous_best = np.asarray(previous_best, dtype=float)
    previous_previous_best = np.asarray(previous_previous_best, dtype=float)
    candidates: list[np.ndarray] = []
    roles: list[str] = []

    def add(x: np.ndarray, role: str) -> None:
        clipped = np.minimum(np.maximum(np.asarray(x, dtype=float), bounds[:, 0]), bounds[:, 1])
        if not any(np.allclose(clipped, existing, atol=1e-10, rtol=0.0) for existing in candidates):
            candidates.append(clipped)
            roles.append(role)

    add(previous_best, "previous_best")
    velocity = previous_best - previous_previous_best
    add(previous_best + 0.85 * velocity, "momentum_prediction")

    if candidate_mode == "calibrated_tracker":
        global_grid = np.linspace(bounds[4, 0], bounds[4, 1], 9)
        for delta_cmd in global_grid:
            add(np.array([*x_ref, delta_cmd], dtype=float), "reference_global_detuning_grid")
    elif candidate_mode != "fair_start":
        raise ValueError(f"Unsupported candidate mode: {candidate_mode}")

    detuning_grid = previous_best[4] + local_detuning_step * np.array([-2.0, -1.0, -0.45, 0.0, 0.45, 1.0, 2.0])
    if candidate_mode == "calibrated_tracker":
        for delta_cmd in detuning_grid:
            add(np.array([*x_ref, delta_cmd], dtype=float), "reference_local_detuning_grid")
    for delta_cmd in detuning_grid:
        add(np.array([*previous_best[:4], delta_cmd], dtype=float), "local_detuning_grid")

    for dim, step in enumerate(CONTROL_SCALES):
        for sign in (-1.0, 1.0):
            trial = previous_best.copy()
            trial[dim] += sign * float(step)
            add(trial, f"coordinate_{dim}_{'plus' if sign > 0 else 'minus'}")

    center = x_ref if candidate_mode == "calibrated_tracker" else previous_best[:4]
    cat_scales = CONTROL_SCALES[:4] * (0.20 if candidate_mode == "calibrated_tracker" else 0.70)
    while len(candidates) < max(population, 4):
        if candidate_mode == "fair_start" and rng.random() < 0.35:
            cat = np.array(
                [
                    rng.uniform(bounds[0, 0], bounds[0, 1]),
                    rng.uniform(bounds[1, 0], bounds[1, 1]),
                    rng.uniform(bounds[2, 0], bounds[2, 1]),
                    rng.uniform(bounds[3, 0], bounds[3, 1]),
                ],
                dtype=float,
            )
        else:
            cat = center + rng.normal(0.0, cat_scales)
        delta_cmd = previous_best[4] + rng.normal(0.0, local_detuning_step)
        add(np.array([*cat, delta_cmd], dtype=float), "random_refinement")
    return candidates[:population], roles[:population]


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
    update_type: str,
    seed: int,
) -> dict:
    row: dict[str, object] = {
        "epoch": int(epoch),
        "update_type": update_type,
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
        "sigma0": np.nan,
        "seed": int(seed),
        "optimizer_generation": int(epoch),
        "optimizer_effective_sigma": np.nan,
        "reheated": 0,
        "drift_storage_detuning": float(mean_eval["storage_detuning_drift"]),
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
    row["storage_detuning_command_error"] = row["tracking_error_storage_detuning"]
    row["storage_detuning_residual_abs"] = abs(float(row["effective_storage_detuning"]))
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
        "drift_storage_detuning": float(item["storage_detuning_drift"]),
    }
    _flatten_vector(row, "command", np.asarray(item["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(item["x_eff"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(item["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(item["tracking_error"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(item["tracking_error"], dtype=float)))
    row["storage_detuning_residual_abs"] = abs(float(row["effective_storage_detuning"]))
    return row


def read_history_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for row in rows:
        converted = {}
        for key, value in row.items():
            if value == "":
                converted[key] = value
                continue
            try:
                if key in ("epoch", "seed", "optimizer_generation", "reheated", "is_feasible", "epoch_best_is_feasible"):
                    converted[key] = int(float(value))
                else:
                    converted[key] = float(value)
            except ValueError:
                converted[key] = value
        out.append(converted)
    return out


def write_decay_snapshot(history: list[dict], sim_cfg: SimulationConfig, path: Path) -> dict[str, str]:
    if not history:
        return {}
    row = history[-1]
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
    result = measure_lifetimes(
        g2,
        eps_d,
        sim_cfg,
        storage_detuning=float(row["effective_storage_detuning"]),
        return_curves=True,
    )
    plot_decay_fit(result, path, "Storage-detuning BLT tracked final decay fits")
    return {"storage_detuning_blt_optimized_decay_fits_final": str(path)}


def plot_comparison(
    baseline: list[dict],
    blt: list[dict],
    *,
    warmup_epoch: int,
    target_bias: float,
    prefix: str,
) -> dict[str, str]:
    figures: dict[str, str] = {}
    set_style()
    datasets = [("SepCMA baseline", baseline, "#165a96"), ("BLT detuning scan", blt, "#c04f15")]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        if rows:
            ax.plot(_series(rows, "epoch"), _series(rows, "bias"), lw=2.0, color=color, label=label)
    ax.axhspan(target_bias * 0.97, target_bias * 1.03, color="#165a96", alpha=0.10, label="success band")
    ax.axhline(target_bias, color="#1f1f1f", ls="--", lw=1.2, label="target")
    ax.axvline(warmup_epoch, color="#5f5f5f", ls=":", lw=1.0, label="warm-up")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Bias $\eta$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_bias_comparison.png"
    _save(fig, path)
    figures["bias_comparison"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        if rows:
            ax.plot(_series(rows, "epoch"), _series(rows, "T_X"), lw=1.8, color=color, ls="-", label=f"{label} T_X")
            ax.plot(_series(rows, "epoch"), _series(rows, "T_Z"), lw=1.8, color=color, ls="--", label=f"{label} T_Z")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lifetime (us)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    path = FIGURES / f"{prefix}_lifetimes_comparison.png"
    _save(fig, path)
    figures["lifetimes_comparison"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    if blt:
        ax.plot(_series(blt, "epoch"), _series(blt, "true_opt_storage_detuning"), color="#1f1f1f", ls="--", lw=2.0, label="true detuning command")
    for label, rows, color in datasets:
        if rows:
            ax.plot(_series(rows, "epoch"), _series(rows, "command_storage_detuning"), lw=2.0, color=color, label=f"{label} command")
    ax.axhline(0.0, color="#5f5f5f", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Detuning command")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_detuning_command_comparison.png"
    _save(fig, path)
    figures["detuning_command_comparison"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        if rows:
            ax.plot(_series(rows, "epoch"), _series(rows, "tracking_error_l2"), lw=2.0, color=color, label=f"{label} tracking L2")
            ax.plot(_series(rows, "epoch"), np.abs(_series(rows, "effective_storage_detuning")), lw=1.4, color=color, ls="--", label=f"{label} |Delta_eff|")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_tracking_error_comparison.png"
    _save(fig, path)
    figures["tracking_error_comparison"] = str(path)
    return figures


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def write_report(
    path: Path,
    *,
    command: str,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    storage_cfg: StorageDetuningConfig,
    bounds_check: dict,
    summary: dict,
    baseline_summary: dict | None,
    optimizer_metadata: dict,
    figures: dict[str, str],
) -> None:
    lines = [
        "# Storage-detuning BLT tracker comparison",
        "",
        "## Reproduction command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Analytic model",
        "",
        "```text",
        "Delta_eff(e) = Delta_cmd(e) - Delta_drift(e)",
        "H_det(e) = Delta_eff(e) a^dag a",
        "x_opt_true(e) = [x_ref, Delta_drift(e)]",
        "```",
        "",
        "Unlike the Kerr four-control case, this benchmark has an exact analytic optimum "
        "for the fifth knob. If the optimizer commands `Delta_cmd = Delta_drift`, the "
        "Hamiltonian sees zero residual detuning and returns to the stationary reference.",
        "",
        f"- candidate mode: `{optimizer_metadata['candidate_mode']}`",
        f"- stationary reference source: `{optimizer_metadata.get('x_reference_source', 'unknown')}`",
        f"- P: `{storage_cfg.period_epochs:g}` epochs",
        f"- bandwidth: `{storage_cfg.bandwidth}`",
        f"- amplitude: `{storage_cfg.amplitude:g}`",
        "",
        "The optimizer does not receive `Delta_drift(e)` or `x_opt_true(e)`. In fair-start "
        "mode it also does not inject the stationary `x_ref` command; in calibrated-tracker "
        "mode it uses `x_ref` as a fixed no-drift calibration prior.",
        "",
        "## Results",
        "",
        f"- final bias: `{summary['final_bias']:.6g}`",
        f"- median bias after warm-up: `{summary['median_bias_after_warmup']:.6g}`",
        "- post-warm-up success-band fraction: "
        f"`{summary['fraction_post_warmup_in_success_band']:.4g}`",
        f"- median tracking L2 after warm-up: `{summary['median_tracking_error_l2_after_warmup']:.6g}`",
        "- median residual detuning after warm-up: "
        f"`{summary['median_storage_detuning_residual_abs_after_warmup']:.6g}`",
        f"- median T_X after warm-up: `{summary['median_T_X_after_warmup']:.6g}` us",
        f"- median T_Z after warm-up: `{summary['median_T_Z_after_warmup']:.6g}` us",
        f"- tracking good: `{summary['tracking_good']}`",
        "",
    ]
    if baseline_summary:
        lines += [
            "SepCMA detuning baseline reference from the current CSV:",
            "",
            f"- baseline final bias: `{baseline_summary.get('final_bias', float('nan')):.6g}`",
            f"- baseline median tracking L2 after warm-up: `{baseline_summary.get('median_tracking_error_l2_after_warmup', float('nan')):.6g}`",
            f"- baseline median residual detuning after warm-up: `{baseline_summary.get('median_storage_detuning_residual_abs_after_warmup', float('nan')):.6g}`",
            "",
        ]
    lines += [
        "## Figures",
        "",
    ]
    for label, fig_path in figures.items():
        lines.append(f"- {label}: `{fig_path}`")
    lines += [
        "",
        "## Run configuration",
        "",
        "```json",
        json.dumps(
            {
                "simulation": asdict(sim_cfg),
                "reward": asdict(reward_cfg),
                "storage_detuning": asdict(storage_cfg),
                "bounds_check": bounds_check,
                "optimizer_metadata": optimizer_metadata,
                "package_versions": package_versions(),
            },
            indent=2,
            default=_json_default,
        ),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    epochs = int(args.epochs if args.epochs is not None else (12 if args.quick else 80))
    sim_cfg = build_simulation_config(args)
    reward_cfg = build_reward_config(float(args.target_bias))
    storage_cfg, x_reference_source = build_storage_config(args)
    bounds_check = verify_storage_detuning_path(storage_cfg, BOUNDS_5D, epochs)
    ref_g2, ref_eps = params_to_complex(np.asarray(storage_cfg.x_reference, dtype=float))
    reference_metrics = measure_lifetimes(ref_g2, ref_eps, sim_cfg, storage_detuning=0.0)

    history, samples, optimizer_metadata = run_blt_detuning_tracking(
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        reference_metrics=reference_metrics,
        epochs=epochs,
        population=int(args.population),
        seed=int(args.seed),
        reference_signature_weight=float(args.reference_signature_weight),
        local_detuning_step=float(args.local_detuning_step),
        candidate_mode=str(args.candidate_mode),
    )
    optimizer_metadata["x_reference_source"] = x_reference_source
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
            "highest_harmonic_period_epochs": float(storage_cfg.period_epochs) / float(storage_cfg.bandwidth),
            "epochs": epochs,
            "population": int(args.population),
            "simulation": asdict(sim_cfg),
            "reward_config": asdict(reward_cfg),
            "storage_detuning_config": asdict(storage_cfg),
            "bounds_check": bounds_check,
            "reference_metrics": reference_metrics,
            "reference_signature_weight": float(args.reference_signature_weight),
            "optimizer_metadata": optimizer_metadata,
            "x_reference_source": x_reference_source,
        }
    )

    prefix = "storage_detuning_blt"
    if args.candidate_mode == "calibrated_tracker":
        prefix = "storage_detuning_blt_calibrated"
    write_csv(RESULTS / f"{prefix}_optimization_history.csv", history)
    write_csv(RESULTS / f"{prefix}_epoch_samples.csv", samples)
    write_json(RESULTS / f"{prefix}_summary_metrics.json", summary)
    write_json(
        RESULTS / f"{prefix}_final_candidate.json",
        {
            "final_mean": history[-1],
            "true_optimum_final": true_storage_detuning_optimum(epochs, storage_cfg).tolist(),
            "detuning_drift_final": storage_detuning_drift(epochs, storage_cfg),
        },
    )

    figures = {
        f"{prefix}_bias_vs_epoch": str(FIGURES / f"{prefix}_bias_vs_epoch.png"),
        f"{prefix}_lifetimes_vs_epoch": str(FIGURES / f"{prefix}_lifetimes_vs_epoch.png"),
        f"{prefix}_reward_or_loss_vs_epoch": str(FIGURES / f"{prefix}_reward_or_loss_vs_epoch.png"),
        f"{prefix}_parameters_vs_epoch": str(FIGURES / f"{prefix}_parameters_vs_epoch.png"),
        f"{prefix}_signal_vs_epoch": str(FIGURES / f"{prefix}_signal_vs_epoch.png"),
        f"{prefix}_tracking_error_vs_epoch": str(FIGURES / f"{prefix}_tracking_error_vs_epoch.png"),
        f"{prefix}_effective_parameters_vs_epoch": str(FIGURES / f"{prefix}_effective_parameters_vs_epoch.png"),
    }
    plot_detuning_bias_vs_epoch(history, float(args.target_bias), BIAS_TOL_REL, int(args.warmup_epoch), FIGURES / f"{prefix}_bias_vs_epoch.png")
    plot_detuning_lifetimes_vs_epoch(history, FIGURES / f"{prefix}_lifetimes_vs_epoch.png")
    plot_detuning_reward_vs_epoch(history, FIGURES / f"{prefix}_reward_or_loss_vs_epoch.png")
    plot_detuning_parameters_vs_epoch(history, FIGURES / f"{prefix}_parameters_vs_epoch.png")
    plot_detuning_signal_vs_epoch(history, FIGURES / f"{prefix}_signal_vs_epoch.png")
    plot_detuning_tracking_error_vs_epoch(history, FIGURES / f"{prefix}_tracking_error_vs_epoch.png")
    plot_detuning_effective_parameters_vs_epoch(history, FIGURES / f"{prefix}_effective_parameters_vs_epoch.png")
    if not args.no_decay_snapshots:
        figures.update(write_decay_snapshot(history, sim_cfg, FIGURES / f"{prefix}_optimized_decay_fits_final.png"))

    baseline_rows = read_history_csv(RESULTS / "storage_detuning_optimization_history.csv")
    baseline_summary = None
    if baseline_rows:
        baseline_summary = compute_summary_metrics(
            baseline_rows,
            warmup_epoch=int(args.warmup_epoch),
            target_bias=float(args.target_bias),
            bias_tol_rel=BIAS_TOL_REL,
        )
    figures.update(
        plot_comparison(
            baseline_rows,
            history,
            warmup_epoch=int(args.warmup_epoch),
            target_bias=float(args.target_bias),
            prefix=f"{prefix}_vs_sepcma",
        )
    )

    command = "python run_storage_detuning_blt_tracking.py"
    if args.quick:
        command += " --quick"
    if args.epochs is not None:
        command += f" --epochs {args.epochs}"
    if args.population != 33:
        command += f" --population {args.population}"
    if args.sim_preset is not None:
        command += f" --sim-preset {args.sim_preset}"
    if args.period_epochs != 96.0:
        command += f" --period-epochs {args.period_epochs}"
    if args.detuning_amplitude != 0.08:
        command += f" --detuning-amplitude {args.detuning_amplitude}"
    if args.warmup_epoch != WARMUP_EPOCH:
        command += f" --warmup-epoch {args.warmup_epoch}"
    if args.reference_signature_weight != 0.0:
        command += f" --reference-signature-weight {args.reference_signature_weight}"
    if args.local_detuning_step != 0.030:
        command += f" --local-detuning-step {args.local_detuning_step}"
    if args.candidate_mode != "fair_start":
        command += f" --candidate-mode {args.candidate_mode}"
    if args.no_decay_snapshots:
        command += " --no-decay-snapshots"

    write_report(
        ROOT / f"{prefix}_comparison_report.md",
        command=command,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        bounds_check=bounds_check,
        summary=summary,
        baseline_summary=baseline_summary,
        optimizer_metadata=optimizer_metadata,
        figures=figures,
    )

    print("\nStorage-detuning BLT tracking summary")
    print(f"final bias={summary['final_bias']:.6g}")
    print(f"median post-warmup bias={summary['median_bias_after_warmup']:.6g}")
    print(f"post-warmup success-band fraction={summary['fraction_post_warmup_in_success_band']:.4g}")
    print(f"median post-warmup tracking L2={summary['median_tracking_error_l2_after_warmup']:.6g}")
    print(f"median post-warmup detuning residual={summary['median_storage_detuning_residual_abs_after_warmup']:.6g}")
    print(f"Saved report to {ROOT / f'{prefix}_comparison_report.md'}")


if __name__ == "__main__":
    main()
