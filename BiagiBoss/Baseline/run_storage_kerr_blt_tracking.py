"""Run a stronger online Kerr-compensation tracker and compare it with SepCMA."""

from __future__ import annotations

import argparse
import csv
import json
import math
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
from run_storage_kerr_compensation_tracking import (
    BOUNDS_5D,
    CAT_REFERENCE_SCALES,
    CONTROL_SUFFIXES,
    WARMUP_EPOCH,
    build_reward_config,
    compute_summary_metrics,
    evaluate_x_kerr5,
)
from storage_kerr_compensation import (
    StorageKerrCompensationConfig,
    kerr_compensation_drift,
    true_kerr_compensation_optimum,
    verify_kerr_compensation_path,
)
from storage_kerr_compensation_plotting import (
    plot_kerr5_bias_vs_epoch,
    plot_kerr5_effective_parameters_vs_epoch,
    plot_kerr5_lifetimes_vs_epoch,
    plot_kerr5_parameters_vs_epoch,
    plot_kerr5_reward_vs_epoch,
    plot_kerr5_signal_vs_epoch,
    plot_kerr5_tracking_error_vs_epoch,
)
from validation import write_csv, write_json


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--population", type=int, default=33)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-bias", type=float, default=TARGET_BIAS)
    parser.add_argument("--warmup-epoch", type=int, default=35)
    parser.add_argument("--period-epochs", type=float, default=100.0)
    parser.add_argument("--kerr-amplitude", type=float, default=0.30)
    parser.add_argument("--reference-signature-weight", type=float, default=0.0)
    parser.add_argument("--cat-reference-weight", type=float, default=0.0)
    parser.add_argument("--local-k-step", type=float, default=0.055)
    parser.add_argument(
        "--candidate-mode",
        choices=("fair_start", "prior_assisted"),
        default="fair_start",
        help=(
            "fair_start uses only current/past measured candidates and random/local probes; "
            "prior_assisted also injects stationary x_ref candidates."
        ),
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


def build_storage_config(args: argparse.Namespace) -> StorageKerrCompensationConfig:
    return StorageKerrCompensationConfig(
        period_epochs=float(args.period_epochs),
        amplitude=float(args.kerr_amplitude),
    )


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("dynamiqs", "jax", "cmaes", "scipy", "matplotlib", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return versions


def run_blt_kerr_tracking(
    *,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    storage_cfg: StorageKerrCompensationConfig,
    reference_metrics: dict[str, object],
    epochs: int,
    population: int,
    seed: int,
    reference_signature_weight: float,
    cat_reference_weight: float,
    local_k_step: float,
    candidate_mode: str,
) -> tuple[list[dict], list[dict], dict]:
    """Direct online Kerr tracker.

    The candidate generator never receives K_drift(epoch) or x_opt_true(epoch).
    It searches K_cmd using the previous command, a momentum prediction, and
    fixed epoch-independent compensation-grid probes.
    """

    clear_measure_cache()
    rng = np.random.default_rng(int(seed))
    x_ref = np.asarray(storage_cfg.x_reference, dtype=float)
    mean = np.array([*OPTIMIZATION_START_X.copy(), 0.0], dtype=float)
    previous_best = mean.copy()
    previous_previous_best = mean.copy()
    history: list[dict] = []
    samples: list[dict] = []
    start = time.time()

    for epoch in range(0, int(epochs) + 1):
        candidates, roles = _blt_candidates(
            mean,
            previous_best,
            previous_previous_best,
            x_ref=x_ref,
            bounds=BOUNDS_5D,
            population=int(population),
            rng=rng,
            local_k_step=float(local_k_step),
            storage_cfg=storage_cfg,
            candidate_mode=candidate_mode,
        )
        evaluated = [
            evaluate_x_kerr5(
                candidate,
                epoch=epoch,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                storage_cfg=storage_cfg,
                reference_metrics=reference_metrics,
                reference_signature_weight=reference_signature_weight,
                cat_reference_weight=cat_reference_weight,
            )
            for candidate in candidates
        ]
        for idx, (role, item) in enumerate(zip(roles, evaluated)):
            samples.append(_sample_row(epoch, idx, role, item))

        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        mean_eval = epoch_best
        history.append(
            _history_row(
                epoch,
                mean_eval=mean_eval,
                epoch_best=epoch_best,
                population_stats=_population_stats(evaluated),
                update_type="BLT_DIRECT_K_SCAN",
                seed=seed,
            )
        )
        previous_previous_best = previous_best.copy()
        previous_best = np.asarray(epoch_best["x"], dtype=float)
        mean = previous_best.copy()

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = history[-1]
            print(
                f"storage-kerr5-blt epoch={epoch:03d}/{epochs} "
                f"bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"track_l2={row['tracking_error_l2']:.4g} "
                f"K_cmd={row['command_storage_kerr']:.4g} "
                f"K_true={row['true_opt_storage_kerr']:.4g} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )

    metadata = {
        "optimizer": "BLT_DIRECT_K_SCAN",
        "initial_mean": [*OPTIMIZATION_START_X.tolist(), 0.0],
        "population": int(population),
        "seed": int(seed),
        "local_k_step": float(local_k_step),
        "candidate_mode": candidate_mode,
        "notes": (
            "Uses measured reward probes plus local momentum. In fair_start mode it does not "
            "inject x_ref candidates and does not use K_drift(epoch) or x_opt_true(epoch) "
            "for candidate generation."
        ),
    }
    return history, samples, metadata


def _blt_candidates(
    mean: np.ndarray,
    previous_best: np.ndarray,
    previous_previous_best: np.ndarray,
    *,
    x_ref: np.ndarray,
    bounds: np.ndarray,
    population: int,
    rng: np.random.Generator,
    local_k_step: float,
    storage_cfg: StorageKerrCompensationConfig,
    candidate_mode: str,
) -> tuple[list[np.ndarray], list[str]]:
    mean = np.asarray(mean, dtype=float)
    previous_best = np.asarray(previous_best, dtype=float)
    previous_previous_best = np.asarray(previous_previous_best, dtype=float)
    candidates: list[np.ndarray] = []
    roles: list[str] = []

    def add(x: np.ndarray, role: str) -> None:
        clipped = np.minimum(np.maximum(np.asarray(x, dtype=float), bounds[:, 0]), bounds[:, 1])
        if not any(np.allclose(clipped, existing, atol=1e-10, rtol=0.0) for existing in candidates):
            candidates.append(clipped)
            roles.append(role)

    add(mean, "current_best")
    add(previous_best, "previous_best")
    velocity = previous_best - previous_previous_best
    add(previous_best + 0.85 * velocity, "momentum_prediction")

    local_grid = previous_best[4] + local_k_step * np.array([-2.0, -1.0, -0.45, 0.0, 0.45, 1.0, 2.0])
    for k in local_grid:
        add(np.array([*previous_best[:4], k], dtype=float), "local_k_grid")

    coordinate_steps = np.array([0.22, 0.16, 0.35, 0.28, local_k_step], dtype=float)
    for dim, step in enumerate(coordinate_steps):
        for sign in (-1.0, 1.0):
            trial = previous_best.copy()
            trial[dim] += sign * float(step)
            add(trial, f"coordinate_{dim}_{'plus' if sign > 0 else 'minus'}")

    if candidate_mode == "prior_assisted":
        k_min = -1.15 * float(storage_cfg.amplitude)
        k_max = 0.25 * float(storage_cfg.amplitude)
        global_grid = np.linspace(max(bounds[4, 0], k_min), min(bounds[4, 1], k_max), 7)
        for k in global_grid:
            add(np.array([*x_ref, k], dtype=float), "stationary_cat_global_k_grid")
        for k in local_grid:
            add(np.array([*x_ref, k], dtype=float), "reference_cat_local_k_grid")
    elif candidate_mode != "fair_start":
        raise ValueError(f"Unsupported candidate mode: {candidate_mode}")

    cat_center = x_ref if candidate_mode == "prior_assisted" else previous_best[:4]
    cat_scales = CAT_REFERENCE_SCALES * (0.18 if candidate_mode == "prior_assisted" else 0.55)
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
            cat = cat_center + rng.normal(0.0, cat_scales)
        k = previous_best[4] + rng.normal(0.0, local_k_step)
        add(np.array([*cat, k], dtype=float), "local_random_refinement")
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
        "cat_reference_penalty": float(mean_eval["cat_reference_penalty"]),
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
    row["storage_kerr_command_error"] = row["tracking_error_storage_kerr"]
    row["storage_kerr_residual_abs"] = abs(float(row["effective_storage_kerr"]))
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
        "cat_reference_penalty": float(item["cat_reference_penalty"]),
        "drift_storage_kerr": float(item["storage_kerr_drift"]),
    }
    _flatten_vector(row, "command", np.asarray(item["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(item["x_eff"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(item["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(item["tracking_error"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(item["tracking_error"], dtype=float)))
    row["storage_kerr_residual_abs"] = abs(float(row["effective_storage_kerr"]))
    return row


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
        storage_kerr=float(row["effective_storage_kerr"]),
        return_curves=True,
    )
    plot_decay_fit(result, path, "Storage-Kerr BLT tracked final decay fits")
    return {"storage_kerr5_blt_optimized_decay_fits_final": str(path)}


def write_sign_consistency(
    storage_cfg: StorageKerrCompensationConfig,
    sim_cfg: SimulationConfig,
    path: Path,
) -> dict:
    x_ref = np.asarray(storage_cfg.x_reference, dtype=float)
    g2, eps_d = params_to_complex(x_ref)
    reference = measure_lifetimes(g2, eps_d, sim_cfg, storage_kerr=0.0)
    rows = []
    for epoch in (0, 10, 20, 30, 45, 60):
        k_drift = kerr_compensation_drift(epoch, storage_cfg)
        for label, k_cmd in (
            ("no_command", 0.0),
            ("correct_cancel_command", -k_drift),
            ("wrong_flipped_command", k_drift),
        ):
            k_eff = k_drift + k_cmd
            metrics = measure_lifetimes(g2, eps_d, sim_cfg, storage_kerr=k_eff)
            rows.append(
                {
                    "epoch": int(epoch),
                    "case": label,
                    "k_drift": float(k_drift),
                    "k_cmd": float(k_cmd),
                    "k_eff": float(k_eff),
                    "T_X": float(metrics["T_X"]),
                    "T_Z": float(metrics["T_Z"]),
                    "bias": float(metrics["bias"]),
                    "bias_error_vs_reference": float(metrics["bias"] - reference["bias"]),
                }
            )
    payload = {"reference": reference, "rows": rows}
    write_json(path, payload)
    return payload


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
    datasets = [("SepCMA baseline", baseline, "#165a96"), ("BLT direct scan", blt, "#c04f15")]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        if rows:
            epochs = _series(rows, "epoch")
            ax.plot(epochs, _series(rows, "bias"), lw=2.0, color=color, label=label)
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
            epochs = _series(rows, "epoch")
            ax.plot(epochs, _series(rows, "T_X"), lw=1.8, color=color, ls="-", label=f"{label} T_X")
            ax.plot(epochs, _series(rows, "T_Z"), lw=1.8, color=color, ls="--", label=f"{label} T_Z")
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
        epochs = _series(blt, "epoch")
        ax.plot(epochs, _series(blt, "true_opt_storage_kerr"), color="#1f1f1f", ls="--", lw=2.0, label="true K command")
    for label, rows, color in datasets:
        if rows:
            epochs = _series(rows, "epoch")
            ax.plot(epochs, _series(rows, "command_storage_kerr"), lw=2.0, color=color, label=f"{label} command")
    ax.axhline(0.0, color="#5f5f5f", lw=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Kerr command")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_kerr_command_comparison.png"
    _save(fig, path)
    figures["kerr_command_comparison"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        if rows:
            epochs = _series(rows, "epoch")
            ax.plot(epochs, _series(rows, "tracking_error_l2"), lw=2.0, color=color, label=f"{label} tracking L2")
            ax.plot(epochs, np.abs(_series(rows, "effective_storage_kerr")), lw=1.4, color=color, ls="--", label=f"{label} |K_eff|")
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
    storage_cfg: StorageKerrCompensationConfig,
    bounds_check: dict,
    summary: dict,
    baseline_summary: dict | None,
    optimizer_metadata: dict,
    sign_check: dict,
    figures: dict[str, str],
) -> None:
    correct_rows = [row for row in sign_check["rows"] if row["case"] == "correct_cancel_command"]
    max_correct_bias_error = max(abs(float(row["bias_error_vs_reference"])) for row in correct_rows)
    lines = [
        "# Storage-Kerr BLT tracker comparison",
        "",
        "## Reproduction command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Model validity checks",
        "",
        "The explicit Kerr knob is additive in the Hamiltonian:",
        "",
        "```text",
        "K_eff(e) = K_drift(e) + K_cmd(e)",
        "H_K(e) = -0.5 K_eff(e) n_a (n_a - I)",
        "x_opt_true(e) = [x_ref, -K_drift(e)]",
        "```",
        "",
        "This is an analytic compensation optimum for the five-knob model because setting "
        "`K_cmd = -K_drift` makes `K_eff = 0` and returns the Hamiltonian to the stationary "
        "reference. It is not a proof that the four original cat knobs have a unique global "
        "optimum under Kerr drift.",
        "",
        f"- maximum bias deviation for the sign-correct cancellation check: `{max_correct_bias_error:.6g}`",
        f"- P: `{storage_cfg.period_epochs:g}` epochs",
        f"- bandwidth: `{storage_cfg.bandwidth}`",
        f"- highest harmonic period: `{storage_cfg.period_epochs / storage_cfg.bandwidth:g}` epochs",
        f"- Kerr amplitude: `{storage_cfg.amplitude:g}`",
        "",
        "The BLT-style tracker does not receive `K_drift(e)` or `x_opt_true(e)` when choosing "
        "candidates. In `fair_start` mode it also does not inject the stationary `x_ref` command; "
        "it probes measured rewards with current/past commands, local coordinate moves, local "
        "`K_cmd` grids, and random exploration. In `prior_assisted` mode it additionally injects "
        "stationary `x_ref` candidates and should be interpreted as a post-calibration tracker.",
        "",
        "## Results",
        "",
        f"- final bias: `{summary['final_bias']:.6g}`",
        f"- median bias after warm-up: `{summary['median_bias_after_warmup']:.6g}`",
        "- post-warm-up success-band fraction: "
        f"`{summary['fraction_post_warmup_in_success_band']:.4g}`",
        f"- median tracking L2 after warm-up: `{summary['median_tracking_error_l2_after_warmup']:.6g}`",
        "- median residual Kerr after warm-up: "
        f"`{summary['median_storage_kerr_residual_abs_after_warmup']:.6g}`",
        f"- median T_X after warm-up: `{summary['median_T_X_after_warmup']:.6g}` us",
        f"- median T_Z after warm-up: `{summary['median_T_Z_after_warmup']:.6g}` us",
        f"- tracking good: `{summary['tracking_good']}`",
        "",
    ]
    if baseline_summary:
        lines += [
            "SepCMA baseline reference from the current Kerr-compensation CSV:",
            "",
            f"- baseline final bias: `{baseline_summary.get('final_bias', float('nan')):.6g}`",
            f"- baseline median tracking L2 after warm-up: `{baseline_summary.get('median_tracking_error_l2_after_warmup', float('nan')):.6g}`",
            f"- baseline median residual Kerr after warm-up: `{baseline_summary.get('median_storage_kerr_residual_abs_after_warmup', float('nan')):.6g}`",
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
        "## Sign-consistency table",
        "",
        "```json",
        json.dumps(sign_check, indent=2, default=_json_default),
        "```",
        "",
        "## Run configuration",
        "",
        "```json",
        json.dumps(
            {
                "simulation": asdict(sim_cfg),
                "reward": asdict(reward_cfg),
                "storage_kerr_compensation": asdict(storage_cfg),
                "bounds_check": bounds_check,
                "optimizer_metadata": optimizer_metadata,
                "candidate_mode": optimizer_metadata.get("candidate_mode", ""),
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

    epochs = int(args.epochs if args.epochs is not None else (12 if args.quick else 60))
    sim_cfg = build_simulation_config(args)
    reward_cfg = build_reward_config(float(args.target_bias))
    storage_cfg = build_storage_config(args)
    bounds_check = verify_kerr_compensation_path(storage_cfg, BOUNDS_5D, epochs)
    ref_g2, ref_eps = params_to_complex(np.asarray(storage_cfg.x_reference, dtype=float))
    reference_metrics = measure_lifetimes(ref_g2, ref_eps, sim_cfg, storage_kerr=0.0)

    sign_check = write_sign_consistency(
        storage_cfg,
        sim_cfg,
        RESULTS / "storage_kerr5_sign_consistency.json",
    )

    history, samples, optimizer_metadata = run_blt_kerr_tracking(
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        reference_metrics=reference_metrics,
        epochs=epochs,
        population=int(args.population),
        seed=int(args.seed),
        reference_signature_weight=float(args.reference_signature_weight),
        cat_reference_weight=float(args.cat_reference_weight),
        local_k_step=float(args.local_k_step),
        candidate_mode=str(args.candidate_mode),
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
            "highest_harmonic_period_epochs": float(storage_cfg.period_epochs) / float(storage_cfg.bandwidth),
            "epochs": epochs,
            "population": int(args.population),
            "simulation": asdict(sim_cfg),
            "reward_config": asdict(reward_cfg),
            "storage_kerr_compensation_config": asdict(storage_cfg),
            "bounds_check": bounds_check,
            "reference_metrics": reference_metrics,
            "reference_signature_weight": float(args.reference_signature_weight),
            "cat_reference_weight": float(args.cat_reference_weight),
            "optimizer_metadata": optimizer_metadata,
        }
    )

    write_csv(RESULTS / "storage_kerr5_blt_optimization_history.csv", history)
    write_csv(RESULTS / "storage_kerr5_blt_epoch_samples.csv", samples)
    write_json(RESULTS / "storage_kerr5_blt_summary_metrics.json", summary)
    write_json(
        RESULTS / "storage_kerr5_blt_final_candidate.json",
        {
            "final_mean": history[-1],
            "true_optimum_final": true_kerr_compensation_optimum(epochs, storage_cfg).tolist(),
            "kerr_drift_final": kerr_compensation_drift(epochs, storage_cfg),
        },
    )

    figures = {
        "storage_kerr5_blt_bias_vs_epoch": str(FIGURES / "storage_kerr5_blt_bias_vs_epoch.png"),
        "storage_kerr5_blt_lifetimes_vs_epoch": str(FIGURES / "storage_kerr5_blt_lifetimes_vs_epoch.png"),
        "storage_kerr5_blt_reward_or_loss_vs_epoch": str(FIGURES / "storage_kerr5_blt_reward_or_loss_vs_epoch.png"),
        "storage_kerr5_blt_parameters_vs_epoch": str(FIGURES / "storage_kerr5_blt_parameters_vs_epoch.png"),
        "storage_kerr5_blt_signal_vs_epoch": str(FIGURES / "storage_kerr5_blt_signal_vs_epoch.png"),
        "storage_kerr5_blt_tracking_error_vs_epoch": str(FIGURES / "storage_kerr5_blt_tracking_error_vs_epoch.png"),
        "storage_kerr5_blt_effective_parameters_vs_epoch": str(FIGURES / "storage_kerr5_blt_effective_parameters_vs_epoch.png"),
    }
    plot_kerr5_bias_vs_epoch(history, float(args.target_bias), BIAS_TOL_REL, int(args.warmup_epoch), FIGURES / "storage_kerr5_blt_bias_vs_epoch.png")
    plot_kerr5_lifetimes_vs_epoch(history, FIGURES / "storage_kerr5_blt_lifetimes_vs_epoch.png")
    plot_kerr5_reward_vs_epoch(history, FIGURES / "storage_kerr5_blt_reward_or_loss_vs_epoch.png")
    plot_kerr5_parameters_vs_epoch(history, FIGURES / "storage_kerr5_blt_parameters_vs_epoch.png")
    plot_kerr5_signal_vs_epoch(history, FIGURES / "storage_kerr5_blt_signal_vs_epoch.png")
    plot_kerr5_tracking_error_vs_epoch(history, FIGURES / "storage_kerr5_blt_tracking_error_vs_epoch.png")
    plot_kerr5_effective_parameters_vs_epoch(history, FIGURES / "storage_kerr5_blt_effective_parameters_vs_epoch.png")
    if not args.no_decay_snapshots:
        figures.update(write_decay_snapshot(history, sim_cfg, FIGURES / "storage_kerr5_blt_optimized_decay_fits_final.png"))

    baseline_rows = read_history_csv(RESULTS / "storage_kerr5_optimization_history.csv")
    baseline_summary = None
    if baseline_rows:
        baseline_summary = compute_summary_metrics(
            baseline_rows,
            warmup_epoch=int(args.warmup_epoch),
            target_bias=float(args.target_bias),
            bias_tol_rel=BIAS_TOL_REL,
        )
    figures.update(plot_comparison(baseline_rows, history, warmup_epoch=int(args.warmup_epoch), target_bias=float(args.target_bias), prefix="storage_kerr5_blt_vs_sepcma"))

    command = "python run_storage_kerr_blt_tracking.py"
    if args.quick:
        command += " --quick"
    if args.epochs is not None:
        command += f" --epochs {args.epochs}"
    if args.population != 13:
        command += f" --population {args.population}"
    if args.sim_preset is not None:
        command += f" --sim-preset {args.sim_preset}"
    if args.period_epochs != 100.0:
        command += f" --period-epochs {args.period_epochs}"
    if args.kerr_amplitude != 0.30:
        command += f" --kerr-amplitude {args.kerr_amplitude}"
    if args.warmup_epoch != 35:
        command += f" --warmup-epoch {args.warmup_epoch}"
    if args.cat_reference_weight != 4.0:
        command += f" --cat-reference-weight {args.cat_reference_weight}"
    if args.reference_signature_weight != 120.0:
        command += f" --reference-signature-weight {args.reference_signature_weight}"
    if args.no_decay_snapshots:
        command += " --no-decay-snapshots"
    if args.candidate_mode != "fair_start":
        command += f" --candidate-mode {args.candidate_mode}"
    if args.local_k_step != 0.055:
        command += f" --local-k-step {args.local_k_step}"

    write_report(
        ROOT / "storage_kerr_blt_comparison_report.md",
        command=command,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        storage_cfg=storage_cfg,
        bounds_check=bounds_check,
        summary=summary,
        baseline_summary=baseline_summary,
        optimizer_metadata=optimizer_metadata,
        sign_check=sign_check,
        figures=figures,
    )

    print("\nStorage-Kerr BLT tracking summary")
    print(f"final bias={summary['final_bias']:.6g}")
    print(f"median post-warmup bias={summary['median_bias_after_warmup']:.6g}")
    print(f"post-warmup success-band fraction={summary['fraction_post_warmup_in_success_band']:.4g}")
    print(f"median post-warmup tracking L2={summary['median_tracking_error_l2_after_warmup']:.6g}")
    print(f"median post-warmup Kerr residual={summary['median_storage_kerr_residual_abs_after_warmup']:.6g}")
    print(f"Saved report to {ROOT / 'storage_kerr_blt_comparison_report.md'}")


if __name__ == "__main__":
    main()
