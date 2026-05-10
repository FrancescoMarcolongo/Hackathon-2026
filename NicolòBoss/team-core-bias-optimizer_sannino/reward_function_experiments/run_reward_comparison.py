"""Run flexible reward vs. old simple reward comparison.

The old reward is the previous simple lifetime formula:

    log(Tx) + log(Tz) - lambda_bias * log((Tz / Tx) / eta_target)**2

The flexible reward uses ``cat_reward_from_metrics`` with optional diagnostic
terms enabled in the config below. All artifacts are saved in a timestamped
run directory.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BASELINE_ROOT = ROOT.parent / "reward_function_baseline"
if BASELINE_ROOT.exists() and str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from reward_function_experiments.flexible_cat_reward import (
    cat_lifetime_loss,
    cat_reward_from_metrics,
    create_run_dir,
    default_reward_config,
    save_run_config,
    save_summary,
)

try:
    from simple_lifetime_reward import cat_reward_from_lifetimes as old_cat_reward_from_lifetimes
except Exception:
    old_cat_reward_from_lifetimes = None

POPULATION_SIZE = 4
N_GENERATIONS = 3
SIGMA0 = 0.25
SEED = 2026

X0 = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
LOWER_BOUNDS = np.array([0.1, -1.0, 0.1, -2.0], dtype=float)
UPPER_BOUNDS = np.array([3.0, 1.0, 8.0, 2.0], dtype=float)


def old_reward_from_metrics(metrics: dict[str, Any], config: dict[str, Any]) -> float:
    eps = 1.0e-12
    Tx = max(eps, float(metrics["Tx"]))
    Tz = max(eps, float(metrics["Tz"]))
    eta = max(eps, float(metrics.get("bias", Tz / Tx)))
    eta_target = max(eps, float(config["eta_target"]))
    if old_cat_reward_from_lifetimes is not None:
        return float(
            old_cat_reward_from_lifetimes(
                Tx,
                Tz,
                eta_target=eta_target,
                lambda_bias=float(config["lambda_bias"]),
            )
        )
    return float(
        math.log(Tx)
        + math.log(Tz)
        - float(config["lambda_bias"]) * math.log(eta / eta_target) ** 2
    )


def evaluate_x(x: np.ndarray, config: dict[str, Any], objective: str) -> tuple[float, dict[str, Any]]:
    loss, metrics = cat_lifetime_loss(jnp.asarray(x), config=config, return_metrics=True)
    metrics = dict(metrics)
    if "error" in metrics:
        return float(loss), metrics

    if objective == "old_simple_reward":
        reward = old_reward_from_metrics(metrics, config)
        metrics["reward"] = reward
        metrics["loss"] = -reward
    elif objective == "flexible_reward":
        reward = cat_reward_from_metrics(metrics, config)
        metrics["reward"] = reward
        metrics["loss"] = -reward
    else:
        raise ValueError(f"unknown objective: {objective}")

    metrics["eta"] = float(metrics["bias"])
    return float(metrics["loss"]), metrics


def run_optimizer(objective: str, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    mean = X0.copy()
    sigma = float(SIGMA0)
    candidate_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    for generation in range(N_GENERATIONS):
        candidates = mean + rng.normal(0.0, sigma, size=(POPULATION_SIZE, 4))
        candidates[0] = mean
        candidates = np.clip(candidates, LOWER_BOUNDS, UPPER_BOUNDS)
        generation_results = []

        for candidate_index, x_candidate in enumerate(candidates):
            loss, metrics = evaluate_x(x_candidate, config, objective)
            row = {
                "objective": objective,
                "generation": generation,
                "candidate_index": candidate_index,
                "loss": float(loss),
                "reward": float(metrics.get("reward", np.nan)),
                "eta": float(metrics.get("eta", metrics.get("bias", np.nan))),
                "Tx": float(metrics.get("Tx", np.nan)),
                "Tz": float(metrics.get("Tz", np.nan)),
                "bias": float(metrics.get("bias", np.nan)),
                "alpha": float(metrics.get("alpha", np.nan)),
                "nbar": float(metrics.get("nbar", np.nan)),
                "parity_plus_z": float(metrics.get("parity_plus_z", np.nan)),
                "parity_minus_z": float(metrics.get("parity_minus_z", np.nan)),
                "parity_contrast": float(metrics.get("parity_contrast", np.nan)),
                "x0": float(x_candidate[0]),
                "x1": float(x_candidate[1]),
                "x2": float(x_candidate[2]),
                "x3": float(x_candidate[3]),
            }
            candidate_rows.append(row)
            generation_results.append(row)
            if best is None or row["loss"] < best["loss"]:
                best = dict(row)

        generation_results.sort(key=lambda row: row["loss"])
        elite_count = max(1, POPULATION_SIZE // 2)
        elite_mean = np.mean(
            [
                np.array([row["x0"], row["x1"], row["x2"], row["x3"]], dtype=float)
                for row in generation_results[:elite_count]
            ],
            axis=0,
        )
        mean = np.clip(0.7 * mean + 0.3 * elite_mean, LOWER_BOUNDS, UPPER_BOUNDS)
        sigma *= 0.85

        epoch_best = generation_results[0]
        assert best is not None
        epoch_rows.append(
            {
                "objective": objective,
                "epoch": generation,
                "epoch_best_loss": epoch_best["loss"],
                "epoch_best_reward": epoch_best["reward"],
                "epoch_best_eta": epoch_best["eta"],
                "epoch_best_Tx": epoch_best["Tx"],
                "epoch_best_Tz": epoch_best["Tz"],
                "best_so_far_loss": best["loss"],
                "best_so_far_reward": best["reward"],
                "best_so_far_eta": best["eta"],
                "best_so_far_Tx": best["Tx"],
                "best_so_far_Tz": best["Tz"],
            }
        )
        print(
            f"{objective} epoch={generation} "
            f"best_loss={best['loss']:.6g} eta={best['eta']:.6g} "
            f"Tx={best['Tx']:.6g} Tz={best['Tz']:.6g}",
            flush=True,
        )

    assert best is not None
    return candidate_rows, epoch_rows, best


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plots(epoch_rows: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    paths: list[Path] = []
    metrics = [
        ("loss_vs_epoch.png", "best_so_far_loss", "Loss"),
        ("eta_vs_epoch.png", "best_so_far_eta", "eta = Tz / Tx"),
        ("Tx_vs_epoch.png", "best_so_far_Tx", "Tx"),
        ("Tz_vs_epoch.png", "best_so_far_Tz", "Tz"),
    ]
    objectives = sorted({row["objective"] for row in epoch_rows})

    for filename, key, ylabel in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        for objective in objectives:
            rows = [row for row in epoch_rows if row["objective"] == objective]
            xs = [int(row["epoch"]) for row in rows]
            ys = [float(row[key]) for row in rows]
            ax.plot(xs, ys, marker="o", lw=1.8, label=objective)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if key in {"best_so_far_eta", "best_so_far_Tx", "best_so_far_Tz"}:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = run_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    run_dir = create_run_dir(ROOT / "reward_function_experiments" / "results", label="reward_comparison")
    config = default_reward_config()
    config.update(
        {
            "use_alpha_penalty": True,
            "alpha_target": 2.0,
            "lambda_alpha": 0.1,
            "use_nbar_penalty": True,
            "nbar_max": 8.0,
            "lambda_nbar": 0.1,
            "use_parity_bonus": True,
            "lambda_parity": 0.1,
            "optimizer_population_size": POPULATION_SIZE,
            "optimizer_n_generations": N_GENERATIONS,
            "optimizer_sigma0": SIGMA0,
            "optimizer_seed": SEED,
        }
    )
    save_run_config(config, run_dir)

    all_candidate_rows: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []
    best_by_objective: dict[str, Any] = {}
    for objective in ("old_simple_reward", "flexible_reward"):
        candidate_rows, epoch_rows, best = run_optimizer(objective, config)
        all_candidate_rows.extend(candidate_rows)
        all_epoch_rows.extend(epoch_rows)
        best_by_objective[objective] = best

    write_csv(run_dir / "candidate_history_comparison.csv", all_candidate_rows)
    write_csv(run_dir / "epoch_history_comparison.csv", all_epoch_rows)
    plot_paths = save_comparison_plots(all_epoch_rows, run_dir)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "population_size": POPULATION_SIZE,
        "n_generations": N_GENERATIONS,
        "sigma0": SIGMA0,
        "seed": SEED,
        "best_by_objective": best_by_objective,
        "plots": [str(path) for path in plot_paths],
    }
    save_summary(summary, run_dir)
    print(f"Saved comparison results to: {run_dir}")


if __name__ == "__main__":
    main()
