"""Sweep no-fit proxy reward hyperparameters and validate winners.

This keeps optimization cheap by using ``cat_proxy_loss`` during search, then
validates the best proxy candidate from each hyperparameter setting with the
full fitted-lifetime objective.
"""

from __future__ import annotations

import csv
import itertools
import os
import sys
import time
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

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from reward_function_experiments.flexible_cat_reward import (
    cat_lifetime_loss,
    cat_proxy_loss,
    create_run_dir,
    default_reward_config,
    save_run_config,
    save_summary,
)

X0 = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
LOWER_BOUNDS = np.array([0.1, -1.0, 0.1, -2.0], dtype=float)
UPPER_BOUNDS = np.array([3.0, 1.0, 8.0, 2.0], dtype=float)

POPULATION_SIZE = 6
N_GENERATIONS = 5
SIGMA0 = 0.45
SEED = 2026

SWEEP_GRID = {
    "lambda_bias": [2.0, 5.0, 10.0],
    "lambda_syndrome_floor": [0.0, 0.10, 0.25],
    "lambda_alpha": [0.0, 0.05],
}


def make_sweep_configs(base_config: dict[str, Any]) -> list[dict[str, Any]]:
    keys = list(SWEEP_GRID)
    configs = []
    for values in itertools.product(*(SWEEP_GRID[key] for key in keys)):
        config = dict(base_config)
        config.update(dict(zip(keys, values)))
        config["use_alpha_penalty"] = bool(config["lambda_alpha"] > 0)
        label = "_".join(f"{key}={config[key]}" for key in keys)
        configs.append({"label": label, "config": config})
    return configs


def evaluate_proxy(x: np.ndarray, config: dict[str, Any]) -> tuple[float, dict[str, Any], float]:
    start = time.perf_counter()
    loss, metrics = cat_proxy_loss(jnp.asarray(x), config=config, return_metrics=True)
    return float(loss), dict(metrics), float(time.perf_counter() - start)


def validate_with_fit(x: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    start = time.perf_counter()
    loss, metrics = cat_lifetime_loss(jnp.asarray(x), config=config, return_metrics=True)
    elapsed = float(time.perf_counter() - start)
    if "error" in metrics:
        return {
            "validated_loss": float(loss),
            "validated_reward": np.nan,
            "validated_Tx": np.nan,
            "validated_Tz": np.nan,
            "validated_eta": np.nan,
            "validation_seconds": elapsed,
        }
    return {
        "validated_loss": float(loss),
        "validated_reward": float(metrics["reward"]),
        "validated_Tx": float(metrics["Tx"]),
        "validated_Tz": float(metrics["Tz"]),
        "validated_eta": float(metrics["bias"]),
        "validation_seconds": elapsed,
    }


def run_one_config(
    sweep_index: int,
    label: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(SEED + sweep_index)
    mean = X0.copy()
    sigma = float(SIGMA0)
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(N_GENERATIONS):
        candidates = mean + rng.normal(0.0, sigma, size=(POPULATION_SIZE, 4))
        candidates[0] = mean
        candidates = np.clip(candidates, LOWER_BOUNDS, UPPER_BOUNDS)
        evaluated = []

        for candidate_index, x_candidate in enumerate(candidates):
            loss, metrics, eval_seconds = evaluate_proxy(x_candidate, config)
            row = {
                "sweep_index": sweep_index,
                "label": label,
                "epoch": epoch,
                "candidate_index": candidate_index,
                "loss": loss,
                "reward": float(metrics.get("reward", np.nan)),
                "proxy_Tx": float(metrics.get("Tx", np.nan)),
                "proxy_Tz": float(metrics.get("Tz", np.nan)),
                "proxy_eta": float(metrics.get("bias", np.nan)),
                "x_contrast": float(metrics.get("x_contrast", np.nan)),
                "z_contrast": float(metrics.get("z_contrast", np.nan)),
                "alpha": float(metrics.get("alpha", np.nan)),
                "nbar": float(metrics.get("nbar", np.nan)),
                "eval_seconds": eval_seconds,
                "lambda_bias": float(config["lambda_bias"]),
                "lambda_syndrome_floor": float(config["lambda_syndrome_floor"]),
                "lambda_alpha": float(config["lambda_alpha"]),
                "x0": float(x_candidate[0]),
                "x1": float(x_candidate[1]),
                "x2": float(x_candidate[2]),
                "x3": float(x_candidate[3]),
            }
            rows.append(row)
            evaluated.append(row)
            if best is None or row["loss"] < best["loss"]:
                best = dict(row)

        evaluated.sort(key=lambda row: row["loss"])
        elite_count = max(2, POPULATION_SIZE // 2)
        elite_xs = [
            np.array([row["x0"], row["x1"], row["x2"], row["x3"]], dtype=float)
            for row in evaluated[:elite_count]
        ]
        mean = np.clip(0.55 * mean + 0.45 * np.mean(elite_xs, axis=0), LOWER_BOUNDS, UPPER_BOUNDS)
        sigma = max(0.08, sigma * 0.82)

    assert best is not None
    best_x = np.array([best["x0"], best["x1"], best["x2"], best["x3"]], dtype=float)
    validation = validate_with_fit(best_x, config)
    summary = {
        **best,
        **validation,
        "total_proxy_eval_seconds": float(sum(row["eval_seconds"] for row in rows)),
        "wall_seconds": float(time.perf_counter() - started),
        "n_proxy_evaluations": len(rows),
    }
    print(
        f"[{sweep_index:02d}] {label} "
        f"proxy_loss={best['loss']:.4g} proxy_eta={best['proxy_eta']:.4g} "
        f"validated_loss={summary['validated_loss']:.4g} "
        f"validated_eta={summary['validated_eta']:.4g}",
        flush=True,
    )
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_sweep_plots(summary_rows: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    paths: list[Path] = []

    ordered = sorted(summary_rows, key=lambda row: float(row["validated_loss"]))
    labels = [str(row["label"]) for row in ordered]
    x = np.arange(len(ordered))

    specs = [
        ("validated_loss_by_config.png", "validated_loss", "Validated fitted loss", False),
        ("validated_eta_by_config.png", "validated_eta", "Validated eta", True),
        ("proxy_eta_by_config.png", "proxy_eta", "Best proxy eta", True),
        ("mean_eval_seconds_by_config.png", "mean_proxy_eval_seconds", "Mean proxy eval seconds", False),
    ]
    for filename, key, ylabel, logy in specs:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.bar(x, [float(row[key]) for row in ordered], color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = run_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def main() -> None:
    run_dir = create_run_dir(ROOT / "reward_function_experiments" / "results", label="reward_hyperparameter_sweep")
    base_config = default_reward_config()
    base_config.update(
        {
            "proxy_mode": "single_endpoint",
            "proxy_tfinal_x": 0.6,
            "proxy_tfinal_z": 30.0,
            "proxy_n_points": 2,
            "use_nbar_penalty": True,
            "nbar_max": 7.5,
            "lambda_nbar": 0.05,
            "use_parity_bonus": False,
            "population_size": POPULATION_SIZE,
            "n_generations": N_GENERATIONS,
            "sigma0": SIGMA0,
            "seed": SEED,
            "sweep_grid": SWEEP_GRID,
        }
    )
    save_run_config(base_config, run_dir)

    all_candidates: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for sweep_index, item in enumerate(make_sweep_configs(base_config)):
        rows, summary = run_one_config(sweep_index, item["label"], item["config"])
        all_candidates.extend(rows)
        summary["mean_proxy_eval_seconds"] = summary["total_proxy_eval_seconds"] / max(1, summary["n_proxy_evaluations"])
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: float(row["validated_loss"]))
    write_csv(run_dir / "candidate_history.csv", all_candidates)
    write_csv(run_dir / "sweep_summary.csv", summary_rows)
    plot_paths = save_sweep_plots(summary_rows, run_dir)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "best_config": summary_rows[0],
        "top_5_configs": summary_rows[:5],
        "n_configs": len(summary_rows),
        "plots": [str(path) for path in plot_paths],
    }
    save_summary(summary, run_dir)
    print(f"Saved reward hyperparameter sweep to: {run_dir}")


if __name__ == "__main__":
    main()

