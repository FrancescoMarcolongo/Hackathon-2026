"""Compare old fitted-lifetime reward with no-fit proxy rewards.

The proxy objectives avoid exponential fits. They convert short-time logical
contrast measurements into effective Tx/Tz proxies, then use the same editable
reward function as the flexible experiment. Each epoch's best candidate is also
validated with the expensive fitted-lifetime metric for an apples-to-apples
readout.
"""

from __future__ import annotations

import csv
import json
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

OBJECTIVES = [
    {
        "name": "old_fit_reward",
        "kind": "fit",
        "population_size": 4,
        "n_generations": 10,
        "sigma0": 0.25,
        "seed": 2026,
    },
    {
        "name": "proxy_single_endpoint_cem",
        "kind": "proxy",
        "proxy_mode": "single_endpoint",
        "population_size": 8,
        "n_generations": 10,
        "sigma0": 0.45,
        "seed": 2026,
    },
    {
        "name": "proxy_two_sided_syndrome_cem",
        "kind": "proxy",
        "proxy_mode": "two_sided_flip_syndrome",
        "population_size": 8,
        "n_generations": 10,
        "sigma0": 0.45,
        "seed": 2027,
    },
]


def evaluate_objective(x: np.ndarray, objective: dict[str, Any], config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    cfg = dict(config)
    if objective["kind"] == "fit":
        return cat_lifetime_loss(jnp.asarray(x), config=cfg, return_metrics=True)

    cfg["proxy_mode"] = objective["proxy_mode"]
    return cat_proxy_loss(jnp.asarray(x), config=cfg, return_metrics=True)


def validate_with_fit(x: np.ndarray, config: dict[str, Any]) -> dict[str, float]:
    t0 = time.perf_counter()
    loss, metrics = cat_lifetime_loss(jnp.asarray(x), config=config, return_metrics=True)
    elapsed = time.perf_counter() - t0
    if "error" in metrics:
        return {
            "validated_loss": float(loss),
            "validated_reward": np.nan,
            "validated_Tx": np.nan,
            "validated_Tz": np.nan,
            "validated_eta": np.nan,
            "validation_seconds": float(elapsed),
        }
    return {
        "validated_loss": float(loss),
        "validated_reward": float(metrics["reward"]),
        "validated_Tx": float(metrics["Tx"]),
        "validated_Tz": float(metrics["Tz"]),
        "validated_eta": float(metrics["bias"]),
        "validation_seconds": float(elapsed),
    }


def run_cem(objective: dict[str, Any], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rng = np.random.default_rng(int(objective["seed"]))
    mean = X0.copy()
    sigma = float(objective["sigma0"])
    population_size = int(objective["population_size"])
    n_generations = int(objective["n_generations"])
    elite_count = max(2, population_size // 2)
    best: dict[str, Any] | None = None
    candidate_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    for epoch in range(n_generations):
        candidates = mean + rng.normal(0.0, sigma, size=(population_size, 4))
        candidates[0] = mean
        candidates = np.clip(candidates, LOWER_BOUNDS, UPPER_BOUNDS)
        evaluated = []

        for candidate_index, x_candidate in enumerate(candidates):
            t0 = time.perf_counter()
            loss, metrics = evaluate_objective(x_candidate, objective, config)
            elapsed = time.perf_counter() - t0
            row = {
                "objective": objective["name"],
                "epoch": epoch,
                "candidate_index": candidate_index,
                "loss": float(loss),
                "reward": float(metrics.get("reward", np.nan)),
                "Tx": float(metrics.get("Tx", np.nan)),
                "Tz": float(metrics.get("Tz", np.nan)),
                "eta": float(metrics.get("bias", metrics.get("eta", np.nan))),
                "x_contrast": float(metrics.get("x_contrast", np.nan)),
                "z_contrast": float(metrics.get("z_contrast", np.nan)),
                "alpha": float(metrics.get("alpha", np.nan)),
                "nbar": float(metrics.get("nbar", np.nan)),
                "eval_seconds": float(elapsed),
                "x0": float(x_candidate[0]),
                "x1": float(x_candidate[1]),
                "x2": float(x_candidate[2]),
                "x3": float(x_candidate[3]),
            }
            candidate_rows.append(row)
            evaluated.append(row)
            if best is None or row["loss"] < best["loss"]:
                best = dict(row)

        evaluated.sort(key=lambda row: row["loss"])
        elite_xs = [
            np.array([row["x0"], row["x1"], row["x2"], row["x3"]], dtype=float)
            for row in evaluated[:elite_count]
        ]
        elite_mean = np.mean(elite_xs, axis=0)
        mean = np.clip(0.55 * mean + 0.45 * elite_mean, LOWER_BOUNDS, UPPER_BOUNDS)
        sigma = max(0.08, sigma * 0.82)

        assert best is not None
        best_x = np.array([best["x0"], best["x1"], best["x2"], best["x3"]], dtype=float)
        validation = validate_with_fit(best_x, config)
        epoch_best = evaluated[0]
        epoch_rows.append(
            {
                "objective": objective["name"],
                "epoch": epoch,
                "epoch_best_loss": epoch_best["loss"],
                "epoch_best_reward": epoch_best["reward"],
                "epoch_best_Tx": epoch_best["Tx"],
                "epoch_best_Tz": epoch_best["Tz"],
                "epoch_best_eta": epoch_best["eta"],
                "best_so_far_loss": best["loss"],
                "best_so_far_reward": best["reward"],
                "best_so_far_Tx": best["Tx"],
                "best_so_far_Tz": best["Tz"],
                "best_so_far_eta": best["eta"],
                **validation,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        best.update(validation)
        print(
            f"{objective['name']} epoch={epoch} "
            f"loss={best['loss']:.4g} eta={best['eta']:.4g} "
            f"validated_eta={validation['validated_eta']:.4g} "
            f"validated_loss={validation['validated_loss']:.4g}",
            flush=True,
        )

    assert best is not None
    return candidate_rows, epoch_rows, best


def summarize_timing(
    objective: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
) -> dict[str, float | int | str]:
    objective_candidates = [
        row for row in candidate_rows if row["objective"] == objective["name"]
    ]
    objective_epochs = [row for row in epoch_rows if row["objective"] == objective["name"]]
    eval_seconds = [float(row["eval_seconds"]) for row in objective_candidates]
    validation_seconds = [float(row["validation_seconds"]) for row in objective_epochs]
    objective_eval_total = float(np.sum(eval_seconds))
    validation_total = float(np.sum(validation_seconds))
    n_evals = len(eval_seconds)
    return {
        "objective": objective["name"],
        "kind": objective["kind"],
        "population_size": int(objective["population_size"]),
        "n_generations": int(objective["n_generations"]),
        "n_objective_evaluations": int(n_evals),
        "mean_objective_eval_seconds": float(np.mean(eval_seconds)) if eval_seconds else np.nan,
        "median_objective_eval_seconds": float(np.median(eval_seconds)) if eval_seconds else np.nan,
        "total_objective_eval_seconds": objective_eval_total,
        "mean_validation_seconds": float(np.mean(validation_seconds)) if validation_seconds else np.nan,
        "total_validation_seconds": validation_total,
        "total_measured_seconds_with_validation": objective_eval_total + validation_total,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(epoch_rows: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    specs = [
        ("loss_vs_epoch.png", "best_so_far_loss", "Optimization loss", False),
        ("eta_vs_epoch.png", "best_so_far_eta", "eta / proxy eta", True),
        ("Tx_vs_epoch.png", "best_so_far_Tx", "Tx / proxy Tx", True),
        ("Tz_vs_epoch.png", "best_so_far_Tz", "Tz / proxy Tz", True),
        ("validated_loss_vs_epoch.png", "validated_loss", "Validated fitted loss", False),
        ("validated_eta_vs_epoch.png", "validated_eta", "Validated eta", True),
        ("validated_Tx_vs_epoch.png", "validated_Tx", "Validated Tx", True),
        ("validated_Tz_vs_epoch.png", "validated_Tz", "Validated Tz", True),
    ]
    objectives = [obj["name"] for obj in OBJECTIVES]
    paths: list[Path] = []

    for filename, key, ylabel, logy in specs:
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        for objective in objectives:
            rows = [row for row in epoch_rows if row["objective"] == objective]
            if not rows:
                continue
            ax.plot(
                [int(row["epoch"]) for row in rows],
                [float(row[key]) for row in rows],
                marker="o",
                lw=1.7,
                label=objective,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = run_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def main() -> None:
    run_dir = create_run_dir(ROOT / "reward_function_experiments" / "results", label="no_fit_proxy_search")
    config = default_reward_config()
    config.update(
        {
            "proxy_tfinal_x": 0.6,
            "proxy_tfinal_z": 30.0,
            "proxy_n_points": 2,
            "lambda_syndrome_floor": 0.15,
            "use_alpha_penalty": True,
            "alpha_target": 1.8,
            "lambda_alpha": 0.05,
            "use_nbar_penalty": True,
            "nbar_max": 7.5,
            "lambda_nbar": 0.05,
            "use_parity_bonus": False,
            "objectives": OBJECTIVES,
        }
    )
    save_run_config(config, run_dir)

    all_candidates: list[dict[str, Any]] = []
    all_epochs: list[dict[str, Any]] = []
    best_by_objective: dict[str, Any] = {}
    timing_by_objective: dict[str, Any] = {}
    for objective in OBJECTIVES:
        candidates, epochs, best = run_cem(objective, config)
        all_candidates.extend(candidates)
        all_epochs.extend(epochs)
        best_by_objective[objective["name"]] = best
        timing_by_objective[objective["name"]] = summarize_timing(objective, candidates, epochs)

    write_csv(run_dir / "candidate_history.csv", all_candidates)
    write_csv(run_dir / "epoch_history.csv", all_epochs)
    plot_paths = save_plots(all_epochs, run_dir)
    best_validated_by_objective = {}
    for objective in OBJECTIVES:
        rows = [row for row in all_epochs if row["objective"] == objective["name"]]
        if rows:
            best_validated_by_objective[objective["name"]] = min(
                rows,
                key=lambda row: float(row["validated_loss"]),
            )

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "best_by_objective": best_by_objective,
        "best_validated_epoch_by_objective": best_validated_by_objective,
        "timing_by_objective": timing_by_objective,
        "plots": [str(path) for path in plot_paths],
    }
    save_summary(summary, run_dir)
    print(f"Saved no-fit proxy comparison to: {run_dir}")


if __name__ == "__main__":
    main()
