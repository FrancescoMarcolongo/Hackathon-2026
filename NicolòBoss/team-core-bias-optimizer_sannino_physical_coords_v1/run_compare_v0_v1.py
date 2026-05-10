"""Compare Version 0 raw coordinates with Version 1 physical coordinates.

This script intentionally keeps the estimator, reward, and SepCMA update rule
the same in both modes.  The only experimental variable is the coordinate
system seen by the optimizer.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from cmaes import SepCMA

from cat_model import SimulationConfig, measure_lifetimes, params_to_complex
from physical_coordinates import (
    physical_bounds_from_raw_reference,
    physical_diagnostics_from_raw,
    physical_to_raw,
    raw_to_physical,
)
from rewards import RewardConfig, compute_reward
from run_core_bias_optimization import BOUNDS as RAW_BOUNDS
from run_core_bias_optimization import select_better

PROJECT_VERSION = "physical-coordinate-baseline-v1"
U0 = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
POPULATION_SIZE = 4
N_GENERATIONS = 3
SIGMA0_RAW = 0.25
SIGMA0_PHYSICAL = 0.35
RANDOM_SEED = 2026


def create_run_dir(base_dir: str | Path = "results", label: str | None = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [f"run_{timestamp}"]
    if label:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_.-")
        if safe:
            parts.append(safe[:60])
    run_dir = base / "_".join(parts)
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{'_'.join(parts)}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=N_GENERATIONS)
    parser.add_argument("--population", type=int, default=POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--quick", action="store_true", help="Use fewer fit points for a smoke run.")
    return parser.parse_args()


def package_config(args: argparse.Namespace, sim_cfg: SimulationConfig, physical_bounds: dict[str, list[float]]) -> dict[str, Any]:
    return {
        "project_version": PROJECT_VERSION,
        "coordinate_modes": ["raw", "physical"],
        "random_seed": int(args.seed),
        "population_size": int(args.population),
        "n_generations": int(args.generations),
        "sigma0_raw": SIGMA0_RAW,
        "sigma0_physical": SIGMA0_PHYSICAL,
        "u0": U0.tolist(),
        "coordinate_mode": "physical",
        "kappa_b": float(sim_cfg.kappa_b),
        "physical_bounds": physical_bounds,
        "simulation_config": vars(sim_cfg),
        "reward_config": vars(RewardConfig()),
    }


def evaluate_raw_u(
    u: np.ndarray,
    *,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    coordinate_mode: str,
    optimizer_coordinate: np.ndarray,
    generation: int,
    candidate_index: int,
    evaluation: int,
) -> dict[str, Any]:
    u = np.asarray(u, dtype=float)
    g2, eps_d = params_to_complex(u)
    metrics = measure_lifetimes(g2, eps_d, sim_cfg)
    reward = compute_reward(metrics, reward_cfg)

    if coordinate_mode == "physical":
        v = np.asarray(optimizer_coordinate, dtype=float)
    else:
        v = raw_to_physical(u, kappa_b=sim_cfg.kappa_b)

    diagnostics = physical_diagnostics_from_raw(u, kappa_b=sim_cfg.kappa_b)
    row: dict[str, Any] = {
        "evaluation": int(evaluation),
        "generation": int(generation),
        "candidate_index": int(candidate_index),
        "coordinate_mode": coordinate_mode,
        "loss": float(reward["loss_to_minimize"]),
        "reward": float(reward["reward"]),
        "Tx": float(metrics.get("T_X", np.nan)),
        "Tz": float(metrics.get("T_Z", np.nan)),
        "bias": float(metrics.get("bias", np.nan)),
        "geo_lifetime": float(metrics.get("geo_lifetime", np.nan)),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_error": float(reward.get("bias_error", np.nan)),
        "v0_log_kappa2": float(v[0]),
        "v1_log_abs_alpha": float(v[1]),
        "v2_arg_alpha": float(v[2]),
        "v3_phi_mismatch": float(v[3]),
        "u0_re_g2": float(u[0]),
        "u1_im_g2": float(u[1]),
        "u2_re_eps_d": float(u[2]),
        "u3_im_eps_d": float(u[3]),
    }
    row.update(diagnostics)
    return row


def row_to_optimizer_coord(row: dict[str, Any], coordinate_mode: str) -> np.ndarray:
    if coordinate_mode == "physical":
        return np.array(
            [
                row["v0_log_kappa2"],
                row["v1_log_abs_alpha"],
                row["v2_arg_alpha"],
                row["v3_phi_mismatch"],
            ],
            dtype=float,
        )
    return np.array(
        [row["u0_re_g2"], row["u1_im_g2"], row["u2_re_eps_d"], row["u3_im_eps_d"]],
        dtype=float,
    )


def row_to_raw(row: dict[str, Any]) -> np.ndarray:
    return np.array(
        [row["u0_re_g2"], row["u1_im_g2"], row["u2_re_eps_d"], row["u3_im_eps_d"]],
        dtype=float,
    )


def run_mode(
    coordinate_mode: str,
    *,
    args: argparse.Namespace,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    bounds_v: np.ndarray,
) -> list[dict[str, Any]]:
    if coordinate_mode == "raw":
        mean0 = U0.copy()
        bounds = RAW_BOUNDS
        sigma0 = SIGMA0_RAW
    elif coordinate_mode == "physical":
        mean0 = raw_to_physical(U0, kappa_b=sim_cfg.kappa_b)
        bounds = bounds_v
        sigma0 = SIGMA0_PHYSICAL
    else:
        raise ValueError("coordinate_mode must be 'raw' or 'physical'")

    optimizer = SepCMA(
        mean=mean0.copy(),
        sigma=float(sigma0),
        bounds=np.asarray(bounds, dtype=float),
        population_size=int(args.population),
        seed=int(args.seed),
    )

    rows: list[dict[str, Any]] = []
    incumbent: dict[str, Any] | None = None
    evaluation = 0

    for generation in range(int(args.generations)):
        candidates = [np.asarray(optimizer.ask(), dtype=float) for _ in range(optimizer.population_size)]
        if len(candidates) >= 1:
            candidates[0] = row_to_optimizer_coord(incumbent, coordinate_mode) if incumbent else mean0.copy()
        if len(candidates) >= 2:
            candidates[1] = mean0.copy()

        evaluated: list[dict[str, Any]] = []
        for candidate_index, coord in enumerate(candidates):
            if coordinate_mode == "physical":
                raw_u = physical_to_raw(coord, kappa_b=sim_cfg.kappa_b)
            else:
                raw_u = np.asarray(coord, dtype=float)
            row = evaluate_raw_u(
                raw_u,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                coordinate_mode=coordinate_mode,
                optimizer_coordinate=coord,
                generation=generation,
                candidate_index=candidate_index,
                evaluation=evaluation,
            )
            evaluation += 1
            rows.append(row)
            evaluated.append(row)
            if incumbent is None or select_better(row, incumbent, reward_cfg):
                incumbent = dict(row)

        optimizer.tell(
            [
                (row_to_optimizer_coord(row, coordinate_mode), float(row["loss"]))
                for row in evaluated
            ]
        )

        best = min(rows, key=lambda item: float(item["loss"]))
        print(
            f"{coordinate_mode} generation={generation} "
            f"best_loss={best['loss']:.4g} reward={best['reward']:.4g} "
            f"Tx={best['Tx']:.4g} Tz={best['Tz']:.4g} bias={best['bias']:.4g}",
            flush=True,
        )

    return rows


def cumulative_best(rows: list[dict[str, Any]], key: str = "loss") -> list[dict[str, Any]]:
    best: dict[str, Any] | None = None
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["evaluation"])):
        if best is None or float(row[key]) < float(best[key]):
            best = row
        out.append(best)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_by_mode(rows: list[dict[str, Any]], run_dir: Path, filename: str, columns: list[str], ylabel: str, best: bool = True) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for mode in ("raw", "physical"):
        mode_rows = [row for row in rows if row["coordinate_mode"] == mode]
        plot_rows = cumulative_best(mode_rows) if best else sorted(mode_rows, key=lambda item: int(item["evaluation"]))
        x = [int(row["evaluation"]) for row in plot_rows]
        for col in columns:
            ax.plot(x, [float(row[col]) for row in plot_rows], marker="o", lw=1.5, label=f"{mode} {col}")
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_plots(rows: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    paths = [
        _plot_by_mode(rows, run_dir, "best_loss_vs_evaluation_v0_v1.png", ["loss"], "Best loss", best=True),
        _plot_by_mode(rows, run_dir, "best_reward_vs_evaluation_v0_v1.png", ["reward"], "Best reward", best=True),
        _plot_by_mode(rows, run_dir, "Tx_Tz_vs_evaluation_v0_v1.png", ["Tx", "Tz"], "Best-candidate lifetimes", best=True),
        _plot_by_mode(rows, run_dir, "bias_vs_evaluation_v0_v1.png", ["bias"], "Best-candidate bias", best=True),
        _plot_by_mode(rows, run_dir, "raw_controls_vs_evaluation_v0_v1.png", ["u0_re_g2", "u1_im_g2", "u2_re_eps_d", "u3_im_eps_d"], "Raw controls", best=False),
    ]

    physical_rows = [row for row in rows if row["coordinate_mode"] == "physical"]
    paths.append(
        _plot_by_mode(
            physical_rows,
            run_dir,
            "physical_controls_vs_evaluation_v1.png",
            ["v0_log_kappa2", "v1_log_abs_alpha", "v2_arg_alpha", "v3_phi_mismatch"],
            "Physical controls",
            best=False,
        )
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if physical_rows:
        sc = ax.scatter(
            [row["kappa_2"] for row in physical_rows],
            [row["alpha_abs"] for row in physical_rows],
            c=[row["loss"] for row in physical_rows],
            cmap="viridis_r",
            s=45,
        )
        fig.colorbar(sc, ax=ax, label="loss")
    ax.set_xlabel("kappa_2")
    ax.set_ylabel("|alpha|")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = run_dir / "physical_samples_alpha_kappa2.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    sim_cfg = SimulationConfig(n_points=35) if args.quick else SimulationConfig()
    reward_cfg = RewardConfig()
    lower_v, upper_v, physical_bounds = physical_bounds_from_raw_reference(U0, kappa_b=sim_cfg.kappa_b)
    bounds_v = np.column_stack([lower_v, upper_v])

    run_dir = create_run_dir(ROOT / "results", label="compare_v0_v1")
    config = package_config(args, sim_cfg, physical_bounds)
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(config), f, indent=2, sort_keys=True)

    all_rows: list[dict[str, Any]] = []
    for mode in ("raw", "physical"):
        all_rows.extend(
            run_mode(
                mode,
                args=args,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                bounds_v=bounds_v,
            )
        )

    write_csv(run_dir / "candidate_history.csv", all_rows)
    plot_paths = save_plots(all_rows, run_dir)

    raw_rows = [row for row in all_rows if row["coordinate_mode"] == "raw"]
    physical_rows = [row for row in all_rows if row["coordinate_mode"] == "physical"]
    best_raw = min(raw_rows, key=lambda row: float(row["loss"]))
    best_physical = min(physical_rows, key=lambda row: float(row["loss"]))
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_version": PROJECT_VERSION,
        "baseline_source": "copied from current project",
        "coordinate_modes_compared": ["raw", "physical"],
        "random_seed": int(args.seed),
        "population_size": int(args.population),
        "n_generations": int(args.generations),
        "best_raw_mode_result": best_raw,
        "best_physical_mode_result": best_physical,
        "best_loss_raw": float(best_raw["loss"]),
        "best_loss_physical": float(best_physical["loss"]),
        "improvement_best_loss": float(best_raw["loss"] - best_physical["loss"]),
        "evaluations_to_threshold_raw": None,
        "evaluations_to_threshold_physical": None,
        "notes": [
            "Version 1 changes only the optimizer coordinate system.",
            "Evaluator, lifetime extraction, reward/loss, and SepCMA update are shared.",
        ],
        "plots": [str(path) for path in plot_paths],
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, sort_keys=True)

    print(f"Saved comparison to: {run_dir}")


if __name__ == "__main__":
    main()

