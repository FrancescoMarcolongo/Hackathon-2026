"""
Plot results from simple_mesolve_cat_optimization.py.

Run:
    python plot_simple_mesolve_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


PLOT_CONFIG = {
    "results_dir": "simple_mesolve_results/",
    "output_dir": "simple_mesolve_results/plots/",
    "show": True,
    "save": True,
}


def to_float(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_history(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = list(csv.DictReader(f))
    if not rows:
        raise ValueError("history.csv is empty")
    return {key: np.asarray([to_float(row[key]) for row in rows], dtype=float) for key in rows[0]}


def finish(fig, path: Path, cfg: Dict[str, object]) -> None:
    fig.tight_layout()
    if cfg["save"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200)
        print(f"saved {path}")
    if cfg["show"]:
        plt.show()
    else:
        plt.close(fig)


def plot_bias(hist: Dict[str, np.ndarray], config: Dict[str, object], out: Path, cfg: Dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["epoch"], hist["bias_mean"], label="bias_mean")
    ax.plot(hist["epoch"], hist["bias_best"], label="bias_best", alpha=0.8)
    ax.axhline(float(config["bias_target"]), color="black", linestyle="--", label="bias_target")
    ax.axhline(hist["baseline_bias"][0], color="gray", linestyle=":", label="baseline_bias")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("T_Z / T_X")
    ax.set_title("Validation bias")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, out / "bias_validation.png", cfg)


def plot_lifetimes(hist: Dict[str, np.ndarray], out: Path, cfg: Dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["epoch"], hist["T_X_mean"], label="T_X_mean")
    ax.plot(hist["epoch"], hist["T_Z_mean"], label="T_Z_mean")
    ax.plot(hist["epoch"], hist["T_X_best"], "--", label="T_X_best", alpha=0.7)
    ax.plot(hist["epoch"], hist["T_Z_best"], "--", label="T_Z_best", alpha=0.7)
    ax.axhline(hist["baseline_T_X"][0], color="C0", linestyle=":", label="baseline_T_X")
    ax.axhline(hist["baseline_T_Z"][0], color="C1", linestyle=":", label="baseline_T_Z")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fitted lifetime")
    ax.set_title("Validation lifetimes")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, out / "lifetimes_validation.png", cfg)


def plot_reward(hist: Dict[str, np.ndarray], out: Path, cfg: Dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["epoch"], hist["reward_best"], label="reward_best")
    ax.plot(hist["epoch"], hist["reward_mean"], label="reward_mean")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Reward")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, out / "reward.png", cfg)


def plot_params(hist: Dict[str, np.ndarray], out: Path, cfg: Dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["epoch"], hist["mean_g2_re"], label="Re(g2)")
    ax.plot(hist["epoch"], hist["mean_g2_im"], label="Im(g2)")
    ax.plot(hist["epoch"], hist["mean_eps_d_re"], label="Re(eps_d)")
    ax.plot(hist["epoch"], hist["mean_eps_d_im"], label="Im(eps_d)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean parameter value")
    ax.set_title("Optimizer mean parameters")
    ax.grid(True, alpha=0.3)
    ax.legend()
    finish(fig, out / "mean_parameters.png", cfg)


def main() -> None:
    cfg = dict(PLOT_CONFIG)
    results_dir = Path(str(cfg["results_dir"]))
    out = Path(str(cfg["output_dir"]))
    hist = load_history(results_dir / "history.csv")
    with (results_dir / "config.json").open("r", encoding="utf-8") as f:
        config = json.load(f)

    plot_bias(hist, config, out, cfg)
    plot_lifetimes(hist, out, cfg)
    plot_reward(hist, out, cfg)
    plot_params(hist, out, cfg)


if __name__ == "__main__":
    main()
