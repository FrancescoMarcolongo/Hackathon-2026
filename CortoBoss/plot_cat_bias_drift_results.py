"""Plot diagnostics for cat_bias_optimization_with_drift.py outputs.

Expected input:
    drift_results/drift_optimization_history.csv
    drift_results/metadata.json

Generated diagnostics include the static core metrics from the baseline
optimizer plus drift-specific tracking plots showing:
- control parameters following `x_static_ref + drift(epoch)`;
- effective Hamiltonian parameters `x_control - drift` staying near reference;
- drift terms, tracking errors, and complex-plane trajectories.

Run:
    python plot_cat_bias_drift_results.py --input-dir drift_results
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / "team-core-bias-optimizer" / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


PARAMS = [
    ("re_g2", r"Re($g_2$)"),
    ("im_g2", r"Im($g_2$)"),
    ("re_eps_d", r"Re($\epsilon_d$)"),
    ("im_eps_d", r"Im($\epsilon_d$)"),
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def f(row: dict, key: str, default: float = np.nan) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def arr(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([f(r, key) for r in rows], dtype=float)


def has(rows: list[dict], *keys: str) -> bool:
    return bool(rows) and all(key in rows[0] for key in keys)


def first_available_prefix(rows: list[dict], prefixes: list[str], params: list[tuple[str, str]]) -> str | None:
    for prefix in prefixes:
        if has(rows, *[f"{prefix}_{param}" for param, _label in params]):
            return prefix
    return None


def save(fig: plt.Figure, path: Path, saved: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    saved.append(path)


def warn_skip(name: str, missing: Iterable[str]) -> None:
    print(f"Skipping {name}: missing columns {', '.join(missing)}")


def plot_reward(rows: list[dict], out: Path, saved: list[Path]) -> None:
    if not has(rows, "epoch", "reward", "train_reward", "validation_reward"):
        return warn_skip("reward_vs_epoch", ["epoch", "reward", "train_reward", "validation_reward"])
    set_style()
    epochs = arr(rows, "epoch")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, arr(rows, "reward"), lw=2, label="incumbent reward")
    ax.plot(epochs, arr(rows, "train_reward"), lw=1.2, alpha=0.75, label="epoch best reward")
    ax.plot(epochs, arr(rows, "validation_reward"), lw=1.2, alpha=0.75, label="mean reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Reward vs epoch")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save(fig, out / "reward_vs_epoch.png", saved)


def plot_bias(rows: list[dict], metadata: dict, out: Path, saved: list[Path]) -> None:
    if not has(rows, "epoch", "bias", "epoch_best_bias", "mean_bias"):
        return warn_skip("bias_vs_epoch", ["epoch", "bias", "epoch_best_bias", "mean_bias"])
    target = float(metadata.get("config", {}).get("target_bias", np.nan))
    epochs = arr(rows, "epoch")
    bias = arr(rows, "bias")
    set_style()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, bias, lw=2, label="incumbent bias")
    ax.plot(epochs, arr(rows, "epoch_best_bias"), lw=1.2, alpha=0.75, label="epoch best")
    ax.plot(epochs, arr(rows, "mean_bias"), lw=1.2, alpha=0.75, label="optimizer mean")
    if np.isfinite(target):
        ax.axhline(target, color="#a83232", ls="--", lw=1.2, label=f"target={target:g}")
    finite = bias[np.isfinite(bias) & (bias > 0)]
    if len(finite) and np.max(finite) / max(np.min(finite), 1e-12) > 20:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Bias $\eta=T_Z/T_X$")
    ax.set_title("Bias robustness under drift")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    save(fig, out / "bias_vs_epoch.png", saved)


def plot_lifetimes(rows: list[dict], out: Path, saved: list[Path]) -> None:
    if not has(rows, "epoch", "T_X", "T_Z"):
        return warn_skip("lifetimes_vs_epoch", ["epoch", "T_X", "T_Z"])
    set_style()
    epochs = arr(rows, "epoch")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, arr(rows, "T_X"), lw=2, label=r"incumbent $T_X$")
    ax.plot(epochs, arr(rows, "T_Z"), lw=2, label=r"incumbent $T_Z$")
    if has(rows, "epoch_best_T_X", "epoch_best_T_Z"):
        ax.plot(epochs, arr(rows, "epoch_best_T_X"), "--", lw=1, alpha=0.65, label=r"epoch best $T_X$")
        ax.plot(epochs, arr(rows, "epoch_best_T_Z"), "--", lw=1, alpha=0.65, label=r"epoch best $T_Z$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lifetime (us)")
    ax.set_yscale("log")
    ax.set_title("Lifetime robustness under drift")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    save(fig, out / "lifetimes_vs_epoch.png", saved)


def plot_parameter_traces(rows: list[dict], out: Path, saved: list[Path]) -> None:
    needed = ["epoch"] + [f"control_{p}" for p, _ in PARAMS]
    if not has(rows, *needed):
        return warn_skip("parameter_traces", needed)
    set_style()
    epochs = arr(rows, "epoch")
    fig, ax = plt.subplots(figsize=(7, 4))
    for param, label in PARAMS:
        ax.plot(epochs, arr(rows, f"control_{param}"), lw=1.8, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Control parameter")
    ax.set_title("Optimized control parameter traces")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    save(fig, out / "parameters_vs_epoch.png", saved)


def plot_control_tracks_target(rows: list[dict], out: Path, saved: list[Path]) -> None:
    prefixes = [
        ("mean_control", "optimizer mean", 2.0, "-", 0.95),
        ("control", "incumbent", 1.6, "-", 0.75),
        ("epoch_best_control", "epoch best", 1.2, ":", 0.85),
    ]
    available = [(prefix, label, lw, ls, alpha) for prefix, label, lw, ls, alpha in prefixes if has(rows, *[f"{prefix}_{p}" for p, _ in PARAMS])]
    if not available or not has(rows, "epoch", *[f"target_control_{p}" for p, _ in PARAMS]):
        return warn_skip("control_tracks_target_drift", ["mean/control/epoch_best_control_*", "target_control_*"])
    set_style()
    epochs = arr(rows, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for ax, (param, label) in zip(axes.flat, PARAMS):
        for prefix, series_label, lw, ls, alpha in available:
            ax.plot(epochs, arr(rows, f"{prefix}_{param}"), lw=lw, ls=ls, alpha=alpha, label=series_label)
        ax.plot(epochs, arr(rows, f"target_control_{param}"), lw=1.4, ls="--", color="black", label="target control")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Control parameter follows ideal drift compensation")
    save(fig, out / "control_tracks_target_drift.png", saved)


def plot_effective_near_ref(rows: list[dict], out: Path, saved: list[Path]) -> None:
    prefixes = [
        ("mean_effective", "optimizer mean", 2.0, "-", 0.95),
        ("effective", "incumbent", 1.6, "-", 0.75),
        ("epoch_best_effective", "epoch best", 1.2, ":", 0.85),
    ]
    available = [(prefix, label, lw, ls, alpha) for prefix, label, lw, ls, alpha in prefixes if has(rows, *[f"{prefix}_{p}" for p, _ in PARAMS])]
    if not available or not has(rows, "epoch", *[f"target_effective_{p}" for p, _ in PARAMS]):
        return warn_skip("effective_params_near_static_ref", ["mean/effective/epoch_best_effective_*", "target_effective_*"])
    set_style()
    epochs = arr(rows, "epoch")
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    for ax, (param, label) in zip(axes.flat, PARAMS):
        for prefix, series_label, lw, ls, alpha in available:
            ax.plot(epochs, arr(rows, f"{prefix}_{param}"), lw=lw, ls=ls, alpha=alpha, label=series_label)
        ax.plot(epochs, arr(rows, f"target_effective_{param}"), lw=1.4, ls="--", color="black", label="static ref")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Effective physical parameters near static reference")
    save(fig, out / "effective_params_near_static_ref.png", saved)


def plot_drift_terms(rows: list[dict], out: Path, saved: list[Path]) -> None:
    needed = ["epoch"] + [f"drift_{p}" for p, _ in PARAMS]
    if not has(rows, *needed):
        return warn_skip("drift_terms", needed)
    set_style()
    epochs = arr(rows, "epoch")
    fig, ax = plt.subplots(figsize=(7, 4))
    for param, label in PARAMS:
        ax.plot(epochs, arr(rows, f"drift_{param}"), lw=1.8, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("External drift")
    ax.set_title("Fourier drift vector")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    save(fig, out / "drift_terms.png", saved)


def plot_tracking_error(rows: list[dict], out: Path, saved: list[Path]) -> None:
    needed = ["epoch", "mean_control_error_norm", "mean_effective_error_norm", "control_error_norm", "effective_error_norm"]
    needed += [f"abs_error_mean_control_{p}" for p, _ in PARAMS]
    needed += [f"abs_error_control_{p}" for p, _ in PARAMS]
    if not has(rows, *needed):
        return warn_skip("tracking_error", needed)
    set_style()
    epochs = arr(rows, "epoch")
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    axes[0].plot(epochs, arr(rows, "mean_control_error_norm"), lw=2, label=r"mean $||x_c-x_c^*||_2$")
    axes[0].plot(epochs, arr(rows, "mean_effective_error_norm"), lw=2, label=r"mean $||x_\mathrm{eff}-x_\mathrm{ref}||_2$")
    axes[0].plot(epochs, arr(rows, "control_error_norm"), lw=1.4, alpha=0.75, label=r"incumbent $||x_c-x_c^*||_2$")
    axes[0].plot(epochs, arr(rows, "effective_error_norm"), lw=1.4, alpha=0.75, label=r"incumbent $||x_\mathrm{eff}-x_\mathrm{ref}||_2$")
    axes[0].set_ylabel("L2 norm")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False, ncol=2)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for idx, (param, label) in enumerate(PARAMS):
        color = colors[idx % len(colors)]
        axes[1].plot(epochs, arr(rows, f"abs_error_mean_control_{param}"), lw=1.5, color=color, label=f"mean {label}")
        axes[1].plot(
            epochs,
            arr(rows, f"abs_error_control_{param}"),
            lw=1.1,
            ls="--",
            alpha=0.75,
            color=color,
            label=f"incumbent {label}",
        )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Abs control error")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Drift tracking error")
    save(fig, out / "tracking_error.png", saved)


def complex_plane(rows: list[dict], out: Path, saved: list[Path], name: str, re_key: str, im_key: str) -> None:
    control_prefix = first_available_prefix(rows, ["mean_control", "epoch_best_control", "control"], PARAMS) or "control"
    effective_prefix = first_available_prefix(rows, ["mean_effective", "epoch_best_effective", "effective"], PARAMS) or "effective"
    cols = [
        f"{control_prefix}_{re_key}",
        f"{control_prefix}_{im_key}",
        f"target_control_{re_key}",
        f"target_control_{im_key}",
        f"{effective_prefix}_{re_key}",
        f"{effective_prefix}_{im_key}",
        f"target_effective_{re_key}",
        f"target_effective_{im_key}",
    ]
    if not has(rows, *cols):
        return warn_skip(f"{name}_complex_plane", cols)
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.plot(arr(rows, cols[0]), arr(rows, cols[1]), lw=2, label="control")
    ax.plot(arr(rows, cols[2]), arr(rows, cols[3]), lw=1.4, ls="--", label="target control")
    ax.plot(arr(rows, cols[4]), arr(rows, cols[5]), lw=1.6, label="effective")
    ax.scatter(arr(rows, cols[6])[:1], arr(rows, cols[7])[:1], s=50, marker="x", color="black", label="static ref")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_title(f"{name} complex-plane trajectory")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    ax.axis("equal")
    save(fig, out / f"{name}_complex_plane.png", saved)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="drift_results/")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    rows = load_rows(input_dir / "drift_optimization_history.csv")
    metadata_path = input_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    saved: list[Path] = []

    plot_reward(rows, out, saved)
    plot_bias(rows, metadata, out, saved)
    plot_lifetimes(rows, out, saved)
    plot_parameter_traces(rows, out, saved)
    plot_control_tracks_target(rows, out, saved)
    plot_effective_near_ref(rows, out, saved)
    plot_drift_terms(rows, out, saved)
    plot_tracking_error(rows, out, saved)
    complex_plane(rows, out, saved, "g2", "re_g2", "im_g2")
    complex_plane(rows, out, saved, "eps_d", "re_eps_d", "im_eps_d")

    print("Saved plots:")
    for path in saved:
        print(f"  {path}")


if __name__ == "__main__":
    main()
