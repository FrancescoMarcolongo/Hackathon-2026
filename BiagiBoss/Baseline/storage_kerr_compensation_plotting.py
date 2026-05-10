"""Plots for storage-Kerr drift tracking with an explicit compensation knob."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from plotting import set_style
import matplotlib.pyplot as plt


CONTROL_SERIES = [
    ("g2_real", "Re(g2)", "#165a96"),
    ("g2_imag", "Im(g2)", "#1f7a4d"),
    ("eps_d_real", "Re(epsilon_d)", "#8a5a00"),
    ("eps_d_imag", "Im(epsilon_d)", "#7a3f98"),
]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _epochs(rows: list[dict]) -> np.ndarray:
    return np.array([int(r["epoch"]) for r in rows], dtype=int)


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(r[key]) for r in rows], dtype=float)


def plot_kerr5_bias_vs_epoch(
    rows: list[dict],
    target_bias: float,
    bias_tol_rel: float,
    warmup_epoch: int,
    path: Path,
) -> None:
    set_style()
    epochs = _epochs(rows)
    bias = _series(rows, "bias")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, bias, color="#165a96", lw=2.0, label="optimizer mean")
    ax.axhspan(
        target_bias * (1.0 - bias_tol_rel),
        target_bias * (1.0 + bias_tol_rel),
        color="#165a96",
        alpha=0.12,
        label="success band",
    )
    ax.axhline(target_bias, color="#a83232", lw=1.3, ls="--", label=f"target = {target_bias:g}")
    ax.axvline(warmup_epoch, color="#5f5f5f", lw=1.0, ls=":", label="warm-up")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Bias $\eta = T_Z / T_X$")
    finite = bias[np.isfinite(bias) & (bias > 0)]
    if len(finite):
        lower = min(float(np.min(finite)), target_bias * (1.0 - bias_tol_rel))
        upper = max(float(np.max(finite)), target_bias * (1.0 + bias_tol_rel))
        ax.set_ylim(max(0.0, 0.92 * lower), 1.08 * upper)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_kerr5_lifetimes_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = _epochs(rows)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, _series(rows, "T_X"), color="#1f7a4d", lw=2.0, label=r"$T_X$")
    ax.plot(epochs, _series(rows, "T_Z"), color="#7a3f98", lw=2.0, label=r"$T_Z$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lifetime (us)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_kerr5_reward_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = _epochs(rows)
    reward = _series(rows, "reward")
    pop_mean = _series(rows, "population_reward_mean")
    pop_std = _series(rows, "population_reward_std")
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, reward, color="#8a5a00", lw=2.0, label="optimizer mean reward")
    finite = np.isfinite(pop_mean) & np.isfinite(pop_std)
    if np.any(finite):
        ax.fill_between(
            epochs[finite],
            pop_mean[finite] - pop_std[finite],
            pop_mean[finite] + pop_std[finite],
            color="#8a5a00",
            alpha=0.14,
            label="population mean +/- std",
        )
        ax.plot(epochs[finite], pop_mean[finite], color="#a77b24", lw=1.0, alpha=0.75)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_kerr5_parameters_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = _epochs(rows)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        dpi=200,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax = axes[0]
    for suffix, label, color in CONTROL_SERIES:
        ax.plot(epochs, _series(rows, f"command_{suffix}"), lw=2.0, color=color, label=f"{label} command")
        ax.plot(
            epochs,
            _series(rows, f"true_opt_{suffix}"),
            lw=1.5,
            color=color,
            ls="--",
            label=f"{label} true optimum",
        )
    ax.set_ylabel("Cat controls")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    k_ax = axes[1]
    k_ax.plot(
        epochs,
        _series(rows, "command_storage_kerr"),
        lw=2.0,
        color="#c04f15",
        label="Kerr compensation command",
    )
    k_ax.plot(
        epochs,
        _series(rows, "true_opt_storage_kerr"),
        lw=1.7,
        color="#c04f15",
        ls="--",
        label="Kerr compensation true optimum",
    )
    k_ax.axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.35)
    k_ax.set_xlabel("Epoch")
    k_ax.set_ylabel("Kerr command")
    k_ax.grid(alpha=0.3)
    k_ax.legend(frameon=False)
    _save(fig, path)


def plot_kerr5_signal_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = _epochs(rows)
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, _series(rows, "drift_storage_kerr"), lw=2.0, color="#c04f15", label="storage Kerr drift")
    ax.axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.35)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Storage Kerr K")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_kerr5_tracking_error_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = _epochs(rows)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        dpi=200,
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    ax = axes[0]
    for suffix, label, color in CONTROL_SERIES:
        ax.plot(epochs, _series(rows, f"tracking_error_{suffix}"), lw=1.35, color=color, label=f"{label} error")
    ax.plot(epochs, _series(rows, "tracking_error_l2"), lw=2.0, color="#1f1f1f", label="tracking error L2")
    ax.plot(epochs, _series(rows, "effective_error_l2"), lw=1.5, color="#5f5f5f", ls="--", label="effective error L2")
    ax.axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.4)
    ax.set_ylabel("Control error")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    k_ax = axes[1]
    k_ax.plot(
        epochs,
        _series(rows, "tracking_error_storage_kerr"),
        color="#c04f15",
        lw=2.0,
        label="Kerr command error",
    )
    k_ax.plot(
        epochs,
        _series(rows, "effective_storage_kerr"),
        color="#5f5f5f",
        ls="--",
        lw=1.6,
        label="residual physical Kerr",
    )
    k_ax.axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.4)
    k_ax.set_xlabel("Epoch")
    k_ax.set_ylabel("Kerr error")
    k_ax.grid(alpha=0.3)
    k_ax.legend(frameon=False)
    _save(fig, path)


def plot_kerr5_effective_parameters_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = _epochs(rows)
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        dpi=200,
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )
    ax = axes[0]
    for suffix, label, color in CONTROL_SERIES:
        ax.plot(epochs, _series(rows, f"effective_{suffix}"), lw=2.0, color=color, label=f"{label} effective")
        ax.plot(epochs, _series(rows, f"reference_{suffix}"), lw=1.4, color=color, ls="--", label=f"{label} reference")
    ax.set_ylabel("Effective cat controls")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)

    k_ax = axes[1]
    k_ax.plot(
        epochs,
        _series(rows, "effective_storage_kerr"),
        lw=2.0,
        color="#c04f15",
        label="residual storage Kerr",
    )
    k_ax.plot(
        epochs,
        _series(rows, "reference_storage_kerr"),
        lw=1.4,
        color="#c04f15",
        ls="--",
        label="Kerr reference",
    )
    k_ax.set_xlabel("Epoch")
    k_ax.set_ylabel("Residual Kerr")
    k_ax.grid(alpha=0.3)
    k_ax.legend(frameon=False)
    _save(fig, path)
