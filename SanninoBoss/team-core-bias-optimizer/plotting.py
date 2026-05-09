"""Publication-quality plotting helpers for the optimizer outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


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


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_bias_vs_epoch(rows: list[dict], target_bias: float, path: Path) -> None:
    set_style()
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)
    bias = np.array([float(r["incumbent_bias"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, bias, color="#165a96", lw=2.0, label="validated incumbent")
    ax.axhline(target_bias, color="#a83232", lw=1.3, ls="--", label=f"target = {target_bias:g}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Bias $\eta = T_Z / T_X$")
    finite_bias = bias[np.isfinite(bias) & (bias > 0)]
    if len(finite_bias) and float(np.max(finite_bias) / max(np.min(finite_bias), 1e-12)) > 20.0:
        ax.set_yscale("log")
    else:
        upper = max(float(np.max(finite_bias)) if len(finite_bias) else target_bias, target_bias)
        ax.set_ylim(0.0, 1.12 * upper)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_lifetimes_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)
    tx = np.array([float(r["incumbent_T_X"]) for r in rows], dtype=float)
    tz = np.array([float(r["incumbent_T_Z"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, tx, color="#1f7a4d", lw=2.0, label=r"$T_X$")
    ax.plot(epochs, tz, color="#7a3f98", lw=2.0, label=r"$T_Z$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lifetime (us)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_reward_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)
    reward = np.array([float(r["incumbent_reward"]) for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.plot(epochs, reward, color="#8a5a00", lw=2.0, label="validated incumbent reward")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)


def plot_parameters_vs_epoch(rows: list[dict], path: Path) -> None:
    set_style()
    epochs = np.array([int(r["epoch"]) for r in rows], dtype=int)
    series = [
        ("incumbent_g2_real", r"Re($g_2$)", "#165a96"),
        ("incumbent_g2_imag", r"Im($g_2$)", "#1f7a4d"),
        ("incumbent_eps_d_real", r"Re($\epsilon_d$)", "#8a5a00"),
        ("incumbent_eps_d_imag", r"Im($\epsilon_d$)", "#7a3f98"),
    ]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for key, label, color in series:
        values = np.array([float(r[key]) for r in rows], dtype=float)
        ax.plot(epochs, values, lw=2.0, color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Parameter value")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    _save(fig, path)


def plot_decay_fit(result: dict, path: Path, title: str) -> None:
    set_style()
    curves = result["curves"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2), dpi=200)
    panels = [
        ("times_x", "values_x", "fit_x", r"$\langle X\rangle$", float(result["T_X"])),
        ("times_z", "values_z", "fit_z", r"$\langle Z\rangle$", float(result["T_Z"])),
    ]
    for ax, (t_key, y_key, f_key, ylabel, tau) in zip(axes, panels):
        t = np.asarray(curves[t_key], dtype=float)
        y = np.asarray(curves[y_key], dtype=float)
        fit = np.asarray(curves[f_key], dtype=float)
        ax.plot(t, y, color="#1f1f1f", lw=1.5, label="mesolve")
        ax.plot(t, fit, color="#d65f5f", lw=1.4, ls="--", label=f"fit, T={tau:.2f} us")
        ax.set_xlabel("Time (us)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle(title)
    _save(fig, path)


def plot_sweep_summary(rows: Iterable[dict], target_bias: float, path: Path) -> None:
    set_style()
    data = list(rows)
    if not data:
        return
    bias = np.array([float(r["bias"]) for r in data], dtype=float)
    geo = np.array([float(r["geo_lifetime"]) for r in data], dtype=float)
    feasible = bias >= target_bias
    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    ax.scatter(bias[~feasible], geo[~feasible], s=18, color="#a9a9a9", alpha=0.65, label="below target")
    ax.scatter(bias[feasible], geo[feasible], s=22, color="#165a96", alpha=0.85, label="target reached")
    ax.axvline(target_bias, color="#a83232", lw=1.3, ls="--", label="target")
    ax.set_xlabel(r"Bias $\eta = T_Z / T_X$")
    ax.set_ylabel(r"Geometric lifetime $\sqrt{T_XT_Z}$ (us)")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    _save(fig, path)
