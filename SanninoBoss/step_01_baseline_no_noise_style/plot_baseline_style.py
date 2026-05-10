"""Redraw the no-noise baseline convergence plots with a shared slide style.

The script reads only persisted outputs from team-core-bias-optimizer/results.
It does not run Dynamiqs, Lindblad simulations, CMA-ES, sweeps, refinements, or
any other numeric optimization.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

sys.dont_write_bytecode = True

MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step01_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

from load_baseline_data import load_step1_data


STEP_DIR = Path(__file__).resolve().parent

PNG_DPI = 320
FIGSIZE = (7.6, 4.28)

COLORS = {
    "ink": "#263238",
    "muted": "#667085",
    "grid": "#D5DAE1",
    "reward": "#B7791F",
    "bias": "#2563EB",
    "target": "#D1495B",
    "target_band": "#D1495B",
    "tx": "#008B8B",
    "tz": "#6D3FB2",
    "g2_re": "#1F4E79",
    "g2_im": "#2A9D8F",
    "eps_re": "#E09F3E",
    "eps_im": "#C04ABC",
    "zero": "#98A2B3",
}


def set_slide_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "semibold",
            "legend.fontsize": 9.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "figure.dpi": PNG_DPI,
            "savefig.dpi": PNG_DPI,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.75,
            "grid.alpha": 0.72,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def as_array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def style_axes(ax: plt.Axes, *, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", pad=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["ink"])


def plot_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str = "o",
    linewidth: float = 2.45,
) -> None:
    markevery = max(1, len(x) // 9)
    ax.plot(
        x,
        y,
        color=color,
        linewidth=linewidth,
        marker=marker,
        markersize=4.8,
        markerfacecolor="white",
        markeredgewidth=1.2,
        markevery=markevery,
        label=label,
        zorder=3,
    )


def finish_figure(fig: plt.Figure, filename: str) -> None:
    png_path = STEP_DIR / f"{filename}.png"
    pdf_path = STEP_DIR / f"{filename}.pdf"
    fig.savefig(png_path, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)


def set_epoch_ticks(ax: plt.Axes, epochs: np.ndarray) -> None:
    ax.set_xlim(float(np.min(epochs)), float(np.max(epochs)))
    ax.margins(x=0.015)


def plot_reward(rows: list[dict]) -> None:
    epochs = as_array(rows, "epoch")
    reward = as_array(rows, "incumbent_reward")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    plot_line(
        ax,
        epochs,
        reward,
        label="validated incumbent reward",
        color=COLORS["reward"],
        marker="o",
    )
    ax.axhline(0.0, color=COLORS["zero"], linewidth=1.15, linestyle=(0, (4, 4)), label="zero reward")
    style_axes(
        ax,
        title="Reward convergence",
        xlabel="Epoch",
        ylabel="Reward",
    )
    y_pad = 0.08 * (float(np.nanmax(reward)) - float(np.nanmin(reward)))
    ax.set_ylim(float(np.nanmin(reward)) - y_pad, float(np.nanmax(reward)) + y_pad)
    set_epoch_ticks(ax, epochs)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "01_reward_convergence")


def plot_bias(rows: list[dict], target_bias: float, target_tolerance_rel: float) -> None:
    epochs = as_array(rows, "epoch")
    bias = as_array(rows, "incumbent_bias")
    lower = target_bias * (1.0 - target_tolerance_rel)
    upper = target_bias * (1.0 + target_tolerance_rel)

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.axhspan(
        lower,
        upper,
        color=COLORS["target_band"],
        alpha=0.085,
        linewidth=0,
        label=f"target band ({lower:.0f}-{upper:.0f})",
        zorder=1,
    )
    ax.axhline(
        target_bias,
        color=COLORS["target"],
        linewidth=1.75,
        linestyle=(0, (5, 4)),
        label=fr"target $\eta={target_bias:.0f}$",
        zorder=2,
    )
    plot_line(
        ax,
        epochs,
        bias,
        label=fr"validated incumbent $\eta=T_Z/T_X$",
        color=COLORS["bias"],
        marker="o",
    )
    style_axes(
        ax,
        title="Bias convergence to target",
        xlabel="Epoch",
        ylabel=fr"Bias $\eta=T_Z/T_X$",
    )
    upper_limit = max(float(np.nanmax(bias)), upper) * 1.09
    ax.set_ylim(0.0, upper_limit)
    set_epoch_ticks(ax, epochs)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "02_bias_convergence")


def plot_lifetimes(rows: list[dict]) -> None:
    epochs = as_array(rows, "epoch")
    tx = as_array(rows, "incumbent_T_X")
    tz = as_array(rows, "incumbent_T_Z")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    plot_line(ax, epochs, tx, label=fr"$T_X$", color=COLORS["tx"], marker="o")
    plot_line(ax, epochs, tz, label=fr"$T_Z$", color=COLORS["tz"], marker="s")
    style_axes(
        ax,
        title="Validated lifetimes",
        xlabel="Epoch",
        ylabel="Lifetime (us)",
    )
    ax.set_yscale("log")
    ymin = min(float(np.nanmin(tx)), float(np.nanmin(tz))) * 0.75
    ymax = max(float(np.nanmax(tx)), float(np.nanmax(tz))) * 1.25
    ax.set_ylim(ymin, ymax)
    lifetime_ticks = [0.2, 0.5, 1, 2, 5, 10, 20]
    ax.yaxis.set_major_locator(FixedLocator(lifetime_ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _pos: f"{value:g}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    set_epoch_ticks(ax, epochs)
    ax.legend(loc="center right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "03_lifetimes_convergence")


def plot_parameters(rows: list[dict]) -> None:
    epochs = as_array(rows, "epoch")
    series: Iterable[tuple[str, str, str, str]] = (
        ("incumbent_g2_real", fr"Re($g_2$)", COLORS["g2_re"], "o"),
        ("incumbent_g2_imag", fr"Im($g_2$)", COLORS["g2_im"], "s"),
        ("incumbent_eps_d_real", fr"Re($\epsilon_d$)", COLORS["eps_re"], "^"),
        ("incumbent_eps_d_imag", fr"Im($\epsilon_d$)", COLORS["eps_im"], "D"),
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for key, label, color, marker in series:
        plot_line(ax, epochs, as_array(rows, key), label=label, color=color, marker=marker, linewidth=2.25)
    style_axes(
        ax,
        title="Control-parameter convergence",
        xlabel="Epoch",
        ylabel="Parameter value",
    )
    set_epoch_ticks(ax, epochs)
    fig.subplots_adjust(bottom=0.28)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=True,
        fancybox=False,
        framealpha=0.94,
        edgecolor="#E5E7EB",
    )
    finish_figure(fig, "04_parameter_convergence")


def main() -> None:
    set_slide_style()
    data = load_step1_data()
    rows = data["trace"]
    plot_reward(rows)
    plot_bias(rows, float(data["target_bias"]), float(data["target_tolerance_rel"]))
    plot_lifetimes(rows)
    plot_parameters(rows)
    print(f"Saved 4 restyled PNG/PDF figure pairs in {STEP_DIR}")


if __name__ == "__main__":
    main()
