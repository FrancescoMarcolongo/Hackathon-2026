"""Produce the three step-3 figures with the step-1 slide style."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

STEP_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step03_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from baseline_loader import (
    BASELINE_N_POINTS,
    TARGET_BIAS,
    TARGET_TOLERANCE_REL,
    load_baseline_bias_trace,
)
from two_points_algorithms import run_two_points_no_noise, run_two_points_with_noise


PNG_DPI = 320
FIGSIZE = (7.6, 4.28)
PANEL_FIGSIZE = (7.6, 6.25)

NOISE_SIGMA = 0.03
OPTIMIZER_SEED = 0
NOISE_SEED = 11
NOISY_EPOCHS = 70
POPULATION = 8
SIGMA0 = 0.25

COLORS = {
    "ink": "#263238",
    "grid": "#D5DAE1",
    "bias": "#2563EB",
    "noisy": "#008B8B",
    "target": "#D1495B",
    "target_band": "#D1495B",
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
    ax.set_title(title, loc="left", pad=10)
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


def add_target(ax: plt.Axes) -> None:
    lower = TARGET_BIAS * (1.0 - TARGET_TOLERANCE_REL)
    upper = TARGET_BIAS * (1.0 + TARGET_TOLERANCE_REL)
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
        TARGET_BIAS,
        color=COLORS["target"],
        linewidth=1.75,
        linestyle=(0, (5, 4)),
        label=fr"target $\eta={TARGET_BIAS:.0f}$",
        zorder=2,
    )


def finish_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(STEP_DIR / f"{filename}.png", facecolor="white")
    fig.savefig(STEP_DIR / f"{filename}.pdf", facecolor="white")
    plt.close(fig)


def set_x_range(ax: plt.Axes, x: np.ndarray) -> None:
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.margins(x=0.015)


def set_bias_ylim(ax: plt.Axes, *series: np.ndarray) -> None:
    upper = max([TARGET_BIAS * (1.0 + TARGET_TOLERANCE_REL)] + [float(np.nanmax(s)) for s in series])
    ax.set_ylim(0.0, upper * 1.09)


def first_target_epoch(rows: list[dict], key: str = "incumbent_bias") -> int | None:
    lower = TARGET_BIAS * (1.0 - TARGET_TOLERANCE_REL)
    upper = TARGET_BIAS * (1.0 + TARGET_TOLERANCE_REL)
    for row in rows:
        bias = float(row[key])
        if lower <= bias <= upper:
            return int(float(row["epoch"]))
    return None


def plot_figure_3a(no_noise_rows: list[dict]) -> None:
    epochs = as_array(no_noise_rows, "epoch")
    bias = as_array(no_noise_rows, "incumbent_bias")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    add_target(ax)
    plot_line(ax, epochs, bias, label=fr"2-points no noise $\eta=T_Z/T_X$", color=COLORS["bias"])
    style_axes(ax, title="2-points no noise", xlabel="Epoch", ylabel=fr"Bias $\eta=T_Z/T_X$")
    set_bias_ylim(ax, bias)
    set_x_range(ax, epochs)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "figure_3A_two_points_no_noise")


def plot_two_panel(
    baseline_rows: list[dict],
    noisy_rows: list[dict],
    *,
    rescale_noisy_x: bool,
    filename: str,
) -> None:
    baseline_x = as_array(baseline_rows, "epoch")
    baseline_bias = as_array(baseline_rows, "bias")
    noisy_x = as_array(noisy_rows, "epoch")
    if rescale_noisy_x:
        noisy_x = noisy_x * (2.0 / float(BASELINE_N_POINTS))
    noisy_bias = as_array(noisy_rows, "incumbent_bias")

    fig, axes = plt.subplots(2, 1, figsize=PANEL_FIGSIZE, constrained_layout=True)

    add_target(axes[0])
    plot_line(axes[0], baseline_x, baseline_bias, label="Baseline no noise", color=COLORS["bias"])
    style_axes(axes[0], title="Baseline no noise", xlabel="Epoch", ylabel="Bias")
    set_bias_ylim(axes[0], baseline_bias, noisy_bias)
    set_x_range(axes[0], baseline_x)
    axes[0].legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")

    add_target(axes[1])
    plot_line(axes[1], noisy_x, noisy_bias, label="2-points with noise", color=COLORS["noisy"])
    xlabel = "Normalized physical cost" if rescale_noisy_x else "Epoch"
    style_axes(axes[1], title="2-points with noise", xlabel=xlabel, ylabel="Bias")
    set_bias_ylim(axes[1], baseline_bias, noisy_bias)
    set_x_range(axes[1], noisy_x)
    axes[1].legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")

    finish_figure(fig, filename)


def main() -> None:
    set_slide_style()
    baseline_rows = load_baseline_bias_trace()
    no_noise_payload = run_two_points_no_noise(verbose=True)
    noisy_payload = run_two_points_with_noise(
        verbose=True,
        epochs=NOISY_EPOCHS,
        population=POPULATION,
        sigma0=SIGMA0,
        optimizer_seed=OPTIMIZER_SEED,
        noise_seed=NOISE_SEED,
        noise_sigma=NOISE_SIGMA,
    )

    no_noise_rows = no_noise_payload["history"]
    noisy_rows = noisy_payload["history"]
    plot_figure_3a(no_noise_rows)
    plot_two_panel(
        baseline_rows,
        noisy_rows,
        rescale_noisy_x=False,
        filename="figure_3B_epochs_baseline_vs_noisy_two_points",
    )
    plot_two_panel(
        baseline_rows,
        noisy_rows,
        rescale_noisy_x=True,
        filename="figure_3C_physical_time_rescaled",
    )

    baseline_first = first_target_epoch(baseline_rows, key="bias")
    no_noise_first = first_target_epoch(no_noise_rows)
    noisy_first = first_target_epoch(noisy_rows)
    final_noisy = noisy_payload["incumbent"]
    print(
        "Saved step 3 figures. "
        f"first target epochs: baseline={baseline_first}, "
        f"2-points no-noise={no_noise_first}, noisy={noisy_first}. "
        f"final noisy bias={float(final_noisy['bias']):.4g}"
    )


if __name__ == "__main__":
    main()
