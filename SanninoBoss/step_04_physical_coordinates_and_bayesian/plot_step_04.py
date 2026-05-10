"""Produce the Step 4 figures with the Step 1 slide style."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

STEP_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step04_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from bayesian_physical_optimizer import BayesianOptConfig, run_bayesian_physical_optimizer
from physical_coordinates_reward import PhysicalOptimizerConfig, run_physical_reward_optimizer
from two_points_with_noise import BIAS_TOL_REL, TARGET_BIAS, run_two_points_with_noise


PNG_DPI = 320
FIGSIZE = (7.6, 4.28)

MAX_EPOCHS = 45
NOISE_SIGMA = 0.03
NOISE_SEED = 11

COLORS = {
    "ink": "#263238",
    "grid": "#D5DAE1",
    "normal": "#008B8B",
    "physical": "#2563EB",
    "bayesian": "#6D3FB2",
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


def style_axes(ax: plt.Axes, *, title: str) -> None:
    ax.set_title(title, loc="left", pad=11)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Bias $\eta=T_Z/T_X$")
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
    lower = TARGET_BIAS * (1.0 - BIAS_TOL_REL)
    upper = TARGET_BIAS * (1.0 + BIAS_TOL_REL)
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


def set_limits(ax: plt.Axes, *series: np.ndarray) -> None:
    upper = max([TARGET_BIAS * (1.0 + BIAS_TOL_REL)] + [float(np.nanmax(s)) for s in series])
    ax.set_xlim(0, MAX_EPOCHS)
    ax.set_ylim(0.0, upper * 1.09)
    ax.margins(x=0.015)


def first_target_epoch(rows: list[dict], key: str = "incumbent_bias") -> int | None:
    lower = TARGET_BIAS * (1.0 - BIAS_TOL_REL)
    upper = TARGET_BIAS * (1.0 + BIAS_TOL_REL)
    for row in rows:
        value = float(row[key])
        if lower <= value <= upper:
            return int(float(row["epoch"]))
    return None


def plot_figure_4a(normal_rows: list[dict], physical_rows: list[dict]) -> None:
    x_normal = as_array(normal_rows, "epoch")
    y_normal = as_array(normal_rows, "incumbent_bias")
    x_physical = as_array(physical_rows, "epoch")
    y_physical = as_array(physical_rows, "incumbent_bias")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    add_target(ax)
    plot_line(ax, x_normal, y_normal, label="2-points with noise", color=COLORS["normal"], marker="o")
    plot_line(ax, x_physical, y_physical, label="physical coordinates + reward", color=COLORS["physical"], marker="s")
    style_axes(ax, title="Physical coordinates improve noisy optimization")
    set_limits(ax, y_normal, y_physical)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "figure_4A_physical_reward_vs_two_points_noise")


def plot_figure_4b(physical_rows: list[dict], bayesian_rows: list[dict]) -> None:
    x_physical = as_array(physical_rows, "epoch")
    y_physical = as_array(physical_rows, "incumbent_bias")
    x_bayes = as_array(bayesian_rows, "epoch")
    y_bayes = as_array(bayesian_rows, "incumbent_bias")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    add_target(ax)
    plot_line(ax, x_physical, y_physical, label="physical coordinates + reward", color=COLORS["physical"], marker="s")
    plot_line(ax, x_bayes, y_bayes, label="Bayesian physical optimizer", color=COLORS["bayesian"], marker="D")
    style_axes(ax, title="Bayesian optimization accelerates convergence")
    set_limits(ax, y_physical, y_bayes)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "figure_4B_bayesian_vs_physical_reward")


def main() -> None:
    set_slide_style()

    normal_payload = run_two_points_with_noise(
        verbose=True,
        epochs=MAX_EPOCHS,
        population=8,
        sigma0=0.25,
        optimizer_seed=0,
        noise_seed=NOISE_SEED,
        noise_sigma=NOISE_SIGMA,
    )
    physical_payload = run_physical_reward_optimizer(
        verbose=True,
        opt_cfg=PhysicalOptimizerConfig(
            epochs=MAX_EPOCHS,
            population=8,
            sigma0=0.32,
            optimizer_seed=4,
            noise_seed=NOISE_SEED,
            noise_sigma=NOISE_SIGMA,
            use_alpha_correction=True,
        ),
    )
    bayesian_payload = run_bayesian_physical_optimizer(
        verbose=True,
        bo_cfg=BayesianOptConfig(
            max_epochs=MAX_EPOCHS,
            n_init=4,
            beta=0.8,
            n_pool=4096,
            local_pool_fraction=0.7,
            local_scale_logs=0.28,
            local_scale_phases=0.50,
            random_seed=9,
            noise_seed=NOISE_SEED,
            noise_sigma=NOISE_SIGMA,
            use_alpha_correction=True,
        ),
    )

    normal_rows = normal_payload["history"]
    physical_rows = physical_payload["history"]
    bayesian_rows = bayesian_payload["history"]

    plot_figure_4a(normal_rows, physical_rows)
    plot_figure_4b(physical_rows, bayesian_rows)

    print(
        "Saved step 4 figures. "
        f"first target epochs: normal={first_target_epoch(normal_rows)}, "
        f"physical={first_target_epoch(physical_rows)}, "
        f"bayesian={first_target_epoch(bayesian_rows)}."
    )


if __name__ == "__main__":
    main()
