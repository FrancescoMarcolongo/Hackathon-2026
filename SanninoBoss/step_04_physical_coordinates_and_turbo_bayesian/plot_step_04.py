"""Produce the revised Step 4 figures and their CSV data tables."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

STEP_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step04_turbo_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from physical_coordinates_reward import PhysicalOptimizerConfig, run_physical_reward_optimizer
from turbo_bayesian_physical_optimizer import (
    TurboBayesianOptConfig,
    run_turbo_bayesian_physical_optimizer,
)
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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows_for_curve(
    rows: list[dict],
    *,
    algorithm: str,
    optimizer_type: str,
    include_turbo_state: bool = False,
) -> list[dict]:
    target_lower = TARGET_BIAS * (1.0 - BIAS_TOL_REL)
    target_upper = TARGET_BIAS * (1.0 + BIAS_TOL_REL)
    output = []
    for row in rows:
        item = {
            "epoch": int(float(row["epoch"])),
            "bias": float(row["incumbent_bias"]),
            "algorithm": algorithm,
            "target_bias": float(TARGET_BIAS),
            "target_lower": float(target_lower),
            "target_upper": float(target_upper),
            "incumbent_reward": float(row.get("incumbent_reward", np.nan)),
            "incumbent_T_X": float(row.get("incumbent_T_X", np.nan)),
            "incumbent_T_Z": float(row.get("incumbent_T_Z", np.nan)),
            "optimizer_type": optimizer_type,
            "trust_region_length": "",
            "success_counter": "",
            "failure_counter": "",
            "best_reward": "",
            "restart_count": "",
        }
        if include_turbo_state:
            item.update(
                {
                    "trust_region_length": float(row["trust_region_length"]),
                    "success_counter": int(row["success_counter"]),
                    "failure_counter": int(row["failure_counter"]),
                    "best_reward": float(row["best_reward"]),
                    "restart_count": int(row["restart_count"]),
                }
            )
        output.append(item)
    return output


CSV_FIELDS = [
    "epoch",
    "bias",
    "algorithm",
    "target_bias",
    "target_lower",
    "target_upper",
    "incumbent_reward",
    "incumbent_T_X",
    "incumbent_T_Z",
    "optimizer_type",
    "trust_region_length",
    "success_counter",
    "failure_counter",
    "best_reward",
    "restart_count",
]


def plot_figure_4a(normal_rows: list[dict], physical_rows: list[dict]) -> None:
    x_normal = as_array(normal_rows, "epoch")
    y_normal = as_array(normal_rows, "incumbent_bias")
    x_physical = as_array(physical_rows, "epoch")
    y_physical = as_array(physical_rows, "incumbent_bias")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    add_target(ax)
    plot_line(ax, x_normal, y_normal, label="2-points with noise", color=COLORS["normal"], marker="o")
    plot_line(ax, x_physical, y_physical, label="physical coordinates + reward", color=COLORS["physical"], marker="s")
    style_axes(ax, title="Physical optimization vs noisy optimization")
    set_limits(ax, y_normal, y_physical)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "figure_4A_physical_reward_vs_two_points_noise")

    csv_rows = (
        csv_rows_for_curve(
            normal_rows,
            algorithm="2-points with noise",
            optimizer_type="raw CMA-ES noisy two-point",
        )
        + csv_rows_for_curve(
            physical_rows,
            algorithm="physical coordinates + reward",
            optimizer_type="physical-coordinate CMA-ES",
        )
    )
    write_csv(STEP_DIR / "figure_4A_physical_reward_vs_two_points_noise.csv", csv_rows, CSV_FIELDS)


def plot_figure_4b(physical_rows: list[dict], turbo_rows: list[dict]) -> None:
    x_physical = as_array(physical_rows, "epoch")
    y_physical = as_array(physical_rows, "incumbent_bias")
    x_turbo = as_array(turbo_rows, "epoch")
    y_turbo = as_array(turbo_rows, "incumbent_bias")

    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    add_target(ax)
    plot_line(ax, x_physical, y_physical, label="physical coordinates + reward", color=COLORS["physical"], marker="s")
    plot_line(ax, x_turbo, y_turbo, label="trust-region Bayesian (TUrBO-like)", color=COLORS["bayesian"], marker="D")
    style_axes(ax, title="Bayesian optimization vs Physical coordinates")
    set_limits(ax, y_physical, y_turbo)
    ax.legend(loc="lower right", frameon=True, fancybox=False, framealpha=0.94, edgecolor="#E5E7EB")
    finish_figure(fig, "figure_4B_bayesian_vs_physical_reward")

    csv_rows = (
        csv_rows_for_curve(
            physical_rows,
            algorithm="physical coordinates + reward",
            optimizer_type="physical-coordinate CMA-ES",
        )
        + csv_rows_for_curve(
            turbo_rows,
            algorithm="trust-region Bayesian (TUrBO-like)",
            optimizer_type="trust-region Bayesian / TUrBO-like",
            include_turbo_state=True,
        )
    )
    write_csv(STEP_DIR / "figure_4B_bayesian_vs_physical_reward.csv", csv_rows, CSV_FIELDS)


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
    turbo_payload = run_turbo_bayesian_physical_optimizer(
        verbose=True,
        turbo_cfg=TurboBayesianOptConfig(
            max_epochs=MAX_EPOCHS,
            n_init=4,
            beta=0.4,
            n_pool=4096,
            global_fraction=0.12,
            initial_trust_region_length=0.58,
            length_min=0.04,
            length_max=0.90,
            success_tolerance=2,
            failure_tolerance=4,
            improvement_tol=1.0e-3,
            expand_factor=1.6,
            shrink_factor=1.7,
            warm_start_length=0.48,
            random_seed=7,
            noise_seed=NOISE_SEED,
            noise_sigma=NOISE_SIGMA,
            use_alpha_correction=True,
        ),
    )

    normal_rows = normal_payload["history"]
    physical_rows = physical_payload["history"]
    turbo_rows = turbo_payload["history"]

    plot_figure_4a(normal_rows, physical_rows)
    plot_figure_4b(physical_rows, turbo_rows)

    print(
        "Saved revised step 4 figures and CSVs. "
        f"first target epochs: normal={first_target_epoch(normal_rows)}, "
        f"physical={first_target_epoch(physical_rows)}, "
        f"turbo={first_target_epoch(turbo_rows)}."
    )


if __name__ == "__main__":
    main()
