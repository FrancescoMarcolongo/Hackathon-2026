"""Simple slide-style exponential decay plot."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

import numpy as np


OUT_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_simple_decay_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

COLORS = {
    "ink": "#263238",
    "grid": "#D5DAE1",
    "blue": "#2563EB",
}


def set_slide_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 15.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "figure.dpi": 320,
            "savefig.dpi": 320,
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


def main() -> None:
    set_slide_style()
    t = np.linspace(0.0, 3.0, 240)
    x_expectation = np.exp(-t)

    fig, ax = plt.subplots(figsize=(7.6, 4.28), constrained_layout=True)
    ax.plot(t, x_expectation, color=COLORS["blue"], linewidth=2.6)
    ax.set_xlabel(r"$t_{\mathrm{sim}}$")
    ax.set_ylabel(r"$\langle X\rangle$")
    ax.set_xlim(0.0, 3.0)
    ax.set_ylim(0.0, 1.04)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", which="major", length=4, width=0.8)

    fig.savefig(OUT_DIR / "exponential_decay_X.png", facecolor="white")
    fig.savefig(OUT_DIR / "exponential_decay_X.pdf", facecolor="white")
    plt.close(fig)

    with (OUT_DIR / "exponential_decay_X.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t_sim", "X_expectation"])
        writer.writerows(zip(t, x_expectation))


if __name__ == "__main__":
    main()
