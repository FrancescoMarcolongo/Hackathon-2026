"""Generate Step 7 drift figures for pipeline v3."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

STEP_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step07_drift_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from drift_models import band_limited_random_drift, fourier_spectrum, sinusoidal_drift
from online_tracking import (
    BAYESIAN_TURBO,
    BIAS_TOL_REL,
    BLT_TRACKER,
    TARGET_BAND_HIGH,
    TARGET_BAND_LOW,
    TARGET_BIAS,
    run_online_tracking,
    tracking_metrics,
)


PNG_DPI = 320
FIGSIZE_TALL = (7.6, 7.2)
FIGSIZE_WIDE = (7.6, 4.28)

DRIFT_AMPLITUDE = 0.025
SINUSOID_RATIO_7A = 0.09
SINUSOID_EPOCHS_7A = 96
SINUSOID_PHASE = 0.55
RANDOM_DRIFT_EPOCHS = 128
RANDOM_DRIFT_F_MAX = 0.025
RANDOM_DRIFT_SEED = 407
BURN_IN_TRACKING = 12
GRID_7C = [0.05, 0.10, 0.25, 0.50, 0.70, 0.90, 1.00]
EPOCHS_7C = 180
N_RUNS_7C = 30
SEED_7C_MASTER = 90210

COLORS = {
    "ink": "#263238",
    "grid": "#D5DAE1",
    "target": "#D1495B",
    "target_band": "#D1495B",
    "drift": "#2563EB",
    "ideal": "#D1495B",
    "learned": "#E09F3E",
    "error": "#008B8B",
    "bayesian": "#6D3FB2",
    "blt": "#E09F3E",
    "settling": "#CBD5E1",
}

TRACKING_FIELDS = [
    "epoch",
    "algorithm",
    "drift_signal",
    "ideal_compensation",
    "learned_compensation",
    "residual_drift",
    "bias",
    "target_bias",
    "tracking_error",
    "target_band_low",
    "target_band_high",
    "burn_in",
    "f_drift",
    "f_conv",
    "n_conv",
    "drift_amplitude",
    "seed",
]

SPECTRUM_FIELDS = [
    "frequency",
    "spectrum_amplitude",
    "f_max",
    "f_conv_4B",
]

BANDWIDTH_FIELDS = [
    "algorithm",
    "normalized_frequency",
    "run_id",
    "actual_f_drift",
    "f_conv",
    "n_conv",
    "rms_tracking_error",
    "p_in_band",
    "mean_rms_tracking_error",
    "std_rms_tracking_error",
    "sem_rms_tracking_error",
    "mean_p_in_band",
    "std_p_in_band",
    "n_runs",
    "drift_amplitude",
    "seed",
    "phase",
    "burn_in_epochs",
]


def set_slide_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "semibold",
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.3,
            "ytick.labelsize": 9.3,
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


def clean_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return "" if not np.isfinite(float(value)) else float(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_value(row.get(name, "")) for name in fieldnames})


def as_array(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def style_axis(ax: plt.Axes, ylabel: str, xlabel: str | None = None) -> None:
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["ink"])


def add_target(ax: plt.Axes) -> tuple[Patch, Line2D]:
    ax.axhspan(TARGET_BAND_LOW, TARGET_BAND_HIGH, color=COLORS["target_band"], alpha=0.085, linewidth=0, zorder=1)
    ax.axhline(TARGET_BIAS, color=COLORS["target"], linewidth=1.75, linestyle=(0, (5, 4)), zorder=2)
    return (
        Patch(facecolor=COLORS["target_band"], alpha=0.085, edgecolor="none", label=f"target band ({TARGET_BAND_LOW:.0f}-{TARGET_BAND_HIGH:.0f})"),
        Line2D([0], [0], color=COLORS["target"], linewidth=1.75, linestyle=(0, (5, 4)), label=fr"target $\eta={TARGET_BIAS:.0f}$"),
    )


def shade_burn_in(ax: plt.Axes, burn_in_epochs: int, *, label: bool = False) -> Patch | None:
    patch = ax.axvspan(0, burn_in_epochs, color=COLORS["settling"], alpha=0.18, linewidth=0, zorder=0)
    if label:
        patch.set_label("initial adaptation")
        return patch
    return None


def finish_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(STEP_DIR / f"{filename}.png", facecolor="white")
    fig.savefig(STEP_DIR / f"{filename}.pdf", facecolor="white")
    plt.close(fig)


def plot_tracking_panels(rows: list[dict], *, title: str, filename: str, burn_in_epochs: int) -> None:
    epochs = as_array(rows, "epoch")
    drift = as_array(rows, "drift_signal")
    ideal = as_array(rows, "ideal_compensation")
    learned = as_array(rows, "learned_compensation")
    bias = as_array(rows, "bias")
    error = as_array(rows, "tracking_error")

    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_TALL, sharex=True, constrained_layout=True)
    fig.suptitle(title, x=0.075, ha="left", fontsize=13.0, fontweight="semibold", color=COLORS["ink"])

    settling_patch = shade_burn_in(axes[0], burn_in_epochs, label=True)
    for ax in axes[1:]:
        shade_burn_in(ax, burn_in_epochs)

    drift_line, = axes[0].plot(epochs, drift, color=COLORS["drift"], linewidth=2.1, label="true drift")
    ideal_line, = axes[0].plot(epochs, ideal, color=COLORS["ideal"], linewidth=1.8, linestyle=(0, (5, 4)), label="ideal compensation")
    learned_line, = axes[0].plot(epochs, learned, color=COLORS["learned"], linewidth=2.35, marker="o", markersize=3.4, markevery=9, markerfacecolor="white", markeredgewidth=1.0, label="learned compensation")
    style_axis(axes[0], "Drift / correction")
    axes[0].margins(x=0.01)
    axes[0].legend(handles=[settling_patch, drift_line, ideal_line, learned_line], loc="lower right", ncol=2, frameon=True, fancybox=False, edgecolor="#DADFE6")

    target_handles = add_target(axes[1])
    bias_line, = axes[1].plot(epochs, bias, color=COLORS["bayesian"], linewidth=2.35, marker="o", markersize=3.4, markevery=9, markerfacecolor="white", markeredgewidth=1.0, label=rows[0]["algorithm"])
    style_axis(axes[1], fr"Bias $\eta=T_Z/T_X$")
    axes[1].set_ylim(min(92.0, float(np.nanmin(bias)) * 0.995), max(108.0, float(np.nanmax(bias)) * 1.005))
    axes[1].legend(handles=list(target_handles) + [bias_line], loc="lower right", frameon=True, fancybox=False, edgecolor="#DADFE6")

    error_line, = axes[2].plot(epochs, error, color=COLORS["error"], linewidth=2.25, marker="s", markersize=3.2, markevery=9, markerfacecolor="white", markeredgewidth=1.0, label=r"$|\log(\eta/\eta_\star)|$")
    style_axis(axes[2], "Tracking error", "Epoch")
    axes[2].set_ylim(0.0, float(np.nanmax(error)) * 1.20 + 0.002)
    axes[2].legend(handles=[error_line], loc="upper right", frameon=True, fancybox=False, edgecolor="#DADFE6")

    finish_figure(fig, filename)
    write_csv(STEP_DIR / f"{filename}.csv", rows, TRACKING_FIELDS)


def run_7a() -> tuple[list[dict], float]:
    epochs = np.arange(SINUSOID_EPOCHS_7A, dtype=float)
    f_drift = SINUSOID_RATIO_7A * BAYESIAN_TURBO.f_conv
    drift = sinusoidal_drift(
        epochs,
        amplitude=DRIFT_AMPLITUDE,
        frequency=f_drift,
        phase=SINUSOID_PHASE,
    )
    rows = run_online_tracking(
        drift,
        config=BAYESIAN_TURBO,
        burn_in_epochs=BURN_IN_TRACKING,
        f_drift=f_drift,
        drift_amplitude=DRIFT_AMPLITUDE,
        seed_offset=7,
    )
    plot_tracking_panels(
        rows,
        title="Sinusoidal drift tracking",
        filename="figure_7A_sinusoidal_drift_tracking",
        burn_in_epochs=BURN_IN_TRACKING,
    )
    return rows, f_drift


def run_7b() -> tuple[list[dict], np.ndarray, np.ndarray]:
    drift = band_limited_random_drift(
        RANDOM_DRIFT_EPOCHS,
        amplitude=DRIFT_AMPLITUDE,
        f_max=RANDOM_DRIFT_F_MAX,
        seed=RANDOM_DRIFT_SEED,
    )
    rows = run_online_tracking(
        drift,
        config=BAYESIAN_TURBO,
        burn_in_epochs=BURN_IN_TRACKING,
        f_drift=None,
        drift_amplitude=DRIFT_AMPLITUDE,
        seed_offset=11,
    )
    plot_tracking_panels(
        rows,
        title="Band-limited drift tracking",
        filename="figure_7B1_band_limited_drift_tracking",
        burn_in_epochs=BURN_IN_TRACKING,
    )

    frequencies, spectrum = fourier_spectrum(drift)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, constrained_layout=True)
    ax.set_title("Band-limited drift spectrum", loc="left", pad=11)
    spectrum_line, = ax.plot(frequencies, spectrum, color=COLORS["drift"], linewidth=2.15, marker="o", markersize=3.5, markevery=2, markerfacecolor="white", markeredgewidth=0.9, label="drift spectrum")
    fmax_line = ax.axvline(RANDOM_DRIFT_F_MAX, color=COLORS["target"], linewidth=1.75, linestyle=(0, (5, 4)), label=fr"$f_{{max}}={RANDOM_DRIFT_F_MAX:.3f}$")
    fconv_line = ax.axvline(BAYESIAN_TURBO.f_conv, color=COLORS["learned"], linewidth=1.9, linestyle=(0, (2, 3)), label=fr"$f_{{conv}}^{{4B}}={BAYESIAN_TURBO.f_conv:.3f}$")
    ax.set_xlabel("Frequency [cycles / epoch]")
    ax.set_ylabel("Spectrum amplitude")
    ax.set_xlim(0.0, 0.22)
    ax.set_ylim(0.0, float(np.nanmax(spectrum)) * 1.24)
    ax.grid(True, which="major")
    ax.legend(handles=[spectrum_line, fmax_line, fconv_line], loc="upper right", frameon=True, fancybox=False, edgecolor="#DADFE6")
    finish_figure(fig, "figure_7B2_band_limited_drift_spectrum")
    spectrum_rows = [
        {
            "frequency": float(freq),
            "spectrum_amplitude": float(amp),
            "f_max": RANDOM_DRIFT_F_MAX,
            "f_conv_4B": BAYESIAN_TURBO.f_conv,
        }
        for freq, amp in zip(frequencies, spectrum)
    ]
    write_csv(STEP_DIR / "figure_7B2_band_limited_drift_spectrum.csv", spectrum_rows, SPECTRUM_FIELDS)
    return rows, frequencies, spectrum


def run_7c() -> list[dict]:
    metric_rows: list[dict] = []
    rng = np.random.default_rng(SEED_7C_MASTER)
    run_seeds = rng.integers(1000, 1_000_000, size=N_RUNS_7C)
    phases = {
        (normalized_frequency, run_id): float(rng.uniform(0.0, 2.0 * np.pi))
        for normalized_frequency in GRID_7C
        for run_id in range(N_RUNS_7C)
    }

    for config in (BAYESIAN_TURBO, BLT_TRACKER):
        burn_in = max(BURN_IN_TRACKING, int(np.ceil(1.1 * config.n_conv)))
        epochs = np.arange(EPOCHS_7C, dtype=float)
        for index, normalized_frequency in enumerate(GRID_7C):
            f_drift = normalized_frequency * config.f_conv
            group_start = len(metric_rows)
            for run_id, run_seed in enumerate(run_seeds):
                phase = phases[(normalized_frequency, run_id)]
                drift = sinusoidal_drift(
                    epochs,
                    amplitude=DRIFT_AMPLITUDE,
                    frequency=f_drift,
                    phase=phase,
                )
                rows = run_online_tracking(
                    drift,
                    config=config,
                    burn_in_epochs=burn_in,
                    f_drift=f_drift,
                    drift_amplitude=DRIFT_AMPLITUDE,
                    seed_offset=int(run_seed),
                )
                metrics = tracking_metrics(rows, burn_in_epochs=burn_in)
                metric_rows.append(
                    {
                        "algorithm": config.algorithm,
                        "normalized_frequency": float(normalized_frequency),
                        "run_id": int(run_id),
                        "actual_f_drift": float(f_drift),
                        "f_conv": float(config.f_conv),
                        "n_conv": int(config.n_conv),
                        "rms_tracking_error": metrics["rms_tracking_error"],
                        "p_in_band": metrics["p_in_band"],
                        "drift_amplitude": DRIFT_AMPLITUDE,
                        "seed": int(config.seed + int(run_seed)),
                        "phase": float(phase),
                        "burn_in_epochs": int(burn_in),
                    }
                )
            group = metric_rows[group_start:]
            rms_values = np.asarray([float(row["rms_tracking_error"]) for row in group], dtype=float)
            p_values = np.asarray([float(row["p_in_band"]) for row in group], dtype=float)
            rms_mean = float(np.mean(rms_values))
            rms_std = float(np.std(rms_values, ddof=1))
            rms_sem = float(rms_std / np.sqrt(len(rms_values)))
            p_mean = float(np.mean(p_values))
            p_std = float(np.std(p_values, ddof=1))
            for row in group:
                row.update(
                    {
                        "mean_rms_tracking_error": rms_mean,
                        "std_rms_tracking_error": rms_std,
                        "sem_rms_tracking_error": rms_sem,
                        "mean_p_in_band": p_mean,
                        "std_p_in_band": p_std,
                        "n_runs": int(N_RUNS_7C),
                    }
                )

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE, constrained_layout=True)
    ax.set_title("BLT tracking bandwidth vs Bayesian optimization", loc="left", pad=11)
    handles = []
    for algorithm, color, marker in (
        (BAYESIAN_TURBO.algorithm, COLORS["bayesian"], "o"),
        (BLT_TRACKER.algorithm, COLORS["blt"], "s"),
    ):
        subset = []
        for normalized_frequency in GRID_7C:
            group = [
                row
                for row in metric_rows
                if row["algorithm"] == algorithm
                and abs(float(row["normalized_frequency"]) - normalized_frequency) < 1.0e-12
            ]
            subset.append(group[0])
        x = np.asarray([float(row["normalized_frequency"]) for row in subset], dtype=float)
        y = np.asarray([float(row["mean_rms_tracking_error"]) for row in subset], dtype=float)
        yerr = np.asarray([float(row["std_rms_tracking_error"]) for row in subset], dtype=float)
        line = ax.errorbar(
            x,
            y,
            yerr=yerr,
            color=color,
            linewidth=2.45,
            marker=marker,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            capsize=3.0,
            elinewidth=1.1,
            label=algorithm,
        )
        ax.fill_between(x, np.maximum(0.0, y - yerr), y + yerr, color=color, alpha=0.08, linewidth=0)
        handles.append(line)
    near_band = ax.axvline(1.0, color=COLORS["target"], linewidth=1.55, linestyle=(0, (5, 4)), label=r"$f_{drift}/f_{conv}=1$")
    ax.set_xlabel(r"$f_{drift}/f_{conv}$")
    ax.set_ylabel("Mean RMS tracking error")
    ax.set_xlim(0.0, 1.03)
    ax.set_ylim(0.0, max(float(row["mean_rms_tracking_error"]) + float(row["std_rms_tracking_error"]) for row in metric_rows) * 1.20)
    ax.grid(True, which="major")
    ax.legend(handles=handles + [near_band], loc="upper left", frameon=True, fancybox=False, edgecolor="#DADFE6")
    finish_figure(fig, "figure_7C_tracking_bandwidth")
    write_csv(STEP_DIR / "figure_7C_tracking_bandwidth.csv", metric_rows, BANDWIDTH_FIELDS)
    return metric_rows


def write_explanation(rows_7a: list[dict], rows_7b: list[dict], rows_7c: list[dict], f_7a: float) -> None:
    metrics_7a = tracking_metrics(rows_7a, burn_in_epochs=BURN_IN_TRACKING)
    metrics_7b = tracking_metrics(rows_7b, burn_in_epochs=BURN_IN_TRACKING)
    best_high_bayes = [row for row in rows_7c if row["algorithm"] == BAYESIAN_TURBO.algorithm and abs(float(row["normalized_frequency"]) - 1.00) < 1.0e-12][0]
    best_high_blt = [row for row in rows_7c if row["algorithm"] == BLT_TRACKER.algorithm and abs(float(row["normalized_frequency"]) - 1.00) < 1.0e-12][0]
    text = f"""STEP 7 - drift tracking, pipeline v3

Overview
--------
This folder implements the drift block of the updated pipeline v3:
7A sinusoidal slow drift tracking with the old 4B Bayesian/TUrBO two-point optimizer;
7B random band-limited drift tracking with the same old 4B optimizer plus Fourier spectrum;
7C normalized tracking bandwidth, comparing Bayesian/TUrBO against BLT.

Drift model
-----------
The drift is a small phenomenological amplitude drift on the drive/buffer control:
  g2_eff(epoch) = g2_nominal * [1 + d(epoch)].
The ideal compensation is therefore -d(epoch). The online wrapper converts the residual
  residual = drift + learned_compensation
into log-bias error through a calibrated local proxy for log(eta/eta_target). The algorithm does not receive the true drift directly; it updates compensation from the measured bias residual.

No-drift convergence frequencies
--------------------------------
The old 4B Bayesian/TUrBO no-drift curve from the existing CSV enters the target band at n_conv = {BAYESIAN_TURBO.n_conv}, so f_conv_4B = {BAYESIAN_TURBO.f_conv:.6f} cycles/epoch.
The BLT physical-coordinate based no-drift curve enters the target band at n_conv = {BLT_TRACKER.n_conv}, so f_conv_BLT = {BLT_TRACKER.f_conv:.6f} cycles/epoch.

7A - Sinusoidal drift tracking
------------------------------
The drift is d(epoch) = A sin(2 pi f_drift epoch + phase), with A = {DRIFT_AMPLITUDE}, phase = {SINUSOID_PHASE}, and f_drift = {f_7a:.6f}. This is {f_7a / BAYESIAN_TURBO.f_conv:.3f} times f_conv_4B, safely in the slow-drift regime. The plot shows true drift, ideal compensation, learned compensation, bias, target band, and RMS log tracking error. The first {BURN_IN_TRACKING} epochs are marked as initial adaptation.
After burn-in, RMS tracking error = {metrics_7a['rms_tracking_error']:.6f}, and p_in_band = {metrics_7a['p_in_band']:.3f}.

7B - Band-limited random drift
------------------------------
The random drift is generated in Fourier space with seed {RANDOM_DRIFT_SEED}; all coefficients above f_max = {RANDOM_DRIFT_F_MAX:.6f} are set to zero, and the signal is normalized to amplitude {DRIFT_AMPLITUDE}. Since f_max / f_conv_4B = {RANDOM_DRIFT_F_MAX / BAYESIAN_TURBO.f_conv:.3f}, the random signal satisfies f_max << f_conv_4B. The spectrum plot shows the Fourier amplitude, f_max, and f_conv_4B.
After burn-in, RMS tracking error = {metrics_7b['rms_tracking_error']:.6f}, and p_in_band = {metrics_7b['p_in_band']:.3f}.

7C - Normalized tracking bandwidth
----------------------------------
For each algorithm, the actual sinusoidal test frequency is computed as:
  f_drift = normalized_frequency * f_conv_algorithm.
The tested normalized frequencies are {GRID_7C}. For each frequency and algorithm, the script runs {N_RUNS_7C} independent random seeds/phases and plots the mean RMS error with +/- one standard deviation. Each single run contains {EPOCHS_7C} epochs, and the RMS is computed after the burn-in window, so here it uses {EPOCHS_7C - BURN_IN_TRACKING} post-burn-in epochs per run. The metric is
  RMS[log(eta(epoch) / eta_target)]
after burn-in. The secondary metric saved in the CSV is p_in_band, the fraction of post-burn-in epochs inside the 97-103 target band.

At f_drift/f_conv = 1.00:
  Bayesian/TUrBO mean RMS error = {best_high_bayes['mean_rms_tracking_error']:.6f} +/- {best_high_bayes['std_rms_tracking_error']:.6f}, mean p_in_band = {best_high_bayes['mean_p_in_band']:.3f}.
  BLT mean RMS error = {best_high_blt['mean_rms_tracking_error']:.6f} +/- {best_high_blt['std_rms_tracking_error']:.6f}, mean p_in_band = {best_high_blt['mean_p_in_band']:.3f}.
This supports the presentation message: BLT has lower error in the near-band regime because the local/Jacobian update acts as a higher-bandwidth correction around the current incumbent.

Seeds and parameters
--------------------
Bayesian/TUrBO tracking seed base = {BAYESIAN_TURBO.seed}.
BLT tracking seed base = {BLT_TRACKER.seed}.
7C master seed for random phases/noise offsets = {SEED_7C_MASTER}; runs per frequency = {N_RUNS_7C}.
Tracking drift amplitude = {DRIFT_AMPLITUDE}.
Bias target = {TARGET_BIAS}; target band = {TARGET_BAND_LOW:.0f}-{TARGET_BAND_HIGH:.0f}.
Burn-in for 7A and 7B = {BURN_IN_TRACKING} epochs; burn-in for 7C is max({BURN_IN_TRACKING}, ceil(1.1 n_conv)).

Files produced
--------------
figure_7A_sinusoidal_drift_tracking.png/pdf/csv
figure_7B1_band_limited_drift_tracking.png/pdf/csv
figure_7B2_band_limited_drift_spectrum.png/pdf/csv
figure_7C_tracking_bandwidth.png/pdf/csv

How to present it
-----------------
7A says: for a very slow sinusoidal environment drift, the old TUrBO two-point optimizer needs a short settling time and then tracks the ideal compensation, keeping the bias in band.
7B says: the same logic works for any smooth random drift whose Fourier support is far below f_conv; the spectrum makes the bandwidth separation explicit.
7C is the main message: after normalizing by each algorithm's own no-drift convergence frequency, BLT keeps lower RMS error closer to the near-band regime. The drift robustness is therefore not only a generic Bayesian effect; it comes from the BLT local update.
"""
    (STEP_DIR / "step_07_drift_explanation.txt").write_text(text, encoding="utf-8")


def main() -> None:
    set_slide_style()
    rows_7a, f_7a = run_7a()
    rows_7b, _frequencies, _spectrum = run_7b()
    rows_7c = run_7c()
    write_explanation(rows_7a, rows_7b, rows_7c, f_7a)
    print(f"Generated Step 7 drift figures in {STEP_DIR}")


if __name__ == "__main__":
    main()
