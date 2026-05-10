"""Physical two-point spot-check for Step 7C tracking bandwidth.

This is intentionally small: it validates a few normalized frequencies with
the real two-point Lindblad evaluator rather than the scalar proxy used for
the presentation-level 7C sweep.
"""

from __future__ import annotations

import csv
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
STEP_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step07_physical_validation_mplconfig"
MPL_CONFIG.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG))
os.environ.setdefault("XDG_CACHE_HOME", str(MPL_CONFIG))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from drift_models import sinusoidal_drift
from online_tracking import TARGET_BAND_HIGH, TARGET_BAND_LOW, TARGET_BIAS
from two_points_with_noise import (
    TwoPointConfig,
    cleanup_local_qutip_cache,
    clear_measure_cache,
    measure_lifetimes_two_point,
    params_to_complex,
)


PNG_DPI = 320
FIGSIZE = (7.6, 4.28)
NORMALIZED_FREQUENCIES = [0.25, 0.70, 1.00]
PHYSICAL_EPOCHS = 36
BURN_IN_EPOCHS = 8
N_RUNS = 2
MASTER_SEED = 314159
DRIFT_AMPLITUDE = 0.025

COLORS = {
    "ink": "#263238",
    "grid": "#D5DAE1",
    "target": "#D1495B",
    "bayesian": "#6D3FB2",
    "blt": "#E09F3E",
}


@dataclass(frozen=True)
class PhysicalTracker:
    algorithm: str
    raw_nominal: tuple[float, float, float, float]
    n_conv: int
    static_compensation: float
    signed_sensitivity: float
    gain: float

    @property
    def f_conv(self) -> float:
        return 1.0 / float(self.n_conv)


TRACKERS = [
    PhysicalTracker(
        algorithm="Bayesian/TUrBO",
        raw_nominal=(0.75494017, -0.82341657, 2.31213644, 2.54031506),
        n_conv=6,
        static_compensation=0.17755737595725798,
        signed_sensitivity=-1.0472116558858227,
        gain=0.30,
    ),
    PhysicalTracker(
        algorithm="BLT",
        raw_nominal=(1.6472, 0.5161, 3.8076, -0.3302),
        n_conv=9,
        static_compensation=0.03914758067112414,
        signed_sensitivity=-1.833888973176507,
        gain=1.50,
    ),
]

CSV_FIELDS = [
    "algorithm",
    "normalized_frequency",
    "run_id",
    "epoch",
    "actual_f_drift",
    "f_conv",
    "n_conv",
    "drift_signal",
    "dynamic_compensation",
    "static_compensation",
    "total_g2_scale",
    "bias",
    "T_X",
    "T_Z",
    "tracking_error",
    "target_bias",
    "target_band_low",
    "target_band_high",
    "burn_in",
    "run_rms_tracking_error",
    "run_p_in_band",
    "mean_rms_tracking_error",
    "std_rms_tracking_error",
    "mean_p_in_band",
    "std_p_in_band",
    "n_runs",
    "phase",
    "seed",
    "evaluator",
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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: clean_value(row.get(name, "")) for name in CSV_FIELDS})


def evaluate_exact_bias(raw_nominal: np.ndarray, total_scale: float, sim_cfg: TwoPointConfig) -> dict:
    raw = np.asarray(raw_nominal, dtype=float).copy()
    raw[:2] *= total_scale
    g2, eps_d = params_to_complex(raw)
    result = measure_lifetimes_two_point(g2, eps_d, sim_cfg, use_cache=True)
    if not result.get("valid", False):
        return {
            "bias": np.nan,
            "T_X": np.nan,
            "T_Z": np.nan,
            "tracking_error": np.inf,
        }
    bias = float(result["bias"])
    return {
        "bias": bias,
        "T_X": float(result["T_X"]),
        "T_Z": float(result["T_Z"]),
        "tracking_error": abs(math.log(max(bias, 1.0e-12) / TARGET_BIAS)),
    }


def run_one_physical_tracking(
    tracker: PhysicalTracker,
    *,
    normalized_frequency: float,
    run_id: int,
    phase: float,
    seed: int,
    sim_cfg: TwoPointConfig,
) -> list[dict]:
    epochs = np.arange(PHYSICAL_EPOCHS, dtype=float)
    actual_f_drift = normalized_frequency * tracker.f_conv
    drift = sinusoidal_drift(
        epochs,
        amplitude=DRIFT_AMPLITUDE,
        frequency=actual_f_drift,
        phase=phase,
    )
    raw_nominal = np.asarray(tracker.raw_nominal, dtype=float)
    dynamic_compensation = 0.0
    rows: list[dict] = []
    for epoch, drift_value in enumerate(drift):
        total_residual = tracker.static_compensation + float(drift_value) + dynamic_compensation
        total_scale = 1.0 + total_residual
        metrics = evaluate_exact_bias(raw_nominal, total_scale, sim_cfg)
        error_signed = math.log(max(float(metrics["bias"]), 1.0e-12) / TARGET_BIAS)
        estimated_residual = error_signed / tracker.signed_sensitivity
        dynamic_compensation = float(
            np.clip(
                dynamic_compensation - tracker.gain * estimated_residual,
                -0.14,
                0.14,
            )
        )
        rows.append(
            {
                "algorithm": tracker.algorithm,
                "normalized_frequency": float(normalized_frequency),
                "run_id": int(run_id),
                "epoch": int(epoch),
                "actual_f_drift": float(actual_f_drift),
                "f_conv": float(tracker.f_conv),
                "n_conv": int(tracker.n_conv),
                "drift_signal": float(drift_value),
                "dynamic_compensation": float(dynamic_compensation),
                "static_compensation": float(tracker.static_compensation),
                "total_g2_scale": float(total_scale),
                "bias": float(metrics["bias"]),
                "T_X": float(metrics["T_X"]),
                "T_Z": float(metrics["T_Z"]),
                "tracking_error": float(metrics["tracking_error"]),
                "target_bias": TARGET_BIAS,
                "target_band_low": TARGET_BAND_LOW,
                "target_band_high": TARGET_BAND_HIGH,
                "burn_in": int(epoch < BURN_IN_EPOCHS),
                "phase": float(phase),
                "seed": int(seed),
                "evaluator": "exact_lindblad_two_point",
            }
        )
    post = [row for row in rows if int(row["epoch"]) >= BURN_IN_EPOCHS]
    errors = np.asarray([float(row["tracking_error"]) for row in post], dtype=float)
    biases = np.asarray([float(row["bias"]) for row in post], dtype=float)
    run_rms = float(np.sqrt(np.mean(errors * errors)))
    run_p_in_band = float(np.mean((biases >= TARGET_BAND_LOW) & (biases <= TARGET_BAND_HIGH)))
    for row in rows:
        row["run_rms_tracking_error"] = run_rms
        row["run_p_in_band"] = run_p_in_band
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    for tracker in TRACKERS:
        for frequency in NORMALIZED_FREQUENCIES:
            group = [
                row
                for row in rows
                if row["algorithm"] == tracker.algorithm
                and abs(float(row["normalized_frequency"]) - frequency) < 1.0e-12
                and int(row["epoch"]) == 0
            ]
            rms = np.asarray([float(row["run_rms_tracking_error"]) for row in group], dtype=float)
            p_in = np.asarray([float(row["run_p_in_band"]) for row in group], dtype=float)
            mean_rms = float(np.mean(rms))
            std_rms = float(np.std(rms, ddof=1)) if len(rms) > 1 else 0.0
            mean_p = float(np.mean(p_in))
            std_p = float(np.std(p_in, ddof=1)) if len(p_in) > 1 else 0.0
            for row in rows:
                if row["algorithm"] == tracker.algorithm and abs(float(row["normalized_frequency"]) - frequency) < 1.0e-12:
                    row["mean_rms_tracking_error"] = mean_rms
                    row["std_rms_tracking_error"] = std_rms
                    row["mean_p_in_band"] = mean_p
                    row["std_p_in_band"] = std_p
                    row["n_runs"] = int(N_RUNS)
    return rows


def plot_summary(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    ax.set_title("Physical validation of tracking bandwidth", loc="left", pad=11)
    handles = []
    for tracker, color, marker in (
        (TRACKERS[0], COLORS["bayesian"], "o"),
        (TRACKERS[1], COLORS["blt"], "s"),
    ):
        subset = []
        for frequency in NORMALIZED_FREQUENCIES:
            subset.append(
                next(
                    row
                    for row in rows
                    if row["algorithm"] == tracker.algorithm
                    and abs(float(row["normalized_frequency"]) - frequency) < 1.0e-12
                )
            )
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
            label=tracker.algorithm,
        )
        handles.append(line)
    near_band = ax.axvline(1.0, color=COLORS["target"], linewidth=1.55, linestyle=(0, (5, 4)), label=r"$f_{drift}/f_{conv}=1$")
    ax.set_xlabel(r"$f_{drift}/f_{conv}$")
    ax.set_ylabel("Mean RMS tracking error")
    ax.set_xlim(0.20, 1.03)
    ax.set_ylim(0.0, max(float(row["mean_rms_tracking_error"]) + float(row["std_rms_tracking_error"]) for row in rows) * 1.22)
    ax.grid(True, which="major")
    ax.legend(handles=handles + [near_band], loc="upper left", frameon=True, fancybox=False, edgecolor="#DADFE6")
    fig.savefig(STEP_DIR / "figure_7C_physical_validation_spotcheck.png", facecolor="white")
    fig.savefig(STEP_DIR / "figure_7C_physical_validation_spotcheck.pdf", facecolor="white")
    plt.close(fig)


def update_explanation(rows: list[dict]) -> None:
    bayes_1 = next(row for row in rows if row["algorithm"] == "Bayesian/TUrBO" and abs(float(row["normalized_frequency"]) - 1.0) < 1.0e-12)
    blt_1 = next(row for row in rows if row["algorithm"] == "BLT" and abs(float(row["normalized_frequency"]) - 1.0) < 1.0e-12)
    text = f"""

7C physical spot-check
----------------------
After the proxy 7C sweep, a small physical validation was added. This is not a full Monte Carlo, but it does call the exact Lindblad two-point evaluator at every epoch.

Frequencies tested: {NORMALIZED_FREQUENCIES}.
Runs per point: {N_RUNS}.
Epochs per run: {PHYSICAL_EPOCHS}.
Burn-in excluded from RMS: {BURN_IN_EPOCHS} epochs.
Post-burn-in epochs per run: {PHYSICAL_EPOCHS - BURN_IN_EPOCHS}.
Evaluator: exact two-point Lindblad lifetimes, no added measurement noise, to isolate drift-tracking error from readout noise.

The controls are statically re-centered so the exact no-drift bias is at the target before applying time-dependent drift:
Bayesian/TUrBO static g2 compensation = {TRACKERS[0].static_compensation:.6f}; signed sensitivity = {TRACKERS[0].signed_sensitivity:.6f}.
BLT static g2 compensation = {TRACKERS[1].static_compensation:.6f}; signed sensitivity = {TRACKERS[1].signed_sensitivity:.6f}.

At f_drift/f_conv = 1.0 in this physical spot-check:
Bayesian/TUrBO mean RMS = {float(bayes_1['mean_rms_tracking_error']):.6f}.
BLT mean RMS = {float(blt_1['mean_rms_tracking_error']):.6f}.

Interpretation: this spot-check is the expensive sanity layer. It validates that the figure is no longer only a scalar proxy; however it should still be presented as a limited physical spot-check, because only three frequencies and {N_RUNS} phases per point were evaluated.
"""
    path = STEP_DIR / "step_07_drift_explanation.txt"
    previous = path.read_text(encoding="utf-8")
    marker = "\n7C physical spot-check\n----------------------"
    if marker in previous:
        previous = previous.split(marker)[0].rstrip() + "\n"
    path.write_text(previous.rstrip() + text, encoding="utf-8")


def main() -> None:
    set_slide_style()
    clear_measure_cache()
    sim_cfg = TwoPointConfig()
    rng = np.random.default_rng(MASTER_SEED)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(len(NORMALIZED_FREQUENCIES), N_RUNS))
    seeds = rng.integers(1000, 1_000_000, size=N_RUNS)

    rows: list[dict] = []
    for tracker in TRACKERS:
        for freq_index, frequency in enumerate(NORMALIZED_FREQUENCIES):
            for run_id in range(N_RUNS):
                print(f"physical 7C {tracker.algorithm} f={frequency} run={run_id + 1}/{N_RUNS}", flush=True)
                rows.extend(
                    run_one_physical_tracking(
                        tracker,
                        normalized_frequency=frequency,
                        run_id=run_id,
                        phase=float(phases[freq_index, run_id]),
                        seed=int(seeds[run_id]),
                        sim_cfg=sim_cfg,
                    )
                )
    rows = summarize(rows)
    write_csv(STEP_DIR / "figure_7C_physical_validation_spotcheck.csv", rows)
    plot_summary(rows)
    update_explanation(rows)
    cleanup_local_qutip_cache()
    print("Generated physical 7C spot-check.")


if __name__ == "__main__":
    main()
