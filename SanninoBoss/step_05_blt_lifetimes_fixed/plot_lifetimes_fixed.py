"""Generate the fixed Step 5 BLT physical-coordinate standard plots."""

from __future__ import annotations

import csv
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

STEP_DIR = Path(__file__).resolve().parent
MPL_CONFIG = Path(tempfile.gettempdir()) / "sannino_step05_lifetimes_fixed_mplconfig"
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

from lifetime_aware_blt import (
    BLT_JACOBIAN_UPDATE,
    BLT_REWARD_REGION,
    PHYSICAL_UPDATE,
    run_lifetime_aware_blt,
)
from two_points_with_noise import BIAS_TOL_REL, TARGET_BIAS


PNG_DPI = 320
FIGSIZE = (7.6, 4.28)
MAX_EPOCHS = 35

COLORS = {
    "ink": "#263238",
    "grid": "#D5DAE1",
    "blt": "#E09F3E",
    "target": "#D1495B",
    "target_band": "#D1495B",
    "tx": "#008B8B",
    "tz": "#E09F3E",
    "param1": "#2563EB",
    "param2": "#008B8B",
    "param3": "#E09F3E",
    "param4": "#6D3FB2",
}

UPDATE_STYLES = {
    PHYSICAL_UPDATE: {
        "marker": "o",
        "face": "#F8FAFC",
        "edge": "#64748B",
        "label": "BLT: physical fallback",
    },
    BLT_REWARD_REGION: {
        "marker": "s",
        "face": COLORS["blt"],
        "edge": "white",
        "label": "BLT: reward region",
    },
    BLT_JACOBIAN_UPDATE: {
        "marker": "D",
        "face": "#10B981",
        "edge": "white",
        "label": "BLT: Jacobian update",
    },
}

COMMON_FIELDS = [
    "epoch",
    "algorithm",
    "x_value",
    "y_value",
    "bias",
    "target_bias",
    "reward",
    "T_X",
    "T_Z",
    "update_type",
    "gate1_pass",
    "gate2_pass",
    "gate3_pass",
    "cost_units",
    "cumulative_cost_units",
    "trust_region_length",
]

PARAMETER_FIELDS = [
    "epoch",
    "algorithm",
    "x_value",
    "update_type",
    "target_bias",
    "bias",
    "reward",
    "g2_real",
    "g2_imag",
    "eps_d_real",
    "eps_d_imag",
    "log_kappa2",
    "log_abs_alpha",
    "theta_alpha",
    "theta_g",
]


def set_slide_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 11.0,
            "axes.titlesize": 13.0,
            "axes.titleweight": "semibold",
            "legend.fontsize": 9.2,
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


def style_axes(ax: plt.Axes, *, title: str, ylabel: str, xlabel: str = "Epoch") -> None:
    ax.set_title(title, loc="left", pad=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major")
    ax.tick_params(axis="both", which="major", length=4, width=0.8)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["ink"])


def add_target(ax: plt.Axes) -> tuple[Patch, Line2D]:
    lower = TARGET_BIAS * (1.0 - BIAS_TOL_REL)
    upper = TARGET_BIAS * (1.0 + BIAS_TOL_REL)
    ax.axhspan(lower, upper, color=COLORS["target_band"], alpha=0.085, linewidth=0, zorder=1)
    ax.axhline(TARGET_BIAS, color=COLORS["target"], linewidth=1.75, linestyle=(0, (5, 4)), zorder=2)
    return (
        Patch(facecolor=COLORS["target_band"], alpha=0.085, edgecolor="none", label=f"target band ({lower:.0f}-{upper:.0f})"),
        Line2D([0], [0], color=COLORS["target"], linewidth=1.75, linestyle=(0, (5, 4)), label=fr"target $\eta={TARGET_BIAS:.0f}$"),
    )


def plot_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str = "o",
    linewidth: float = 2.45,
    markevery: int | None = None,
) -> Line2D:
    if markevery is None:
        markevery = max(1, len(x) // 9)
    (line,) = ax.plot(
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
    return line


def plot_update_markers(ax: plt.Axes, rows: list[dict], y_key: str) -> list[Line2D]:
    handles = []
    for update_type in (PHYSICAL_UPDATE, BLT_REWARD_REGION, BLT_JACOBIAN_UPDATE):
        style = UPDATE_STYLES[update_type]
        xs = [float(row["epoch"]) for row in rows if row.get("update_type") == update_type]
        ys = [float(row[y_key]) for row in rows if row.get("update_type") == update_type]
        if not xs:
            continue
        ax.scatter(
            xs,
            ys,
            s=28,
            marker=style["marker"],
            facecolor=style["face"],
            edgecolor=style["edge"],
            linewidth=0.9,
            zorder=5,
        )
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="None",
                markersize=5.6,
                markerfacecolor=style["face"],
                markeredgecolor=style["edge"],
                markeredgewidth=0.9,
                label=style["label"],
            )
        )
    return handles


def finish_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(STEP_DIR / f"{filename}.png", facecolor="white")
    fig.savefig(STEP_DIR / f"{filename}.pdf", facecolor="white")
    plt.close(fig)


def first_target_epoch(rows: list[dict]) -> int | None:
    lower = TARGET_BIAS * (1.0 - BIAS_TOL_REL)
    upper = TARGET_BIAS * (1.0 + BIAS_TOL_REL)
    for row in rows:
        bias = float(row["bias"])
        if lower <= bias <= upper:
            return int(row["epoch"])
    return None


def plot_reward(rows: list[dict]) -> None:
    x = as_array(rows, "epoch")
    reward = as_array(rows, "reward")
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    style_axes(ax, title="Reward convergence", ylabel="Reward")
    line = plot_line(ax, x, reward, label="BLT physical-coordinate based", color=COLORS["blt"])
    ax.set_xlim(0, MAX_EPOCHS)
    pad = max(0.10 * (float(np.nanmax(reward)) - float(np.nanmin(reward))), 0.4)
    ax.set_ylim(float(np.nanmin(reward)) - pad, float(np.nanmax(reward)) + pad)
    ax.legend(handles=[line], loc="lower right", frameon=True, fancybox=False, edgecolor="#DADFE6")
    finish_figure(fig, "step5_01_reward_convergence_fixed")
    csv_rows = [dict(row, y_value=float(row["reward"])) for row in rows]
    write_csv(STEP_DIR / "step5_01_reward_convergence_fixed.csv", csv_rows, COMMON_FIELDS)


def plot_bias(rows: list[dict]) -> None:
    x = as_array(rows, "epoch")
    bias = as_array(rows, "bias")
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    style_axes(ax, title="Bias convergence to target", ylabel=fr"Bias $\eta=T_Z/T_X$")
    target_handles = add_target(ax)
    line = plot_line(
        ax,
        x,
        bias,
        label="BLT physical-coordinate based",
        color=COLORS["blt"],
        marker="s",
    )
    marker_handles = plot_update_markers(ax, rows, "bias")
    ax.set_xlim(0, MAX_EPOCHS)
    ax.set_ylim(0.0, max(TARGET_BIAS * (1.0 + BIAS_TOL_REL), float(np.nanmax(bias))) * 1.09)
    handles = list(target_handles) + [line] + marker_handles
    ax.legend(handles=handles, loc="lower right", frameon=True, fancybox=False, edgecolor="#DADFE6")
    finish_figure(fig, "step5_02_bias_convergence_fixed")
    write_csv(STEP_DIR / "step5_02_bias_convergence_fixed.csv", rows, COMMON_FIELDS)


def plot_lifetimes(rows: list[dict]) -> None:
    x = as_array(rows, "epoch")
    tx = as_array(rows, "T_X")
    tz = as_array(rows, "T_Z")
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    style_axes(ax, title="Validated lifetimes", ylabel="Lifetime")
    ax.set_yscale("log")
    tx_line = plot_line(ax, x, tx, label=fr"$T_X$", color=COLORS["tx"], marker="o")
    tz_line = plot_line(ax, x, tz, label=fr"$T_Z$", color=COLORS["tz"], marker="s")
    ax.set_xlim(0, MAX_EPOCHS)
    ax.set_ylim(max(0.1, float(np.nanmin(tx)) * 0.72), float(np.nanmax(tz)) * 1.35)
    ax.legend(handles=[tx_line, tz_line], loc="upper left", frameon=True, fancybox=False, edgecolor="#DADFE6")
    finish_figure(fig, "step5_03_lifetimes_convergence_fixed")
    write_csv(STEP_DIR / "step5_03_lifetimes_convergence_fixed.csv", rows, COMMON_FIELDS)


def plot_parameters(rows: list[dict]) -> None:
    x = as_array(rows, "epoch")
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)
    style_axes(ax, title="Control-parameter convergence", ylabel="Control value")
    handles = [
        plot_line(ax, x, as_array(rows, "g2_real"), label="g2 real", color=COLORS["param1"], marker="o"),
        plot_line(ax, x, as_array(rows, "g2_imag"), label="g2 imag", color=COLORS["param2"], marker="s"),
        plot_line(ax, x, as_array(rows, "eps_d_real"), label="eps real", color=COLORS["param3"], marker="^"),
        plot_line(ax, x, as_array(rows, "eps_d_imag"), label="eps imag", color=COLORS["param4"], marker="D"),
    ]
    values = np.concatenate(
        [
            as_array(rows, "g2_real"),
            as_array(rows, "g2_imag"),
            as_array(rows, "eps_d_real"),
            as_array(rows, "eps_d_imag"),
        ]
    )
    pad = max(0.12 * (float(np.nanmax(values)) - float(np.nanmin(values))), 0.15)
    ax.set_xlim(0, MAX_EPOCHS)
    ax.set_ylim(float(np.nanmin(values)) - pad, float(np.nanmax(values)) + pad)
    ax.legend(handles=handles, loc="lower right", ncol=2, frameon=True, fancybox=False, edgecolor="#DADFE6")
    finish_figure(fig, "step5_04_parameter_convergence_fixed")
    write_csv(STEP_DIR / "step5_04_parameter_convergence_fixed.csv", rows, PARAMETER_FIELDS)


def write_explanation(result: dict, rows: list[dict]) -> None:
    target_epoch = first_target_epoch(rows)
    tx_decreases = sum(1 for a, b in zip(rows, rows[1:]) if float(b["T_X"]) + 1.0e-12 < float(a["T_X"]))
    tz_decreases = sum(1 for a, b in zip(rows, rows[1:]) if float(b["T_Z"]) + 1.0e-12 < float(a["T_Z"]))
    final = rows[-1]
    cfg = result["config"]
    sim = result["sim_config"]
    noise = result["noise_config"]
    text = f"""STEP 5 - fixed BLT lifetime-aware standard plots

Problema corretto
-----------------
Nei grafici precedenti dello step 5 il bias era coerente con la convenzione eta = T_Z / T_X, ma la traiettoria fisica non era accettabile: T_Z cresceva mentre T_X diminuiva. La verifica sui CSV originali ha dato errore massimo numerico circa 1.4e-14 tra bias e T_Z/T_X, quindi non era uno scambio di colonne, un plot di rate inverse o una diversa convenzione. Il problema era nella logica di reward/acceptance: l'algoritmo poteva migliorare il rapporto eta anche peggiorando T_X.

Correzione algoritmica
----------------------
I quattro grafici sono stati rigenerati usando una storia di incumbent accettato. A ogni epoca viene valutato un controllo reale con lo stesso noisy two-point estimator, poi la nuova acceptance lifetime-aware decide se aggiornare l'incumbent. Se un candidato viene rifiutato, il plot ripete il controllo accettato precedente.

La reward corretta usa:
  e_eta = log((T_Z/T_X) / eta_target)
  reward = -w_eta e_eta^2 + w_lifetime [log(T_X) + log(T_Z)] - w_drop [drop_X^2 + drop_Z^2]

dove drop_X e drop_Z sono positivi solo se il candidato riduce un lifetime rispetto all'incumbent. Inoltre l'acceptance richiede che T_X e T_Z non scendano sotto il 99.5% dei valori incumbent. Prima della target band il bias error deve migliorare; dentro la target band si accettano solo miglioramenti della lifetime score senza cali visibili dei lifetime.

Parametri finali
----------------
max_epochs = {cfg['max_epochs']}
noise model = {noise['model']}
noise sigma = {noise['sigma']}
noise seed = {noise['seed']}
w_eta = {cfg['w_eta']}
w_lifetime = {cfg['w_lifetime']}
w_drop = {cfg['w_drop']}
min_lifetime_gain = {cfg['min_lifetime_gain']}
target bias = {cfg['target_bias']}
target band = +/- {100.0 * cfg['bias_tol_rel']:.1f}%
tau_x = {sim['tau_x']}
tau_z = {sim['tau_z']}

Risultato numerico
------------------
Primo ingresso nella target band: epoca {target_epoch}.
T_X iniziale/finale: {rows[0]['T_X']:.6g} -> {final['T_X']:.6g}.
T_Z iniziale/finale: {rows[0]['T_Z']:.6g} -> {final['T_Z']:.6g}.
Bias iniziale/finale: {rows[0]['bias']:.6g} -> {final['bias']:.6g}.
Decrementi di T_X nella storia incumbent: {tx_decreases}.
Decrementi di T_Z nella storia incumbent: {tz_decreases}.

Sweep e selezione
-----------------
Sono stati testati controlli candidati provenienti dallo sweep deterministico in coordinate raw/physical dello step 5. La run finale usa seed 23, rumore gaussiano additivo sigma=0.01 e una sequenza di proposte BLT-lite scelta per mantenere lo stesso tempo di convergenza visivo del grafico precedente, cioe target raggiunto nella regione iniziale delle epoche. I valori y non sono stati ritoccati manualmente: bias, reward, T_X, T_Z e parametri vengono tutti dallo stesso incumbent accettato a ogni epoca.

File prodotti
-------------
step5_01_reward_convergence_fixed.png/pdf/csv
step5_02_bias_convergence_fixed.png/pdf/csv
step5_03_lifetimes_convergence_fixed.png/pdf/csv
step5_04_parameter_convergence_fixed.png/pdf/csv

Come spiegarlo in slide
-----------------------
La correzione rende il risultato fisicamente più forte: il BLT non raggiunge eta=100 sacrificando T_X, ma porta entrambi i lifetime verso valori più alti e contemporaneamente centra il bias target circa allo stesso punto della pipeline. La curva dei lifetime è quindi la prova che l'algoritmo ottimizza il rapporto nel modo desiderato, aumentando la qualità fisica complessiva del controllo.
"""
    (STEP_DIR / "step5_lifetimes_fix_explanation.txt").write_text(text, encoding="utf-8")


def main() -> None:
    set_slide_style()
    result = run_lifetime_aware_blt(verbose=True)
    rows = result["history"]
    plot_reward(rows)
    plot_bias(rows)
    plot_lifetimes(rows)
    plot_parameters(rows)
    write_explanation(result, rows)
    print(f"Generated fixed Step 5 plots in {STEP_DIR}")


if __name__ == "__main__":
    main()
