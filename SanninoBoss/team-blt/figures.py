"""Publication-quality figure helpers for BLT validation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


def _pyplot():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def set_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    return paths


def add_identity_line(ax, *, label: str = "y = x", color: str = "0.25") -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lo = min(xlim[0], ylim[0])
    hi = max(xlim[1], ylim[1])
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=color, linewidth=1.0, label=label)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def add_target_eta_line(ax, eta_target: float, *, orientation: str = "horizontal") -> None:
    if orientation == "horizontal":
        ax.axhline(eta_target, linestyle="--", color="tab:red", linewidth=1.0)
        ax.annotate(
            f"target eta = {eta_target:g}",
            xy=(0.99, eta_target),
            xycoords=("axes fraction", "data"),
            xytext=(-4, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            color="tab:red",
            fontsize=8,
        )
    else:
        ax.axvline(eta_target, linestyle="--", color="tab:red", linewidth=1.0)


def _finite_float(row: dict, key: str) -> float:
    try:
        value = float(row.get(key, np.nan))
    except (TypeError, ValueError):
        value = np.nan
    return value


def _valid_xy(rows: Sequence[dict], x_key: str, y_key: str) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for row in rows:
        x = _finite_float(row, x_key)
        y = _finite_float(row, y_key)
        if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
            xs.append(x)
            ys.append(y)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def plot_decay_fit(curves: dict, axis: str, t0: float, t1: float, out_dir: Path) -> list[Path]:
    plt = _pyplot()

    set_style()
    times = np.asarray(curves[f"times_{axis}"], dtype=float)
    values = np.asarray(curves[f"values_{axis}"], dtype=float)
    fit = curves[f"fit_{axis}"]
    y_fit = np.asarray(fit["fit"], dtype=float)
    residual = values - y_fit
    tau = float(fit["params"][1])
    r2 = float(fit["r2"])

    fig, (ax, axr) = plt.subplots(
        2,
        1,
        figsize=(6.2, 4.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax.scatter(times, values, s=18, color="tab:blue", alpha=0.85, label="measurements")
    ax.plot(times, y_fit, color="tab:orange", linewidth=2.0, label="robust exponential fit")
    for marker_t, label in ((t0, "BLT t0"), (t1, "BLT t1")):
        ax.axvline(marker_t, color="0.25", linestyle=":", linewidth=1.2)
        ax.annotate(label, xy=(marker_t, 0.97), xycoords=("data", "axes fraction"), rotation=90, va="top", fontsize=8)
    ax.set_ylabel(f"{axis.upper()} contrast")
    ax.set_title(f"Representative {axis.upper()} decay fit: T = {tau:.3g}, R2 = {r2:.3f}")
    ax.legend(frameon=False, loc="best")

    axr.axhline(0.0, color="0.25", linewidth=1.0)
    axr.scatter(times, residual, s=14, color="tab:purple", alpha=0.85)
    axr.set_xlabel("wait time")
    axr.set_ylabel("residual")
    fig.align_ylabels([ax, axr])
    return savefig(fig, out_dir, f"representative_{axis.upper()}_decay_fit")


def plot_blt_vs_gold_rates(rows: Sequence[dict], out_dir: Path) -> list[Path]:
    plt = _pyplot()

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    specs = [
        ("lite_gamma_x", "BLT-lite gamma_X", "gamma_x_gold", "gamma_x_lite"),
        ("lite_gamma_z", "BLT-lite gamma_Z", "gamma_z_gold", "gamma_z_lite"),
        ("full_gamma_x", "BLT-full gamma_X", "gamma_x_gold", "gamma_x_full"),
        ("full_gamma_z", "BLT-full gamma_Z", "gamma_z_gold", "gamma_z_full"),
    ]
    paths = []
    for ax, (_, title, x_key, y_key) in zip(axes.ravel(), specs):
        x, y = _valid_xy(rows, x_key, y_key)
        ax.scatter(x, y, s=22, alpha=0.72, edgecolor="none")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("gold rate")
        ax.set_ylabel("BLT rate")
        ax.set_title(title)
        if len(x):
            lo = min(float(np.min(x)), float(np.min(y)))
            hi = max(float(np.max(x)), float(np.max(y)))
            ax.set_xlim(lo * 0.75, hi * 1.35)
            ax.set_ylim(lo * 0.75, hi * 1.35)
            add_identity_line(ax)
            med = float(np.median(np.abs(np.log(y / x))))
            ax.text(0.04, 0.94, f"median |log err| = {med:.3f}", transform=ax.transAxes, va="top")
    fig.tight_layout()
    paths.extend(savefig(fig, out_dir, "blt_vs_gold_rates"))
    return paths


def plot_blt_vs_gold_reward_eta(rows: Sequence[dict], out_dir: Path, eta_target: float) -> list[Path]:
    plt = _pyplot()

    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))
    specs = [
        ("eta_gold", "eta_lite", "BLT-lite eta"),
        ("eta_gold", "eta_full", "BLT-full eta"),
        ("reward_gold", "reward_lite", "BLT-lite reward"),
        ("reward_gold", "reward_full", "BLT-full reward"),
    ]
    for ax, (x_key, y_key, title) in zip(axes.ravel(), specs):
        x, y = _valid_xy(rows, x_key, y_key)
        if "eta" in title:
            ax.set_xscale("log")
            ax.set_yscale("log")
            add_target_eta_line(ax, eta_target)
        ax.scatter(x, y, s=22, alpha=0.72, edgecolor="none")
        ax.set_xlabel("gold")
        ax.set_ylabel("BLT")
        ax.set_title(title)
        if len(x):
            add_identity_line(ax)
    fig.tight_layout()
    return savefig(fig, out_dir, "blt_vs_gold_reward_eta")


def plot_error_histograms(rows: Sequence[dict], out_dir: Path) -> list[Path]:
    plt = _pyplot()

    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.1))
    metrics = [("gamma_x", "log error gamma_X"), ("gamma_z", "log error gamma_Z"), ("eta", "log error eta")]
    for ax, (metric, title) in zip(axes, metrics):
        for method, color in (("lite", "tab:blue"), ("full", "tab:orange")):
            errs = []
            for row in rows:
                gold = _finite_float(row, f"{metric}_gold")
                est = _finite_float(row, f"{metric}_{method}")
                if gold > 0 and est > 0 and np.isfinite(gold) and np.isfinite(est):
                    errs.append(np.log(est / gold))
            if errs:
                ax.hist(errs, bins=18, alpha=0.55, label=method, color=color)
        ax.axvline(0.0, color="0.25", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("log(est/gold)")
        ax.set_ylabel("count")
        ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, out_dir, "estimator_error_histograms")


def plot_estimator_cost(rows: Sequence[dict], out_dir: Path) -> list[Path]:
    plt = _pyplot()

    set_style()
    labels = ["gold", "BLT-lite", "BLT-full"]
    settings = [
        np.nanmedian([_finite_float(r, "settings_gold") for r in rows]),
        np.nanmedian([_finite_float(r, "settings_lite") for r in rows]),
        np.nanmedian([_finite_float(r, "settings_full") for r in rows]),
    ]
    waits = [
        np.nanmedian([_finite_float(r, "wait_gold") for r in rows]),
        np.nanmedian([_finite_float(r, "wait_lite") for r in rows]),
        np.nanmedian([_finite_float(r, "wait_full") for r in rows]),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(3)
    width = 0.35
    ax.bar(x - width / 2, settings, width, label="settings")
    ax.bar(x + width / 2, waits, width, label="physical wait-time")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("median cost per candidate")
    ax.legend(frameon=False)
    if np.isfinite(settings[1]) and settings[1] > 0:
        ax.annotate(f"{settings[0] / settings[1]:.1f}x fewer settings", xy=(1, settings[1]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
    if np.isfinite(waits[1]) and waits[1] > 0:
        ax.annotate(f"{waits[0] / waits[1]:.1f}x less wait", xy=(1 + width / 2, waits[1]), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8)
    fig.tight_layout()
    return savefig(fig, out_dir, "estimator_cost_comparison")


def plot_heatmap(
    rows: Sequence[dict],
    out_dir: Path,
    *,
    value_key: str,
    name: str,
    title: str,
    cmap: str = "viridis",
) -> list[Path]:
    plt = _pyplot()

    set_style()
    t0s = sorted({_finite_float(r, "t0") for r in rows if np.isfinite(_finite_float(r, "t0"))})
    t1s = sorted({_finite_float(r, "t1") for r in rows if np.isfinite(_finite_float(r, "t1"))})
    grid = np.full((len(t0s), len(t1s)), np.nan)
    grouped: dict[tuple[float, float], list[float]] = defaultdict(list)
    for row in rows:
        value = _finite_float(row, value_key)
        if np.isfinite(value):
            grouped[(_finite_float(row, "t0"), _finite_float(row, "t1"))].append(value)
    for i, t0 in enumerate(t0s):
        for j, t1 in enumerate(t1s):
            vals = grouped.get((t0, t1), [])
            if vals:
                grid[i, j] = float(np.nanmedian(vals))

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(t1s)))
    ax.set_xticklabels([f"{t:g}" for t in t1s])
    ax.set_yticks(np.arange(len(t0s)))
    ax.set_yticklabels([f"{t:g}" for t in t0s])
    ax.set_xlabel("t1")
    ax.set_ylabel("t0")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(value_key)
    fig.tight_layout()
    return savefig(fig, out_dir, name)


def plot_optimization_traces(rows: Sequence[dict], out_dir: Path, eta_target: float) -> list[Path]:
    plt = _pyplot()

    set_style()
    paths = []
    for x_key, y_key, name, ylabel in (
        ("cumulative_wait", "best_gold_reward", "optimization_reward_vs_wait_time", "best gold reward"),
        ("cumulative_settings", "best_gold_reward", "optimization_reward_vs_settings", "best gold reward"),
        ("cumulative_wait", "gold_eta", "optimization_eta_vs_wait_time", "gold eta"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 4.1))
        methods = sorted({str(r["method"]) for r in rows})
        for method in methods:
            method_rows = [r for r in rows if r["method"] == method]
            evals = sorted({int(float(r["eval_index"])) for r in method_rows})
            xs, means, sems = [], [], []
            for ev in evals:
                vals = [_finite_float(r, y_key) for r in method_rows if int(float(r["eval_index"])) == ev]
                xvals = [_finite_float(r, x_key) for r in method_rows if int(float(r["eval_index"])) == ev]
                vals = [v for v in vals if np.isfinite(v)]
                xvals = [v for v in xvals if np.isfinite(v)]
                if vals and xvals:
                    xs.append(float(np.mean(xvals)))
                    means.append(float(np.mean(vals)))
                    sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            if xs:
                xs_arr = np.asarray(xs)
                means_arr = np.asarray(means)
                sems_arr = np.asarray(sems)
                ax.plot(xs_arr, means_arr, linewidth=1.8, marker="o", markersize=3, label=method)
                ax.fill_between(xs_arr, means_arr - sems_arr, means_arr + sems_arr, alpha=0.16)
        if x_key == "cumulative_wait":
            ax.set_xscale("log")
        if y_key == "gold_eta":
            ax.set_yscale("log")
            add_target_eta_line(ax, eta_target)
        ax.set_xlabel("cumulative physical wait-time" if x_key == "cumulative_wait" else "cumulative settings")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False, ncol=2)
        fig.tight_layout()
        paths.extend(savefig(fig, out_dir, name))
    return paths


def plot_final_optimizer_figures(summary_rows: Sequence[dict], out_dir: Path, eta_target: float) -> list[Path]:
    plt = _pyplot()

    set_style()
    paths = []
    methods = sorted({str(r["method"]) for r in summary_rows})
    data = [[_finite_float(r, "final_gold_reward") for r in summary_rows if r["method"] == m] for m in methods]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.boxplot(data, labels=methods, showmeans=True)
    ax.set_ylabel("final gold reward")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    paths.extend(savefig(fig, out_dir, "final_reward_boxplot"))

    fig, ax = plt.subplots(figsize=(5.3, 4.4))
    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        tx = [_finite_float(r, "final_gold_T_X") for r in rows]
        tz = [_finite_float(r, "final_gold_T_Z") for r in rows]
        ax.scatter(tx, tz, s=30, alpha=0.75, label=method)
    all_tx = [_finite_float(r, "final_gold_T_X") for r in summary_rows if _finite_float(r, "final_gold_T_X") > 0]
    if all_tx:
        xs = np.logspace(np.log10(min(all_tx) * 0.8), np.log10(max(all_tx) * 1.25), 100)
        ax.plot(xs, eta_target * xs, linestyle="--", color="tab:red", linewidth=1.1, label="target bias")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("gold T_X")
    ax.set_ylabel("gold T_Z")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths.extend(savefig(fig, out_dir, "final_TX_TZ_pareto"))

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for method in methods:
        rows = [r for r in summary_rows if r["method"] == method]
        ax.scatter(
            [_finite_float(r, "total_wait") for r in rows],
            [_finite_float(r, "final_gold_reward") for r in rows],
            s=30,
            alpha=0.75,
            label=method,
        )
    ax.set_xscale("log")
    ax.set_xlabel("total physical wait-time cost")
    ax.set_ylabel("final gold reward")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    paths.extend(savefig(fig, out_dir, "final_cost_vs_reward"))
    return paths


def plot_finite_shot_figures(estimator_rows: Sequence[dict], opt_rows: Sequence[dict], out_dir: Path) -> list[Path]:
    plt = _pyplot()

    set_style()
    paths = []
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    shot_labels = sorted({str(r["n_shots"]) for r in estimator_rows}, key=lambda s: float("inf") if s == "None" else float(s))
    for method, color in (("lite", "tab:blue"), ("full", "tab:orange")):
        centers, medians = [], []
        for label in shot_labels:
            vals = []
            for row in estimator_rows:
                if str(row["n_shots"]) == label:
                    gx = _finite_float(row, "gamma_x_gold")
                    ex = _finite_float(row, f"gamma_x_{method}")
                    gz = _finite_float(row, "gamma_z_gold")
                    ez = _finite_float(row, f"gamma_z_{method}")
                    if gx > 0 and ex > 0 and gz > 0 and ez > 0:
                        vals.append(0.5 * (abs(np.log(ex / gx)) + abs(np.log(ez / gz))))
            centers.append(len(centers))
            medians.append(float(np.nanmedian(vals)) if vals else np.nan)
        ax.plot(centers, medians, marker="o", color=color, label=method)
    ax.set_xticks(range(len(shot_labels)))
    ax.set_xticklabels(shot_labels)
    ax.set_xlabel("n_shots")
    ax.set_ylabel("median rate log-error")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths.extend(savefig(fig, out_dir, "finite_shot_estimator_error"))

    if opt_rows:
        methods = sorted({str(r["method"]) for r in opt_rows})
        labels = sorted({str(r["n_shots"]) for r in opt_rows}, key=lambda s: float("inf") if s == "None" else float(s))
        fig, ax = plt.subplots(figsize=(6.6, 3.8))
        width = 0.8 / max(1, len(methods))
        x = np.arange(len(labels))
        for i, method in enumerate(methods):
            means = []
            sems = []
            for label in labels:
                vals = [_finite_float(r, "final_gold_reward") for r in opt_rows if r["method"] == method and str(r["n_shots"]) == label]
                vals = [v for v in vals if np.isfinite(v)]
                means.append(float(np.mean(vals)) if vals else np.nan)
                sems.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, means, width, yerr=sems, label=method)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("n_shots")
        ax.set_ylabel("final gold reward")
        ax.legend(frameon=False)
        fig.tight_layout()
        paths.extend(savefig(fig, out_dir, "finite_shot_optimization_reward"))
    return paths
