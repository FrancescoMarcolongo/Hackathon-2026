"""Noisy raw-vs-physical coordinate comparison.

Noise is added to the simulated decay samples before the robust exponential
fit used to extract Tx and Tz.  The optimizer sees the noisy fitted loss, while
the same candidates are also logged with clean fitted metrics for resilience
diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
ORIGINAL_PROJECT = ROOT.parent / "team-core-bias-optimizer_sannino"
CACHE_ROOT = ROOT / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
(CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
from cmaes import SepCMA

from cat_model import (
    SimulationConfig,
    _fit_penalty,
    measure_lifetimes,
    params_to_complex,
    robust_exp_fit,
    truncation_penalty,
)
from physical_coordinates import (
    physical_bounds_from_raw_reference,
    physical_diagnostics_from_raw,
    physical_to_raw,
    raw_to_physical,
)
from rewards import RewardConfig, compute_reward
from run_core_bias_optimization import BOUNDS as RAW_BOUNDS
from run_core_bias_optimization import select_better

PROJECT_VERSION = "physical-coordinate-baseline-v1-noise-resilience-robust-reward"
U0 = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
DEFAULT_POPULATION_SIZE = 4
DEFAULT_N_GENERATIONS = 3
SIGMA0_RAW = 0.25
SIGMA0_PHYSICAL = 0.35
DEFAULT_SEED = 2026
COORDINATE_MODES = ("raw", "physical", "robust")


@dataclass(frozen=True)
class RobustRewardConfig:
    """Reward used only by the robust third training mode."""

    target_bias: float = 100.0
    bias_tol_rel: float = 0.03
    w_lifetime: float = 0.5
    w_bias_exact: float = 95.0
    w_fit: float = 2.0
    feasibility_bonus: float = 12.0
    min_tx: float = 0.05
    min_tz: float = 5.0
    floor_weight: float = 12.0
    alpha_target: float = 2.0
    w_alpha: float = 0.35
    nbar_min: float = 2.0
    nbar_max: float = 8.0
    w_nbar: float = 0.18
    w_uncertainty: float = 35.0
    uncertainty_scale: float = 4.0


ROBUST_REWARD_CFG = RobustRewardConfig()


def create_run_dir(base_dir: str | Path = "results", label: str | None = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [f"run_{timestamp}"]
    if label:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_.-")
        if safe:
            parts.append(safe[:60])
    run_dir = base / "_".join(parts)
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{'_'.join(parts)}_{suffix:02d}"
        suffix += 1
    run_dir.mkdir(parents=True)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=DEFAULT_N_GENERATIONS)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noise-std", type=float, default=0.02)
    parser.add_argument("--quick", action="store_true", help="Use fewer fit points for a faster smoke run.")
    return parser.parse_args()


def _finite_positive(*values: float) -> bool:
    return all(np.isfinite(value) and value > 0.0 for value in values)


def metric_uncertainty(metrics: dict[str, Any], robust_cfg: RobustRewardConfig = ROBUST_REWARD_CFG) -> float:
    """Small scalar uncertainty proxy from noise and fit diagnostics."""

    fit_x_rmse = float(metrics.get("fit_x_rmse", 0.0))
    fit_z_rmse = float(metrics.get("fit_z_rmse", 0.0))
    fit_x_r2 = float(metrics.get("fit_x_r2", 1.0))
    fit_z_r2 = float(metrics.get("fit_z_r2", 1.0))
    noise_std = float(metrics.get("noise_std", 0.0))
    r2_gap = max(0.0, 0.995 - fit_x_r2) + max(0.0, 0.995 - fit_z_r2)
    raw = noise_std + fit_x_rmse + fit_z_rmse + r2_gap
    return float(max(0.0, robust_cfg.uncertainty_scale * raw))


def compute_robust_reward(
    metrics: dict[str, Any],
    cfg: RobustRewardConfig = ROBUST_REWARD_CFG,
) -> dict[str, float | bool]:
    """Alpha/nbar-aware reward with conservative penalties for noisy fits.

    This reward is intentionally used only by the third ``robust`` optimizer.
    Raw and physical modes continue to use the original baseline reward.
    """

    Tx = float(metrics.get("T_X", np.nan))
    Tz = float(metrics.get("T_Z", np.nan))
    bias = float(metrics.get("bias", np.nan))
    alpha_abs = float(metrics.get("alpha_abs", np.nan))
    nbar = float(metrics.get("nbar", alpha_abs**2 if np.isfinite(alpha_abs) else np.nan))
    fit_penalty = float(metrics.get("fit_penalty", 1.0e3))
    valid = bool(metrics.get("valid", False))

    if not _finite_positive(Tx, Tz, bias) or not np.isfinite(alpha_abs) or not np.isfinite(nbar):
        return {
            "reward": -1.0e9,
            "loss_to_minimize": 1.0e9,
            "is_feasible": False,
            "bias_error": np.inf,
            "bias_rel_error": np.inf,
            "lifetime_score": -np.inf,
            "floor_penalty": 1.0e6,
            "alpha_penalty": 1.0e6,
            "nbar_penalty": 1.0e6,
            "uncertainty": np.inf,
            "uncertainty_penalty": 1.0e6,
        }

    uncertainty = metric_uncertainty(metrics, cfg)
    log_eta = math.log(bias)
    log_target = math.log(cfg.target_bias)
    bias_error = abs(log_eta - log_target)
    # Make target matching conservative under uncertain fits.
    robust_bias_error = bias_error + uncertainty
    bias_rel_error = abs(bias / cfg.target_bias - 1.0)
    lifetime_score = 0.5 * (math.log(Tx) + math.log(Tz))
    floor_penalty = cfg.floor_weight * (
        max(0.0, math.log(cfg.min_tx / Tx)) ** 2
        + max(0.0, math.log(cfg.min_tz / Tz)) ** 2
    )
    alpha_penalty = cfg.w_alpha * (alpha_abs - cfg.alpha_target) ** 2
    nbar_penalty = cfg.w_nbar * (
        max(0.0, cfg.nbar_min - nbar) ** 2 + max(0.0, nbar - cfg.nbar_max) ** 2
    )
    uncertainty_penalty = cfg.w_uncertainty * uncertainty**2
    is_within_band = bool(bias_rel_error <= cfg.bias_tol_rel)
    is_feasible = bool(valid and is_within_band and uncertainty < 0.25)

    reward = (
        cfg.feasibility_bonus * float(is_within_band)
        + cfg.w_lifetime * lifetime_score
        - cfg.w_bias_exact * robust_bias_error**2
        - cfg.w_fit * fit_penalty
        - floor_penalty
        - alpha_penalty
        - nbar_penalty
        - uncertainty_penalty
    )
    return {
        "reward": float(reward),
        "loss_to_minimize": -float(reward),
        "is_feasible": is_feasible,
        "bias_error": float(bias_error),
        "bias_rel_error": float(bias_rel_error),
        "lifetime_score": float(lifetime_score),
        "floor_penalty": float(floor_penalty),
        "alpha_penalty": float(alpha_penalty),
        "nbar_penalty": float(nbar_penalty),
        "uncertainty": float(uncertainty),
        "uncertainty_penalty": float(uncertainty_penalty),
    }


def objective_reward(
    metrics: dict[str, Any],
    reward_cfg: RewardConfig,
    coordinate_mode: str,
) -> dict[str, float | bool]:
    if coordinate_mode == "robust":
        return compute_robust_reward(metrics)
    return compute_reward(metrics, reward_cfg)


def clean_reward_metrics(
    clean_metrics: dict[str, Any],
    reward_cfg: RewardConfig,
    coordinate_mode: str,
) -> dict[str, float | bool]:
    metrics = {k: v for k, v in clean_metrics.items() if k != "curves"}
    reward = objective_reward(metrics, reward_cfg, coordinate_mode)
    baseline_reward = compute_reward(metrics, reward_cfg)
    return {
        "clean_loss": float(reward["loss_to_minimize"]),
        "clean_reward": float(reward["reward"]),
        "baseline_clean_loss": float(baseline_reward["loss_to_minimize"]),
        "baseline_clean_reward": float(baseline_reward["reward"]),
        "clean_Tx": float(metrics.get("T_X", np.nan)),
        "clean_Tz": float(metrics.get("T_Z", np.nan)),
        "clean_bias": float(metrics.get("bias", np.nan)),
        "clean_geo_lifetime": float(metrics.get("geo_lifetime", np.nan)),
        "clean_fit_penalty": float(metrics.get("fit_penalty", np.nan)),
        "clean_valid": bool(metrics.get("valid", False)),
        "clean_uncertainty": float(metric_uncertainty(metrics)),
        "clean_alpha_abs": float(metrics.get("alpha_abs", np.nan)),
        "clean_nbar": float(metrics.get("nbar", np.nan)),
    }


def noisy_measure_lifetimes(
    g2: complex,
    eps_d: complex,
    cfg: SimulationConfig,
    *,
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return noisy fitted metrics and clean fitted metrics for the same curves."""

    clean = measure_lifetimes(g2, eps_d, cfg, use_cache=True, return_curves=True)
    if "curves" not in clean:
        return {k: v for k, v in clean.items() if k != "curves"}, clean

    curves = clean["curves"]
    times_x = np.asarray(curves["times_x"], dtype=float)
    times_z = np.asarray(curves["times_z"], dtype=float)
    clean_x = np.asarray(curves["values_x"], dtype=float)
    clean_z = np.asarray(curves["values_z"], dtype=float)

    noisy_x = np.clip(clean_x + rng.normal(0.0, noise_std, size=clean_x.shape), -1.25, 1.25)
    noisy_z = np.clip(clean_z + rng.normal(0.0, noise_std, size=clean_z.shape), -1.25, 1.25)

    fit_x = robust_exp_fit(
        times_x,
        noisy_x,
        max_tau=max(100.0, cfg.max_tau_factor * float(cfg.t_final_x)),
    )
    fit_z = robust_exp_fit(
        times_z,
        noisy_z,
        max_tau=max(1000.0, cfg.max_tau_factor * float(cfg.t_final_z)),
    )

    T_X = float(fit_x["params"][1])
    T_Z = float(fit_z["params"][1])
    finite = bool(np.isfinite(T_X) and np.isfinite(T_Z) and T_X > 0 and T_Z > 0)
    bias = float(T_Z / T_X) if finite else np.nan
    alpha_abs = float(clean.get("alpha_abs", np.nan))
    alpha_penalty = truncation_penalty(alpha_abs, cfg.na, margin=cfg.alpha_margin)
    fit_penalty = _fit_penalty(fit_x, fit_z) + alpha_penalty

    noisy = {
        "g2_real": float(g2.real),
        "g2_imag": float(g2.imag),
        "eps_d_real": float(eps_d.real),
        "eps_d_imag": float(eps_d.imag),
        "T_X": T_X,
        "T_Z": T_Z,
        "bias": bias,
        "geo_lifetime": float(math.sqrt(T_X * T_Z)) if finite else np.nan,
        "alpha_abs": alpha_abs,
        "nbar": float(alpha_abs**2) if np.isfinite(alpha_abs) else np.nan,
        "fit_ok": bool(fit_x["success"] and fit_z["success"]),
        "fit_penalty": float(fit_penalty),
        "fit_x_r2": float(fit_x["r2"]),
        "fit_z_r2": float(fit_z["r2"]),
        "fit_x_rmse": float(fit_x["rmse"]),
        "fit_z_rmse": float(fit_z["rmse"]),
        "fit_x_hit_tau_bound": bool(fit_x["hit_tau_bound"]),
        "fit_z_hit_tau_bound": bool(fit_z["hit_tau_bound"]),
        "valid": bool(fit_x["success"] and fit_z["success"] and finite and fit_penalty < 10.0),
        "reason": "" if bool(fit_x["success"] and fit_z["success"]) else "noisy fit failed",
        "noise_std": float(noise_std),
        "noise_x_rms": float(np.sqrt(np.mean((noisy_x - clean_x) ** 2))),
        "noise_z_rms": float(np.sqrt(np.mean((noisy_z - clean_z) ** 2))),
    }
    return noisy, clean


def evaluate_noisy_candidate(
    u: np.ndarray,
    *,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    coordinate_mode: str,
    optimizer_coordinate: np.ndarray,
    generation: int,
    candidate_index: int,
    evaluation: int,
    rng: np.random.Generator,
    noise_std: float,
) -> dict[str, Any]:
    u = np.asarray(u, dtype=float)
    g2, eps_d = params_to_complex(u)
    noisy_metrics, clean_metrics = noisy_measure_lifetimes(
        g2,
        eps_d,
        sim_cfg,
        rng=rng,
        noise_std=noise_std,
    )
    reward = objective_reward(noisy_metrics, reward_cfg, coordinate_mode)

    if coordinate_mode in ("physical", "robust"):
        v = np.asarray(optimizer_coordinate, dtype=float)
    else:
        v = raw_to_physical(u, kappa_b=sim_cfg.kappa_b)
    baseline_noisy_reward = compute_reward(noisy_metrics, reward_cfg)

    row: dict[str, Any] = {
        "evaluation": int(evaluation),
        "generation": int(generation),
        "candidate_index": int(candidate_index),
        "coordinate_mode": coordinate_mode,
        "objective": "robust_alpha_nbar_uncertainty" if coordinate_mode == "robust" else "baseline",
        "loss": float(reward["loss_to_minimize"]),
        "reward": float(reward["reward"]),
        "baseline_loss": float(baseline_noisy_reward["loss_to_minimize"]),
        "baseline_reward": float(baseline_noisy_reward["reward"]),
        "Tx": float(noisy_metrics.get("T_X", np.nan)),
        "Tz": float(noisy_metrics.get("T_Z", np.nan)),
        "bias": float(noisy_metrics.get("bias", np.nan)),
        "geo_lifetime": float(noisy_metrics.get("geo_lifetime", np.nan)),
        "alpha_abs": float(noisy_metrics.get("alpha_abs", np.nan)),
        "nbar": float(noisy_metrics.get("nbar", np.nan)),
        "fit_penalty": float(noisy_metrics.get("fit_penalty", np.nan)),
        "fit_x_r2": float(noisy_metrics.get("fit_x_r2", np.nan)),
        "fit_z_r2": float(noisy_metrics.get("fit_z_r2", np.nan)),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_error": float(reward.get("bias_error", np.nan)),
        "alpha_penalty": float(reward.get("alpha_penalty", 0.0)),
        "nbar_penalty": float(reward.get("nbar_penalty", 0.0)),
        "uncertainty": float(reward.get("uncertainty", metric_uncertainty(noisy_metrics))),
        "uncertainty_penalty": float(reward.get("uncertainty_penalty", 0.0)),
        "noise_std": float(noise_std),
        "noise_x_rms": float(noisy_metrics.get("noise_x_rms", np.nan)),
        "noise_z_rms": float(noisy_metrics.get("noise_z_rms", np.nan)),
        "v0_log_kappa2": float(v[0]),
        "v1_log_abs_alpha": float(v[1]),
        "v2_arg_alpha": float(v[2]),
        "v3_phi_mismatch": float(v[3]),
        "u0_re_g2": float(u[0]),
        "u1_im_g2": float(u[1]),
        "u2_re_eps_d": float(u[2]),
        "u3_im_eps_d": float(u[3]),
    }
    diagnostics = physical_diagnostics_from_raw(u, kappa_b=sim_cfg.kappa_b)
    diagnostics["physical_alpha_abs"] = float(diagnostics.get("alpha_abs", np.nan))
    diagnostics["physical_alpha_phase"] = float(diagnostics.get("alpha_phase", np.nan))
    diagnostics.pop("alpha_abs", None)
    diagnostics.pop("alpha_phase", None)
    row.update(clean_reward_metrics(clean_metrics, reward_cfg, coordinate_mode))
    row.update(diagnostics)
    return row


def row_to_optimizer_coord(row: dict[str, Any], coordinate_mode: str) -> np.ndarray:
    if coordinate_mode in ("physical", "robust"):
        return np.array(
            [row["v0_log_kappa2"], row["v1_log_abs_alpha"], row["v2_arg_alpha"], row["v3_phi_mismatch"]],
            dtype=float,
        )
    return np.array([row["u0_re_g2"], row["u1_im_g2"], row["u2_re_eps_d"], row["u3_im_eps_d"]], dtype=float)


def run_mode(
    coordinate_mode: str,
    *,
    args: argparse.Namespace,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    bounds_v: np.ndarray,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if coordinate_mode == "raw":
        mean0 = U0.copy()
        bounds = RAW_BOUNDS
        sigma0 = SIGMA0_RAW
    elif coordinate_mode in ("physical", "robust"):
        mean0 = raw_to_physical(U0, kappa_b=sim_cfg.kappa_b)
        bounds = bounds_v
        sigma0 = SIGMA0_PHYSICAL
    else:
        raise ValueError(f"coordinate_mode must be one of {COORDINATE_MODES}")

    optimizer = SepCMA(
        mean=mean0.copy(),
        sigma=float(sigma0),
        bounds=np.asarray(bounds, dtype=float),
        population_size=int(args.population),
        seed=int(args.seed),
    )

    rows: list[dict[str, Any]] = []
    incumbent: dict[str, Any] | None = None
    evaluation = 0
    for generation in range(int(args.generations)):
        candidates = [np.asarray(optimizer.ask(), dtype=float) for _ in range(optimizer.population_size)]
        if len(candidates) >= 1:
            candidates[0] = row_to_optimizer_coord(incumbent, coordinate_mode) if incumbent else mean0.copy()
        if len(candidates) >= 2:
            candidates[1] = mean0.copy()

        evaluated: list[dict[str, Any]] = []
        for candidate_index, coord in enumerate(candidates):
            raw_u = physical_to_raw(coord, kappa_b=sim_cfg.kappa_b) if coordinate_mode in ("physical", "robust") else coord
            row = evaluate_noisy_candidate(
                raw_u,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                coordinate_mode=coordinate_mode,
                optimizer_coordinate=coord,
                generation=generation,
                candidate_index=candidate_index,
                evaluation=evaluation,
                rng=rng,
                noise_std=float(args.noise_std),
            )
            evaluation += 1
            rows.append(row)
            evaluated.append(row)
            if incumbent is None:
                incumbent = dict(row)
            elif coordinate_mode == "robust":
                if float(row["loss"]) < float(incumbent["loss"]):
                    incumbent = dict(row)
            elif select_better(row, incumbent, reward_cfg):
                incumbent = dict(row)

        optimizer.tell([(row_to_optimizer_coord(row, coordinate_mode), float(row["loss"])) for row in evaluated])
        noisy_best = min(rows, key=lambda item: float(item["loss"]))
        clean_best = min(rows, key=lambda item: float(item["clean_loss"]))
        print(
            f"{coordinate_mode} generation={generation} "
            f"noisy_best_loss={noisy_best['loss']:.4g} clean_best_loss={clean_best['clean_loss']:.4g} "
            f"clean_bias={clean_best['clean_bias']:.4g}",
            flush=True,
        )
    return rows


def cumulative_best(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    best: dict[str, Any] | None = None
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["evaluation"])):
        if best is None or float(row[key]) < float(best[key]):
            best = row
        out.append(best)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def epoch_best_rows(rows: list[dict[str, Any]], selector_key: str) -> list[dict[str, Any]]:
    """Return one row per mode/epoch, selected by the requested loss key."""

    out: list[dict[str, Any]] = []
    for mode in COORDINATE_MODES:
        mode_rows = [row for row in rows if row["coordinate_mode"] == mode]
        epochs = sorted({int(row["generation"]) for row in mode_rows})
        for epoch in epochs:
            epoch_rows = [row for row in mode_rows if int(row["generation"]) == epoch]
            if epoch_rows:
                out.append(min(epoch_rows, key=lambda row: float(row[selector_key])))
    return out


def plot_metric_along_best(
    rows: list[dict[str, Any]],
    run_dir: Path,
    filename: str,
    metric_key: str,
    selector_key: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for mode in COORDINATE_MODES:
        mode_rows = [row for row in rows if row["coordinate_mode"] == mode]
        best_rows = cumulative_best(mode_rows, selector_key)
        ax.plot(
            [row["evaluation"] for row in best_rows],
            [row[metric_key] for row in best_rows],
            marker="o",
            lw=1.5,
            label=mode,
        )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_epoch_metric(
    epoch_rows: list[dict[str, Any]],
    run_dir: Path,
    filename: str,
    metric_key: str,
    ylabel: str,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for mode in COORDINATE_MODES:
        mode_rows = sorted(
            [row for row in epoch_rows if row["coordinate_mode"] == mode],
            key=lambda row: int(row["generation"]),
        )
        ax.plot(
            [int(row["generation"]) for row in mode_rows],
            [float(row[metric_key]) for row in mode_rows],
            marker="o",
            lw=1.5,
            label=mode,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = run_dir / filename
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_plots(rows: list[dict[str, Any]], run_dir: Path) -> list[Path]:
    clean_epoch_rows = epoch_best_rows(rows, "clean_loss")
    noisy_epoch_rows = epoch_best_rows(rows, "loss")
    paths = [
        plot_metric_along_best(rows, run_dir, "noisy_objective_loss_vs_evaluation_all_modes.png", "loss", "loss", "Noisy objective loss"),
        plot_metric_along_best(rows, run_dir, "clean_objective_loss_vs_evaluation_all_modes.png", "clean_loss", "clean_loss", "Clean objective loss"),
        plot_metric_along_best(rows, run_dir, "clean_baseline_loss_vs_evaluation_all_modes.png", "baseline_clean_loss", "baseline_clean_loss", "Clean baseline-validation loss"),
        plot_metric_along_best(rows, run_dir, "noisy_bias_vs_evaluation_all_modes.png", "bias", "loss", "Noisy fitted bias of best objective candidate"),
        plot_metric_along_best(rows, run_dir, "clean_bias_vs_evaluation_all_modes.png", "clean_bias", "clean_loss", "Clean fitted bias of best clean-objective candidate"),
        plot_metric_along_best(rows, run_dir, "clean_Tx_vs_evaluation_all_modes.png", "clean_Tx", "clean_loss", "Clean fitted Tx of best clean-objective candidate"),
        plot_metric_along_best(rows, run_dir, "clean_Tz_vs_evaluation_all_modes.png", "clean_Tz", "clean_loss", "Clean fitted Tz of best clean-objective candidate"),
        plot_metric_along_best(rows, run_dir, "alpha_vs_evaluation_all_modes.png", "clean_alpha_abs", "clean_loss", "Clean |alpha| of best clean-objective candidate"),
        plot_metric_along_best(rows, run_dir, "nbar_vs_evaluation_all_modes.png", "clean_nbar", "clean_loss", "Clean nbar of best clean-objective candidate"),
        plot_epoch_metric(clean_epoch_rows, run_dir, "epoch_clean_objective_loss_all_modes.png", "clean_loss", "Epoch-best clean objective loss"),
        plot_epoch_metric(clean_epoch_rows, run_dir, "epoch_clean_baseline_loss_all_modes.png", "baseline_clean_loss", "Epoch-best clean baseline-validation loss"),
        plot_epoch_metric(clean_epoch_rows, run_dir, "epoch_clean_bias_all_modes.png", "clean_bias", "Clean fitted bias of epoch-best clean-objective candidate"),
        plot_epoch_metric(clean_epoch_rows, run_dir, "epoch_clean_Tx_all_modes.png", "clean_Tx", "Clean fitted Tx of epoch-best clean-objective candidate"),
        plot_epoch_metric(clean_epoch_rows, run_dir, "epoch_clean_Tz_all_modes.png", "clean_Tz", "Clean fitted Tz of epoch-best clean-objective candidate"),
        plot_epoch_metric(noisy_epoch_rows, run_dir, "epoch_noisy_objective_loss_all_modes.png", "loss", "Epoch-best noisy objective loss"),
        plot_epoch_metric(noisy_epoch_rows, run_dir, "epoch_noisy_bias_all_modes.png", "bias", "Noisy fitted bias of epoch-best objective candidate"),
    ]
    write_csv(run_dir / "epoch_best_clean_history.csv", clean_epoch_rows)
    write_csv(run_dir / "epoch_best_noisy_history.csv", noisy_epoch_rows)

    physical_rows = [row for row in rows if row["coordinate_mode"] in ("physical", "robust")]
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if physical_rows:
        sc = ax.scatter(
            [row["kappa_2"] for row in physical_rows],
            [row["physical_alpha_abs"] for row in physical_rows],
            c=[row["clean_loss"] for row in physical_rows],
            cmap="viridis_r",
            s=45,
        )
        fig.colorbar(sc, ax=ax, label="clean loss")
    ax.set_xlabel("kappa_2")
    ax.set_ylabel("|alpha|")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = run_dir / "physical_samples_alpha_kappa2_noisy.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> None:
    args = parse_args()
    sim_cfg = SimulationConfig(n_points=35) if args.quick else SimulationConfig()
    reward_cfg = RewardConfig()
    lower_v, upper_v, physical_bounds = physical_bounds_from_raw_reference(U0, kappa_b=sim_cfg.kappa_b)
    bounds_v = np.column_stack([lower_v, upper_v])
    run_dir = create_run_dir(ROOT / "results", label="noise_resilience_raw_physical_robust")

    rng_raw = np.random.default_rng(int(args.seed) + 1000)
    rng_physical = np.random.default_rng(int(args.seed) + 2000)
    rng_robust = np.random.default_rng(int(args.seed) + 3000)
    raw_rows = run_mode("raw", args=args, sim_cfg=sim_cfg, reward_cfg=reward_cfg, bounds_v=bounds_v, rng=rng_raw)
    physical_rows = run_mode("physical", args=args, sim_cfg=sim_cfg, reward_cfg=reward_cfg, bounds_v=bounds_v, rng=rng_physical)
    robust_rows = run_mode("robust", args=args, sim_cfg=sim_cfg, reward_cfg=reward_cfg, bounds_v=bounds_v, rng=rng_robust)
    rows = raw_rows + physical_rows + robust_rows

    write_csv(run_dir / "candidate_history.csv", rows)
    plot_paths = save_plots(rows, run_dir)

    best_raw_noisy = min(raw_rows, key=lambda row: float(row["loss"]))
    best_physical_noisy = min(physical_rows, key=lambda row: float(row["loss"]))
    best_robust_noisy = min(robust_rows, key=lambda row: float(row["loss"]))
    best_raw_clean = min(raw_rows, key=lambda row: float(row["clean_loss"]))
    best_physical_clean = min(physical_rows, key=lambda row: float(row["clean_loss"]))
    best_robust_clean = min(robust_rows, key=lambda row: float(row["clean_loss"]))
    best_raw_baseline_clean = min(raw_rows, key=lambda row: float(row["baseline_clean_loss"]))
    best_physical_baseline_clean = min(physical_rows, key=lambda row: float(row["baseline_clean_loss"]))
    best_robust_baseline_clean = min(robust_rows, key=lambda row: float(row["baseline_clean_loss"]))
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "project_version": PROJECT_VERSION,
        "original_project_compared": str(ORIGINAL_PROJECT),
        "physical_project_compared": str(ROOT),
        "noise_model": {
            "type": "additive Gaussian on decay samples before exponential fit",
            "noise_std": float(args.noise_std),
            "clip_range": [-1.25, 1.25],
        },
        "random_seed": int(args.seed),
        "population_size": int(args.population),
        "n_generations": int(args.generations),
        "simulation_config": vars(sim_cfg),
        "physical_bounds": physical_bounds,
        "baseline_reward_config": vars(reward_cfg),
        "robust_reward_config": vars(ROBUST_REWARD_CFG),
        "coordinate_modes_compared": list(COORDINATE_MODES),
        "best_raw_noisy_objective": best_raw_noisy,
        "best_physical_noisy_objective": best_physical_noisy,
        "best_robust_noisy_objective": best_robust_noisy,
        "best_raw_clean_validation": best_raw_clean,
        "best_physical_clean_validation": best_physical_clean,
        "best_robust_clean_validation": best_robust_clean,
        "best_raw_baseline_clean_validation": best_raw_baseline_clean,
        "best_physical_baseline_clean_validation": best_physical_baseline_clean,
        "best_robust_baseline_clean_validation": best_robust_baseline_clean,
        "improvement_noisy_loss": float(best_raw_noisy["loss"] - best_physical_noisy["loss"]),
        "improvement_clean_loss": float(best_raw_clean["clean_loss"] - best_physical_clean["clean_loss"]),
        "robust_vs_raw_baseline_clean_loss": float(
            best_raw_baseline_clean["baseline_clean_loss"] - best_robust_baseline_clean["baseline_clean_loss"]
        ),
        "robust_vs_physical_baseline_clean_loss": float(
            best_physical_baseline_clean["baseline_clean_loss"]
            - best_robust_baseline_clean["baseline_clean_loss"]
        ),
        "notes": [
            "Raw and physical modes use the unchanged baseline reward; robust mode uses a separate alpha/nbar/uncertainty-aware reward.",
            "Noisy loss drives optimization; clean loss is logged for resilience validation on the same evaluated candidates.",
            "Baseline clean validation loss is also logged for all modes because robust objective loss is not directly comparable to baseline objective loss.",
        ],
        "plots": [str(path) for path in plot_paths],
    }
    with (run_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(
                {
                    "noise_std": float(args.noise_std),
                    "seed": int(args.seed),
                    "population_size": int(args.population),
                    "n_generations": int(args.generations),
                    "simulation_config": vars(sim_cfg),
                    "physical_bounds": physical_bounds,
                    "coordinate_modes": list(COORDINATE_MODES),
                    "baseline_reward_config": vars(reward_cfg),
                    "robust_reward_config": vars(ROBUST_REWARD_CFG),
                }
            ),
            f,
            indent=2,
            sort_keys=True,
        )
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, indent=2, sort_keys=True)
    print(f"Saved noisy comparison to: {run_dir}")


if __name__ == "__main__":
    main()
