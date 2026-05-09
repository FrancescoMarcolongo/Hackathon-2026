"""Validation pipeline for Boundary Liouvillian Tracking.

Examples:
    python team-blt/validation.py --simulator-check
    python team-blt/validation.py --estimator-scan --n-points 80 --mode medium
    python team-blt/validation.py --all --mode quick --n-points 15 --seeds 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "outputs_validation"
_CACHE_ROOT = OUT_ROOT / ".cache"
_MPL_ROOT = OUT_ROOT / ".mplconfig"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_ROOT))

from cat_env import (
    MEAS_AXES,
    PREP_STATES,
    CatLab,
    CatSystemConfig,
    default_q_bounds,
    physical_summary,
    q_to_knobs,
)
from estimators import (
    EstimateResult,
    RewardConfig,
    blt_full_estimate,
    blt_lite_estimate,
    gold_full_fit_details,
    gold_full_fit_estimate,
    spectral_reward,
)
from figures import (
    plot_blt_vs_gold_rates,
    plot_blt_vs_gold_reward_eta,
    plot_decay_fit,
    plot_error_histograms,
    plot_estimator_cost,
    plot_final_optimizer_figures,
    plot_finite_shot_figures,
    plot_heatmap,
    plot_optimization_traces,
)


@dataclass(frozen=True)
class ModeConfig:
    na: int
    nb: int
    kappa_b: float
    kappa_a: float
    k_points: int
    t_final_x: float
    t_final_z: float
    t0: float
    t1: float
    optimizer_evals: int
    optimizer_wait_budget: float
    time_window_subset: int
    finite_shot_points: int


MODE_CONFIGS = {
    "quick": ModeConfig(
        na=10,
        nb=3,
        kappa_b=8.0,
        kappa_a=0.15,
        k_points=30,
        t_final_x=12.0,
        t_final_z=100.0,
        t0=1.0,
        t1=6.0,
        optimizer_evals=3,
        optimizer_wait_budget=1200.0,
        time_window_subset=4,
        finite_shot_points=8,
    ),
    "medium": ModeConfig(
        na=10,
        nb=3,
        kappa_b=8.0,
        kappa_a=0.15,
        k_points=70,
        t_final_x=14.0,
        t_final_z=140.0,
        t0=1.0,
        t1=8.0,
        optimizer_evals=10,
        optimizer_wait_budget=18000.0,
        time_window_subset=18,
        finite_shot_points=24,
    ),
    "full": ModeConfig(
        na=12,
        nb=4,
        kappa_b=10.0,
        kappa_a=0.12,
        k_points=100,
        t_final_x=16.0,
        t_final_z=180.0,
        t0=1.0,
        t1=8.0,
        optimizer_evals=14,
        optimizer_wait_budget=30000.0,
        time_window_subset=32,
        finite_shot_points=40,
    ),
}


SIM_CHECK_FIELDS = ["mode", "check", "case", "value", "threshold", "pass", "notes"]
SCAN_FIELDS = [
    "mode",
    "point_id",
    "n_shots",
    "z_observable",
    "q0",
    "q1",
    "q2",
    "q3",
    "kappa_2",
    "nbar",
    "alpha_real",
    "alpha_imag",
    "reward_gold",
    "gamma_x_gold",
    "gamma_z_gold",
    "T_X_gold",
    "T_Z_gold",
    "eta_gold",
    "valid_gold",
    "settings_gold",
    "wait_gold",
    "fit_x_r2",
    "fit_z_r2",
    "fit_x_rmse",
    "fit_z_rmse",
    "reward_challenge",
    "gamma_x_challenge",
    "gamma_z_challenge",
    "eta_challenge",
    "valid_challenge",
    "reward_lite",
    "gamma_x_lite",
    "gamma_z_lite",
    "T_X_lite",
    "T_Z_lite",
    "eta_lite",
    "valid_lite",
    "settings_lite",
    "wait_lite",
    "lite_reason",
    "reward_full",
    "gamma_x_full",
    "gamma_z_full",
    "T_X_full",
    "T_Z_full",
    "eta_full",
    "valid_full",
    "settings_full",
    "wait_full",
    "full_reason",
    "gamma_x_full_eigen",
    "gamma_z_full_eigen",
    "gamma_x_full_contrast",
    "gamma_z_full_contrast",
]
SUMMARY_FIELDS = [
    "mode",
    "n_rows",
    "estimator",
    "median_abs_logerr_gamma_x",
    "median_abs_logerr_gamma_z",
    "median_abs_logerr_eta",
    "median_reward_error",
    "spearman_reward",
    "pearson_log_gamma_x",
    "pearson_log_gamma_z",
    "failure_rate",
    "settings_speedup",
    "wait_speedup",
]
TIME_WINDOW_FIELDS = [
    "mode",
    "point_id",
    "t0",
    "t1",
    "reward_gold",
    "gamma_x_gold",
    "gamma_z_gold",
    "reward_lite",
    "gamma_x_lite",
    "gamma_z_lite",
    "eta_lite",
    "valid_lite",
    "lite_logerr_gamma_x",
    "lite_logerr_gamma_z",
    "lite_reward_error",
    "reward_full",
    "gamma_x_full",
    "gamma_z_full",
    "eta_full",
    "valid_full",
    "full_logerr_gamma_x",
    "full_logerr_gamma_z",
    "full_reward_error",
    "lite_reward_rankcorr",
    "full_reward_rankcorr",
]
OPT_FIELDS = [
    "mode",
    "comparison",
    "seed",
    "method",
    "eval_index",
    "estimator",
    "q0",
    "q1",
    "q2",
    "q3",
    "estimated_reward",
    "estimated_valid",
    "gold_reward",
    "gold_T_X",
    "gold_T_Z",
    "gold_eta",
    "gold_valid",
    "best_gold_reward",
    "best_gold_T_X",
    "best_gold_T_Z",
    "best_gold_eta",
    "cumulative_settings",
    "cumulative_wait",
    "wall_time_s",
    "n_shots",
]
OPT_SUMMARY_FIELDS = [
    "mode",
    "comparison",
    "seed",
    "method",
    "n_shots",
    "final_gold_reward",
    "final_gold_T_X",
    "final_gold_T_Z",
    "final_gold_eta",
    "success",
    "improved_over_initial",
    "total_settings",
    "total_wait",
    "wall_time_s",
]


def mode_config(mode: str) -> ModeConfig:
    return MODE_CONFIGS[mode]


def make_lab(cfg: ModeConfig, *, seed: int, z_observable: str = "ideal", measurement_sigma: float = 0.0) -> CatLab:
    return CatLab(
        CatSystemConfig(
            na=cfg.na,
            nb=cfg.nb,
            kappa_b=cfg.kappa_b,
            kappa_a=cfg.kappa_a,
            z_observable=z_observable,
            measurement_sigma=measurement_sigma,
            seed=seed,
        )
    )


def q_init() -> np.ndarray:
    return np.array([np.log(0.75), np.log(1.35), 0.0, 0.0], dtype=float)


def sample_qs(n_points: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    bounds = default_q_bounds()
    return rng.uniform(bounds[:, 0], bounds[:, 1], size=(int(n_points), bounds.shape[0]))


def append_row(path: Path, row: Dict[str, object], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def existing_keys(path: Path, keys: Iterable[str]) -> set[tuple[str, ...]]:
    key_list = list(keys)
    return {tuple(str(row.get(key, "")) for key in key_list) for row in read_rows(path)}


def finite(value: object, default: float = np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def result_reason(result: EstimateResult) -> str:
    return str(result.diagnostics.get("reason", ""))


def physical_cols(q: np.ndarray, lab: CatLab) -> Dict[str, float]:
    knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    summary = physical_summary(knobs, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    return {
        "kappa_2": summary.kappa_2,
        "nbar": summary.nbar,
        "alpha_real": summary.alpha.real,
        "alpha_imag": summary.alpha.imag,
    }


def result_cols(prefix: str, result: EstimateResult) -> Dict[str, object]:
    return {
        f"reward_{prefix}": result.reward,
        f"gamma_x_{prefix}": result.gamma_x,
        f"gamma_z_{prefix}": result.gamma_z,
        f"T_X_{prefix}": result.t_x,
        f"T_Z_{prefix}": result.t_z,
        f"eta_{prefix}": result.eta,
        f"valid_{prefix}": int(result.valid),
        f"settings_{prefix}": result.settings,
        f"wait_{prefix}": result.wait_time_cost,
    }


def random_density(dim: int, rng: np.random.Generator, *, pure: bool = False) -> np.ndarray:
    if pure:
        psi = rng.normal(size=dim) + 1.0j * rng.normal(size=dim)
        psi = psi / np.linalg.norm(psi)
        return np.outer(psi, psi.conj())
    mat = rng.normal(size=(dim, dim)) + 1.0j * rng.normal(size=(dim, dim))
    rho = mat @ mat.conj().T
    return rho / np.trace(rho)


def run_simulator_check(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, object]]:
    cfg = mode_config(args.mode)
    path = out_dir / "simulator_checks.csv"
    if path.exists() and not args.resume:
        path.unlink()
    rows: List[Dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    lab = make_lab(cfg, seed=args.seed)
    bounds = default_q_bounds()

    def record(check: str, case: str, value: float, threshold: float, notes: str = "") -> None:
        row = {
            "mode": args.mode,
            "check": check,
            "case": case,
            "value": value,
            "threshold": threshold,
            "pass": int(np.isfinite(value) and value <= threshold),
            "notes": notes,
        }
        rows.append(row)
        append_row(path, row, SIM_CHECK_FIELDS)

    for idx in range(4):
        q = rng.uniform(bounds[:, 0], bounds[:, 1])
        knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
        h = lab.oracle_hamiltonian(knobs)
        herm_err = np.linalg.norm(h - h.conj().T) / max(np.linalg.norm(h), 1e-12)
        record("hamiltonian_hermiticity", f"random_{idx}", float(herm_err), 1e-11)

    q = q_init()
    knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    dim = lab._dim
    for idx in range(5):
        rho = random_density(dim, rng)
        lrho = lab.oracle_apply_liouvillian(knobs, rho)
        record("liouvillian_trace_preservation", f"rho_{idx}", abs(np.trace(lrho)), 1e-10)
        herm_err = np.linalg.norm(lrho - lrho.conj().T) / max(np.linalg.norm(lrho), 1e-12)
        record("liouvillian_hermiticity_preservation", f"rho_{idx}", float(herm_err), 1e-10)

    for idx, pure in enumerate([True, False, True, False]):
        rho = random_density(dim, rng, pure=pure)
        rho_t = lab.oracle_evolve_density(knobs, rho, 0.1)
        eig_min = float(np.min(np.linalg.eigvalsh(0.5 * (rho_t + rho_t.conj().T))))
        record("short_time_positivity", f"state_{idx}", max(0.0, -eig_min), 1e-8, f"min_eig={eig_min:.3e}")

    for prep in PREP_STATES:
        for axis in MEAS_AXES:
            value = lab.run_experiment(knobs, prep, axis, 0.0)
            expected = 1.0 if prep[1] == axis and prep[0] == "+" else -1.0 if prep[1] == axis and prep[0] == "-" else 0.0
            record("initial_logical_observable", f"{prep}_meas_{axis}", abs(value - expected), 0.25, f"value={value:.6g}, expected~{expected:g}")

    trunc_dims = [(10, 3), (12, 4)]
    if args.mode == "full" or args.include_large_truncation:
        trunc_dims.append((15, 5))
    trunc_results = []
    for na, nb in trunc_dims:
        try:
            t_cfg = ModeConfig(
                na=na,
                nb=nb,
                kappa_b=cfg.kappa_b,
                kappa_a=cfg.kappa_a,
                k_points=max(18, min(30, cfg.k_points)),
                t_final_x=cfg.t_final_x,
                t_final_z=cfg.t_final_z,
                t0=cfg.t0,
                t1=cfg.t1,
                optimizer_evals=cfg.optimizer_evals,
                optimizer_wait_budget=cfg.optimizer_wait_budget,
                time_window_subset=cfg.time_window_subset,
                finite_shot_points=cfg.finite_shot_points,
            )
            t_lab = make_lab(t_cfg, seed=args.seed + na + nb)
            t_knobs = q_to_knobs(q, kappa_b=t_lab.config.kappa_b, kappa_a=t_lab.config.kappa_a)
            res = gold_full_fit_estimate(
                t_lab.run_experiment,
                t_knobs,
                k_points=t_cfg.k_points,
                t_final_x=t_cfg.t_final_x,
                t_final_z=t_cfg.t_final_z,
                cfg=RewardConfig(eta_target=args.eta_target),
                style="contrast",
                adaptive=False,
            )
            trunc_results.append((na, nb, res))
        except Exception as exc:
            append_row(
                path,
                {
                    "mode": args.mode,
                    "check": "truncation_sanity",
                    "case": f"{na},{nb}",
                    "value": np.nan,
                    "threshold": np.nan,
                    "pass": 0,
                    "notes": f"failed_or_skipped: {exc}",
                },
                SIM_CHECK_FIELDS,
            )
    if trunc_results:
        base = trunc_results[0][2]
        for na, nb, res in trunc_results:
            rel_tx = abs(res.t_x / base.t_x - 1.0) if base.t_x > 0 and res.t_x > 0 else np.nan
            rel_tz = abs(res.t_z / base.t_z - 1.0) if base.t_z > 0 and res.t_z > 0 else np.nan
            record("truncation_sanity_T_X", f"{na},{nb}", float(rel_tx), 0.35, f"T_X={res.t_x:.6g}")
            record("truncation_sanity_T_Z", f"{na},{nb}", float(rel_tz), 0.55, f"T_Z={res.t_z:.6g}")
    elif args.mode != "full":
        append_row(
            path,
            {
                "mode": args.mode,
                "check": "truncation_sanity",
                "case": "15,5",
                "value": np.nan,
                "threshold": np.nan,
                "pass": 0,
                "notes": "large truncation skipped in quick/medium; pass --include-large-truncation to run it",
            },
            SIM_CHECK_FIELDS,
        )

    return rows


def representative_fit_figures(args: argparse.Namespace, out_dir: Path) -> list[Path]:
    cfg = mode_config(args.mode)
    lab = make_lab(cfg, seed=args.seed)
    knobs = q_to_knobs(q_init(), kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    _, curves = gold_full_fit_details(
        lab.run_experiment,
        knobs,
        k_points=cfg.k_points,
        t_final_x=cfg.t_final_x,
        t_final_z=cfg.t_final_z,
        cfg=RewardConfig(eta_target=args.eta_target),
        style="contrast",
        adaptive=args.mode != "quick",
    )
    paths = []
    paths.extend(plot_decay_fit(curves, "x", cfg.t0, cfg.t1, out_dir))
    paths.extend(plot_decay_fit(curves, "z", cfg.t0, cfg.t1, out_dir))
    return paths


def run_estimator_scan(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    n_shots: int | None = None,
    z_observable: str = "ideal",
    filename: str = "estimator_scan.csv",
    n_points_override: int | None = None,
    overwrite: bool = True,
) -> List[Dict[str, object]]:
    cfg = mode_config(args.mode)
    reward_cfg = RewardConfig(eta_target=args.eta_target)
    path = out_dir / filename
    keys = existing_keys(path, ["mode", "point_id", "n_shots", "z_observable"]) if args.resume else set()
    if path.exists() and not args.resume and overwrite:
        path.unlink()
        keys = set()

    qs = sample_qs(n_points_override or args.n_points, args.seed + 101)
    rows: List[Dict[str, object]] = []
    for point_id, q in enumerate(qs):
        key = (args.mode, str(point_id), str(n_shots), z_observable)
        if key in keys:
            continue
        lab = make_lab(cfg, seed=args.seed + point_id, z_observable=z_observable, measurement_sigma=args.gaussian_sigma)
        knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
        lab.reset_counters()
        gold, _ = gold_full_fit_details(
            lab.run_experiment,
            knobs,
            k_points=cfg.k_points,
            t_final_x=cfg.t_final_x,
            t_final_z=cfg.t_final_z,
            cfg=reward_cfg,
            style="contrast",
            n_shots=None,
            adaptive=args.mode != "quick",
        )
        challenge = gold_full_fit_estimate(
            lab.run_experiment,
            knobs,
            k_points=max(18, cfg.k_points // 2 if args.mode == "quick" else cfg.k_points),
            t_final_x=cfg.t_final_x,
            t_final_z=cfg.t_final_z,
            cfg=reward_cfg,
            style="challenge",
            n_shots=None,
            adaptive=False,
        )
        lite = blt_lite_estimate(lab.run_experiment, knobs, t0=cfg.t0, t1=cfg.t1, cfg=reward_cfg, n_shots=n_shots)
        full = blt_full_estimate(lab.run_experiment, knobs, t0=cfg.t0, t1=cfg.t1, cfg=reward_cfg, n_shots=n_shots)
        row: Dict[str, object] = {
            "mode": args.mode,
            "point_id": point_id,
            "n_shots": str(n_shots),
            "z_observable": z_observable,
            "q0": q[0],
            "q1": q[1],
            "q2": q[2],
            "q3": q[3],
            "fit_x_r2": gold.diagnostics.get("fit_x_r2", np.nan),
            "fit_z_r2": gold.diagnostics.get("fit_z_r2", np.nan),
            "fit_x_rmse": gold.diagnostics.get("fit_x_rmse", np.nan),
            "fit_z_rmse": gold.diagnostics.get("fit_z_rmse", np.nan),
            "lite_reason": result_reason(lite),
            "full_reason": result_reason(full),
            "gamma_x_full_eigen": full.diagnostics.get("gamma_x_eigen", np.nan),
            "gamma_z_full_eigen": full.diagnostics.get("gamma_z_eigen", np.nan),
            "gamma_x_full_contrast": full.diagnostics.get("gamma_x_contrast", np.nan),
            "gamma_z_full_contrast": full.diagnostics.get("gamma_z_contrast", np.nan),
        }
        row.update({f"q{i}": float(q[i]) for i in range(4)})
        row.update(physical_cols(q, lab))
        row.update(result_cols("gold", gold))
        row.update(result_cols("challenge", challenge))
        row.update(result_cols("lite", lite))
        row.update(result_cols("full", full))
        rows.append(row)
        append_row(path, row, SCAN_FIELDS)

    all_rows = [dict(row) for row in read_rows(path)]
    if filename == "estimator_scan.csv" and z_observable == "ideal" and n_shots is None:
        summary_rows = summarize_estimator_scan(all_rows, args.mode)
        write_rows(out_dir / "estimator_summary.csv", summary_rows, SUMMARY_FIELDS)
        plot_blt_vs_gold_rates(all_rows, out_dir)
        plot_blt_vs_gold_reward_eta(all_rows, out_dir, args.eta_target)
        plot_error_histograms(all_rows, out_dir)
        plot_estimator_cost(all_rows, out_dir)
    return rows


def _metric_rows_for_estimator(rows: List[Dict[str, object]], estimator: str, mode: str) -> Dict[str, object]:
    valid = []
    for row in rows:
        if int(finite(row.get(f"valid_{estimator}"), 0)) and int(finite(row.get("valid_gold"), 0)):
            valid.append(row)
    n_total = len(rows)
    n_valid = len(valid)

    def log_errors(metric: str) -> list[float]:
        vals = []
        for row in valid:
            gold = finite(row.get(f"{metric}_gold"))
            est = finite(row.get(f"{metric}_{estimator}"))
            if gold > 0 and est > 0:
                vals.append(abs(np.log(est / gold)))
        return vals

    def corr(x_key: str, y_key: str, kind: str) -> float:
        xs, ys = [], []
        for row in valid:
            x = finite(row.get(x_key))
            y = finite(row.get(y_key))
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
        if len(xs) < 3:
            return np.nan
        if kind == "spearman":
            return float(spearmanr(xs, ys).correlation)
        return float(pearsonr(xs, ys).statistic)

    reward_err = [finite(r.get(f"reward_{estimator}")) - finite(r.get("reward_gold")) for r in valid]
    settings_gold = np.nanmedian([finite(r.get("settings_gold")) for r in valid]) if valid else np.nan
    settings_est = np.nanmedian([finite(r.get(f"settings_{estimator}")) for r in valid]) if valid else np.nan
    wait_gold = np.nanmedian([finite(r.get("wait_gold")) for r in valid]) if valid else np.nan
    wait_est = np.nanmedian([finite(r.get(f"wait_{estimator}")) for r in valid]) if valid else np.nan
    return {
        "mode": mode,
        "n_rows": n_total,
        "estimator": estimator,
        "median_abs_logerr_gamma_x": float(np.nanmedian(log_errors("gamma_x"))) if n_valid else np.nan,
        "median_abs_logerr_gamma_z": float(np.nanmedian(log_errors("gamma_z"))) if n_valid else np.nan,
        "median_abs_logerr_eta": float(np.nanmedian(log_errors("eta"))) if n_valid else np.nan,
        "median_reward_error": float(np.nanmedian(reward_err)) if n_valid else np.nan,
        "spearman_reward": corr("reward_gold", f"reward_{estimator}", "spearman"),
        "pearson_log_gamma_x": corr("gamma_x_gold", f"gamma_x_{estimator}", "pearson"),
        "pearson_log_gamma_z": corr("gamma_z_gold", f"gamma_z_{estimator}", "pearson"),
        "failure_rate": 1.0 - n_valid / max(1, n_total),
        "settings_speedup": settings_gold / settings_est if settings_est and settings_est > 0 else np.nan,
        "wait_speedup": wait_gold / wait_est if wait_est and wait_est > 0 else np.nan,
    }


def summarize_estimator_scan(rows: List[Dict[str, object]], mode: str) -> List[Dict[str, object]]:
    return [_metric_rows_for_estimator(rows, "lite", mode), _metric_rows_for_estimator(rows, "full", mode)]


def run_time_window_scan(args: argparse.Namespace, out_dir: Path) -> List[Dict[str, object]]:
    cfg = mode_config(args.mode)
    reward_cfg = RewardConfig(eta_target=args.eta_target)
    path = out_dir / "time_window_scan.csv"
    if path.exists() and not args.resume:
        path.unlink()
    keys = existing_keys(path, ["mode", "point_id", "t0", "t1"]) if args.resume else set()
    t0_grid = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    t1_grid = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
    qs = sample_qs(min(args.n_points, cfg.time_window_subset), args.seed + 202)
    rows: List[Dict[str, object]] = []

    gold_by_point = {}
    for point_id, q in enumerate(qs):
        lab = make_lab(cfg, seed=args.seed + 300 + point_id)
        knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
        gold = gold_full_fit_estimate(
            lab.run_experiment,
            knobs,
            k_points=max(18, min(cfg.k_points, 40 if args.mode == "quick" else cfg.k_points)),
            t_final_x=cfg.t_final_x,
            t_final_z=cfg.t_final_z,
            cfg=reward_cfg,
            style="contrast",
            adaptive=False,
        )
        gold_by_point[point_id] = (q, knobs, gold)

    for point_id, (q, knobs, gold) in gold_by_point.items():
        for t0 in t0_grid:
            for t1 in t1_grid:
                if t1 <= t0:
                    continue
                key = (args.mode, str(point_id), str(t0), str(t1))
                if key in keys:
                    continue
                lab = make_lab(cfg, seed=args.seed + 700 + point_id)
                lite = blt_lite_estimate(lab.run_experiment, knobs, t0=t0, t1=t1, cfg=reward_cfg)
                full = blt_full_estimate(lab.run_experiment, knobs, t0=t0, t1=t1, cfg=reward_cfg)
                row = {
                    "mode": args.mode,
                    "point_id": point_id,
                    "t0": t0,
                    "t1": t1,
                    "reward_gold": gold.reward,
                    "gamma_x_gold": gold.gamma_x,
                    "gamma_z_gold": gold.gamma_z,
                    "reward_lite": lite.reward,
                    "gamma_x_lite": lite.gamma_x,
                    "gamma_z_lite": lite.gamma_z,
                    "eta_lite": lite.eta,
                    "valid_lite": int(lite.valid),
                    "lite_logerr_gamma_x": abs(np.log(lite.gamma_x / gold.gamma_x)) if lite.gamma_x > 0 and gold.gamma_x > 0 else np.nan,
                    "lite_logerr_gamma_z": abs(np.log(lite.gamma_z / gold.gamma_z)) if lite.gamma_z > 0 and gold.gamma_z > 0 else np.nan,
                    "lite_reward_error": lite.reward - gold.reward if np.isfinite(lite.reward) and np.isfinite(gold.reward) else np.nan,
                    "reward_full": full.reward,
                    "gamma_x_full": full.gamma_x,
                    "gamma_z_full": full.gamma_z,
                    "eta_full": full.eta,
                    "valid_full": int(full.valid),
                    "full_logerr_gamma_x": abs(np.log(full.gamma_x / gold.gamma_x)) if full.gamma_x > 0 and gold.gamma_x > 0 else np.nan,
                    "full_logerr_gamma_z": abs(np.log(full.gamma_z / gold.gamma_z)) if full.gamma_z > 0 and gold.gamma_z > 0 else np.nan,
                    "full_reward_error": full.reward - gold.reward if np.isfinite(full.reward) and np.isfinite(gold.reward) else np.nan,
                }
                rows.append(row)
                append_row(path, row, TIME_WINDOW_FIELDS)

    all_rows = [dict(row) for row in read_rows(path)]
    rank_rows = []
    for t0 in t0_grid:
        for t1 in t1_grid:
            if t1 <= t0:
                continue
            subset = [r for r in all_rows if finite(r.get("t0")) == t0 and finite(r.get("t1")) == t1]
            if len(subset) >= 3:
                gold_rewards = [finite(r.get("reward_gold")) for r in subset]
                lite_rewards = [finite(r.get("reward_lite")) for r in subset]
                full_rewards = [finite(r.get("reward_full")) for r in subset]
                lite_corr = float(spearmanr(gold_rewards, lite_rewards).correlation)
                full_corr = float(spearmanr(gold_rewards, full_rewards).correlation)
                for r in subset:
                    r["lite_reward_rankcorr"] = lite_corr
                    r["full_reward_rankcorr"] = full_corr
                    rank_rows.append(r)
    if rank_rows:
        write_rows(path, rank_rows, TIME_WINDOW_FIELDS)
        all_rows = rank_rows
    plot_heatmap(all_rows, out_dir, value_key="lite_logerr_gamma_x", name="time_window_heatmap_lite_gammaX", title="BLT-lite median log-error gamma_X")
    plot_heatmap(all_rows, out_dir, value_key="lite_logerr_gamma_z", name="time_window_heatmap_lite_gammaZ", title="BLT-lite median log-error gamma_Z")
    plot_heatmap(all_rows, out_dir, value_key="lite_reward_rankcorr", name="time_window_heatmap_lite_reward_rankcorr", title="BLT-lite reward rank correlation")
    plot_heatmap(all_rows, out_dir, value_key="full_reward_rankcorr", name="time_window_heatmap_full_reward_rankcorr", title="BLT-full reward rank correlation")
    return rows


@dataclass
class CandidateEval:
    q: np.ndarray
    estimated: EstimateResult
    gold: EstimateResult
    cumulative_settings: int
    cumulative_wait: float
    wall_time: float


def make_estimator(
    name: str,
    lab: CatLab,
    cfg: ModeConfig,
    reward_cfg: RewardConfig,
    n_shots: int | None,
) -> Callable[[np.ndarray], EstimateResult]:
    def evaluate(q: np.ndarray) -> EstimateResult:
        knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
        if name == "gold_contrast":
            return gold_full_fit_estimate(
                lab.run_experiment,
                knobs,
                k_points=cfg.k_points,
                t_final_x=cfg.t_final_x,
                t_final_z=cfg.t_final_z,
                cfg=reward_cfg,
                style="contrast",
                n_shots=n_shots,
                adaptive=False,
            )
        if name == "blt_lite":
            return blt_lite_estimate(lab.run_experiment, knobs, t0=cfg.t0, t1=cfg.t1, cfg=reward_cfg, n_shots=n_shots)
        if name == "blt_full":
            return blt_full_estimate(lab.run_experiment, knobs, t0=cfg.t0, t1=cfg.t1, cfg=reward_cfg, n_shots=n_shots)
        raise ValueError(f"unknown estimator {name}")

    return evaluate


def run_optimization_driver(
    *,
    method: str,
    estimator_name: str,
    comparison: str,
    seed: int,
    args: argparse.Namespace,
    out_dir: Path,
    n_shots: int | None = None,
    eval_budget: int | None = None,
    wait_budget: float | None = None,
    runs_file: str = "optimization_runs.csv",
    summary_file: str = "optimization_summary.csv",
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    cfg = mode_config(args.mode)
    reward_cfg = RewardConfig(eta_target=args.eta_target)
    bounds = default_q_bounds()
    rng = np.random.default_rng(seed)
    lab = make_lab(cfg, seed=seed, measurement_sigma=args.gaussian_sigma)
    gold_lab = make_lab(cfg, seed=seed + 100000)
    evaluate_est = make_estimator(estimator_name, lab, cfg, reward_cfg, n_shots)

    q = np.minimum(np.maximum(q_init(), bounds[:, 0]), bounds[:, 1])
    mean = q.copy()
    sigma = 0.23
    best_est_reward = -np.inf
    best_est_q = q.copy()
    best_gold: EstimateResult | None = None
    rows: List[Dict[str, object]] = []
    cumulative_settings = 0
    cumulative_wait = 0.0
    start = time.perf_counter()

    def evaluate_candidate(q_candidate: np.ndarray, eval_index: int) -> CandidateEval:
        nonlocal cumulative_settings, cumulative_wait, best_gold, best_est_reward, best_est_q
        est = evaluate_est(q_candidate)
        cumulative_settings += int(est.settings)
        cumulative_wait += float(est.wait_time_cost)
        knobs = q_to_knobs(q_candidate, kappa_b=gold_lab.config.kappa_b, kappa_a=gold_lab.config.kappa_a)
        gold = gold_full_fit_estimate(
            gold_lab.run_experiment,
            knobs,
            k_points=max(12, min(cfg.k_points, 18 if args.mode == "quick" else cfg.k_points)),
            t_final_x=cfg.t_final_x,
            t_final_z=cfg.t_final_z,
            cfg=reward_cfg,
            style="contrast",
            n_shots=None,
            adaptive=False,
        )
        if best_gold is None or gold.reward > best_gold.reward:
            best_gold = gold
        if est.reward > best_est_reward:
            best_est_reward = est.reward
            best_est_q = q_candidate.copy()
        return CandidateEval(
            q=q_candidate.copy(),
            estimated=est,
            gold=gold,
            cumulative_settings=cumulative_settings,
            cumulative_wait=cumulative_wait,
            wall_time=time.perf_counter() - start,
        )

    def add_eval(ev: CandidateEval, eval_index: int) -> None:
        assert best_gold is not None
        row = {
            "mode": args.mode,
            "comparison": comparison,
            "seed": seed,
            "method": method,
            "eval_index": eval_index,
            "estimator": estimator_name,
            "q0": ev.q[0],
            "q1": ev.q[1],
            "q2": ev.q[2],
            "q3": ev.q[3],
            "estimated_reward": ev.estimated.reward,
            "estimated_valid": int(ev.estimated.valid),
            "gold_reward": ev.gold.reward,
            "gold_T_X": ev.gold.t_x,
            "gold_T_Z": ev.gold.t_z,
            "gold_eta": ev.gold.eta,
            "gold_valid": int(ev.gold.valid),
            "best_gold_reward": best_gold.reward,
            "best_gold_T_X": best_gold.t_x,
            "best_gold_T_Z": best_gold.t_z,
            "best_gold_eta": best_gold.eta,
            "cumulative_settings": ev.cumulative_settings,
            "cumulative_wait": ev.cumulative_wait,
            "wall_time_s": ev.wall_time,
            "n_shots": str(n_shots),
        }
        rows.append(row)
        append_row(out_dir / runs_file, row, OPT_FIELDS)

    max_evals = eval_budget or cfg.optimizer_evals
    eval_index = 0
    if method.endswith("spsa"):
        ev = evaluate_candidate(q, eval_index)
        add_eval(ev, eval_index)
        eval_index += 1
        while eval_index < max_evals and (wait_budget is None or cumulative_wait < wait_budget):
            k = max(1, eval_index)
            ck = 0.11 / k**0.101
            ak = 0.20 / (k + 2.0) ** 0.602
            delta = rng.choice(np.array([-1.0, 1.0]), size=q.shape)
            q_plus = np.minimum(np.maximum(q + ck * delta, bounds[:, 0]), bounds[:, 1])
            q_minus = np.minimum(np.maximum(q - ck * delta, bounds[:, 0]), bounds[:, 1])
            ev_plus = evaluate_candidate(q_plus, eval_index)
            add_eval(ev_plus, eval_index)
            eval_index += 1
            if eval_index >= max_evals or (wait_budget is not None and cumulative_wait >= wait_budget):
                break
            ev_minus = evaluate_candidate(q_minus, eval_index)
            add_eval(ev_minus, eval_index)
            eval_index += 1
            grad = (ev_plus.estimated.reward - ev_minus.estimated.reward) / (2.0 * ck * delta)
            if not np.all(np.isfinite(grad)):
                grad = np.zeros_like(q)
            step = ak * grad
            norm = np.linalg.norm(step)
            if norm > 0.22:
                step *= 0.22 / norm
            q = np.minimum(np.maximum(q + step, bounds[:, 0]), bounds[:, 1])
    elif method.endswith("cmaes") or method.endswith("es"):
        while eval_index < max_evals and (wait_budget is None or cumulative_wait < wait_budget):
            pop = min(4, max_evals - eval_index)
            candidates = [mean] + [
                np.minimum(np.maximum(mean + sigma * rng.normal(size=mean.shape), bounds[:, 0]), bounds[:, 1])
                for _ in range(max(0, pop - 1))
            ]
            scored = []
            for cand in candidates:
                if eval_index >= max_evals or (wait_budget is not None and cumulative_wait >= wait_budget):
                    break
                ev = evaluate_candidate(cand, eval_index)
                add_eval(ev, eval_index)
                scored.append((ev.estimated.reward, cand))
                eval_index += 1
            if scored:
                scored.sort(key=lambda item: item[0], reverse=True)
                elites = np.array([item[1] for item in scored[: max(1, len(scored) // 2)]])
                mean = np.mean(elites, axis=0)
                sigma *= 0.72
    elif method == "random_search":
        while eval_index < max_evals and (wait_budget is None or cumulative_wait < wait_budget):
            cand = rng.uniform(bounds[:, 0], bounds[:, 1]) if eval_index else q
            ev = evaluate_candidate(cand, eval_index)
            add_eval(ev, eval_index)
            eval_index += 1
    else:
        raise ValueError(f"unknown method {method}")

    final_knobs = q_to_knobs(best_est_q, kappa_b=gold_lab.config.kappa_b, kappa_a=gold_lab.config.kappa_a)
    final_gold = gold_full_fit_estimate(
        gold_lab.run_experiment,
        final_knobs,
        k_points=cfg.k_points,
        t_final_x=cfg.t_final_x,
        t_final_z=cfg.t_final_z,
        cfg=reward_cfg,
        style="contrast",
        adaptive=False,
    )
    initial_gold = rows[0]["gold_reward"] if rows else np.nan
    success = (
        final_gold.valid
        and final_gold.eta >= args.eta_target / 1.25
        and final_gold.eta <= args.eta_target * 1.25
        and final_gold.reward > finite(initial_gold)
    )
    summary = {
        "mode": args.mode,
        "comparison": comparison,
        "seed": seed,
        "method": method,
        "n_shots": str(n_shots),
        "final_gold_reward": final_gold.reward,
        "final_gold_T_X": final_gold.t_x,
        "final_gold_T_Z": final_gold.t_z,
        "final_gold_eta": final_gold.eta,
        "success": int(success),
        "improved_over_initial": int(final_gold.reward > finite(initial_gold)),
        "total_settings": cumulative_settings,
        "total_wait": cumulative_wait,
        "wall_time_s": time.perf_counter() - start,
    }
    append_row(out_dir / summary_file, summary, OPT_SUMMARY_FIELDS)
    return rows, summary


def run_optimizer_comparison(args: argparse.Namespace, out_dir: Path) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if (out_dir / "optimization_runs.csv").exists() and not args.resume:
        (out_dir / "optimization_runs.csv").unlink()
    if (out_dir / "optimization_summary.csv").exists() and not args.resume:
        (out_dir / "optimization_summary.csv").unlink()
    cfg = mode_config(args.mode)
    methods = [
        ("naive_fullfit_cmaes", "gold_contrast"),
        ("naive_fullfit_spsa", "gold_contrast"),
        ("blt_lite_spsa", "blt_lite"),
        ("blt_full_spsa", "blt_full"),
        ("blt_lite_es", "blt_lite"),
        ("random_search", "gold_contrast"),
    ]
    method_seed_offsets = {name: 17 + 101 * idx for idx, (name, _) in enumerate(methods)}
    all_rows: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []
    for comparison in ("equal_candidates", "equal_wait"):
        for seed_offset in range(args.seeds):
            for method, estimator in methods:
                seed = args.seed + 1000 * seed_offset + method_seed_offsets[method]
                rows, summary = run_optimization_driver(
                    method=method,
                    estimator_name=estimator,
                    comparison=comparison,
                    seed=seed,
                    args=args,
                    out_dir=out_dir,
                    n_shots=None,
                    eval_budget=cfg.optimizer_evals,
                    wait_budget=cfg.optimizer_wait_budget if comparison == "equal_wait" else None,
                )
                all_rows.extend(rows)
                summaries.append(summary)
    plot_optimization_traces(read_rows(out_dir / "optimization_runs.csv"), out_dir, args.eta_target)
    plot_final_optimizer_figures(read_rows(out_dir / "optimization_summary.csv"), out_dir, args.eta_target)
    return all_rows, summaries


def run_finite_shot_validation(args: argparse.Namespace, out_dir: Path) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    shot_values: list[int | None] = [None, 10000, 1000, 300]
    finite_scan_path = out_dir / "finite_shot_estimator_scan.csv"
    if finite_scan_path.exists() and not args.resume:
        finite_scan_path.unlink()
    estimator_rows: List[Dict[str, object]] = []
    for shots in shot_values:
        estimator_rows.extend(
            run_estimator_scan(
                args,
                out_dir,
                n_shots=shots,
                filename="finite_shot_estimator_scan.csv",
                n_points_override=min(args.n_points, mode_config(args.mode).finite_shot_points),
                overwrite=False,
            )
        )
    opt_path = out_dir / "finite_shot_optimization.csv"
    if opt_path.exists() and not args.resume:
        opt_path.unlink()
    opt_rows: List[Dict[str, object]] = []
    for shots in shot_values:
        for seed_offset in range(max(1, min(args.seeds, 4 if args.mode == "quick" else args.seeds))):
            rows, summary = run_optimization_driver(
                method="blt_lite_spsa",
                estimator_name="blt_lite",
                comparison="finite_shot",
                seed=args.seed + 5000 + seed_offset + (0 if shots is None else shots),
                args=args,
                out_dir=out_dir,
                n_shots=shots,
                eval_budget=max(4, mode_config(args.mode).optimizer_evals // 2),
                wait_budget=None,
                runs_file="finite_shot_optimization_runs.csv",
                summary_file="finite_shot_optimization.csv",
            )
            opt_rows.append(summary)
    all_est = read_rows(out_dir / "finite_shot_estimator_scan.csv")
    all_opt = read_rows(opt_path)
    plot_finite_shot_figures(all_est, all_opt, out_dir)
    return estimator_rows, opt_rows


def run_z_proxy_comparison(args: argparse.Namespace, out_dir: Path) -> None:
    rows = []
    path = out_dir / "z_observable_comparison.csv"
    if path.exists() and not args.resume:
        path.unlink()
    for z_obs in ("ideal", "quadrature_proxy"):
        rows.extend(
            run_estimator_scan(
                args,
                out_dir,
                n_shots=None,
                z_observable=z_obs,
                filename="z_observable_comparison.csv",
                n_points_override=min(args.n_points, 12 if args.mode == "quick" else 30),
                overwrite=False,
            )
        )


def aggregate_optimizer_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple[str, str], List[Dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["comparison"], row["method"]), []).append(row)
    out = []
    for (comparison, method), group in grouped.items():
        rewards = np.array([finite(r.get("final_gold_reward")) for r in group], dtype=float)
        success = np.array([finite(r.get("success"), 0.0) for r in group], dtype=float)
        waits = np.array([finite(r.get("total_wait")) for r in group], dtype=float)
        out.append(
            {
                "comparison": comparison,
                "method": method,
                "n": len(group),
                "reward_mean": float(np.nanmean(rewards)),
                "reward_sem": float(np.nanstd(rewards, ddof=1) / np.sqrt(len(rewards))) if len(rewards) > 1 else 0.0,
                "success_rate": float(np.nanmean(success)),
                "median_wait": float(np.nanmedian(waits)),
            }
        )
    return out


def write_report(args: argparse.Namespace, out_dir: Path) -> Path:
    scan_summary = read_rows(out_dir / "estimator_summary.csv")
    opt_summary_rows = read_rows(out_dir / "optimization_summary.csv")
    opt_agg = aggregate_optimizer_summary(opt_summary_rows) if opt_summary_rows else []
    sim_rows = read_rows(out_dir / "simulator_checks.csv")
    fail_checks = [r for r in sim_rows if str(r.get("pass")) == "0"]
    path = ROOT / "report_validation.md"

    def table(rows: List[Dict[str, object]], columns: List[str]) -> str:
        if not rows:
            return "_No rows generated yet._\n"
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in rows:
            vals = []
            for col in columns:
                val = row.get(col, "")
                try:
                    fval = float(val)
                    vals.append(f"{fval:.4g}")
                except (TypeError, ValueError):
                    vals.append(str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines) + "\n"

    text = f"""# BLT Validation Report

Generated by `python team-blt/validation.py` in `{args.mode}` mode.

## 1. Simulator Consistency

The simulator uses the challenge storage+buffer Hamiltonian and jump operators, with no added effective two-photon dissipator.  The validation-only checks cover Hamiltonian Hermiticity, trace preservation, Hermiticity preservation, short-time positivity, initial logical observables, and truncation sensitivity.

Failed or flagged checks: {len(fail_checks)}.  Some initial logical observable flags are expected when the coherent-state logical basis is non-orthogonal at modest cat size; the raw values are in `outputs_validation/simulator_checks.csv`.

## 2. Naive Full-Fit Baseline

The main scientific reference is the contrast-style full fit using `+/-x` and `+/-z` contrasts.  Challenge-style `+x`, `+z` fits are kept for comparison.  Representative decay plots include raw points, robust exponential fits, BLT two-time markers, and residual panels.

## 3-4. BLT Accuracy Against Gold

{table(scan_summary, ["estimator", "median_abs_logerr_gamma_x", "median_abs_logerr_gamma_z", "median_abs_logerr_eta", "spearman_reward", "failure_rate", "settings_speedup", "wait_speedup"])}

BLT-lite is cheapest because it uses 8 settings per candidate.  BLT-full costs 36 settings but can expose off-axis logical mixing and reports diagonal, eigenvalue, and direct-contrast rate variants.

## 5. Time Window

The default `(t0,t1)` is not assumed blindly.  `time_window_scan.csv` and the heatmaps scan `t0 = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]` and `t1 = [3.0, 4.0, 6.0, 8.0, 12.0, 16.0]`.  The chosen default is defensible only where the heatmaps show low median log-error and positive reward rank correlation.

## 6-9. Optimizer Comparison

Final performance is always judged by the gold contrast full-fit estimator, never by the optimizer's internal estimated reward.

{table(opt_agg, ["comparison", "method", "n", "reward_mean", "reward_sem", "success_rate", "median_wait"])}

Equal-candidate comparisons probe optimizer quality.  Equal-wait comparisons probe lab efficiency and are the main test of BLT's measurement-cost advantage.

## 10. Failure Modes

BLT can fail when the two-time window sees too little decay, when contrast is poor, when no clean slow-logical timescale exists after burn-in, or when `gamma_Z` is so small that `t1` must be much larger.  Reward-only reporting can also hide a tradeoff where eta improves because `T_Z` is reduced; the Pareto figure is included for that reason.

## 11. Finite-Shot Robustness

Finite-shot scans use `n_shots = None, 10000, 1000, 300`.  The estimator-noise and optimization-noise figures summarize when two-time BLT remains usable and when shot noise destabilizes the logarithmic contrast ratio.

## 12. Honest Claim

The defensible claim is conditional: BLT is a lab-accessible, low-cost boundary-mode estimator that can match the ranking of gold full fits in regimes with adequate contrast and timescale separation, and can reduce physical wait-time cost substantially.  It should not be claimed as universally superior; failures are visible in the estimator scan, time-window scan, finite-shot scan, and Pareto plots.

## Generated Figures

- `representative_X_decay_fit.png/pdf`
- `representative_Z_decay_fit.png/pdf`
- `blt_vs_gold_rates.png/pdf`
- `blt_vs_gold_reward_eta.png/pdf`
- `estimator_error_histograms.png/pdf`
- `estimator_cost_comparison.png/pdf`
- `time_window_heatmap_lite_gammaX.png/pdf`
- `time_window_heatmap_lite_gammaZ.png/pdf`
- `time_window_heatmap_lite_reward_rankcorr.png/pdf`
- `time_window_heatmap_full_reward_rankcorr.png/pdf`
- `optimization_reward_vs_wait_time.png/pdf`
- `optimization_reward_vs_settings.png/pdf`
- `optimization_eta_vs_wait_time.png/pdf`
- `final_reward_boxplot.png/pdf`
- `final_TX_TZ_pareto.png/pdf`
- `final_cost_vs_reward.png/pdf`
- `finite_shot_estimator_error.png/pdf`
- `finite_shot_optimization_reward.png/pdf`
"""
    path.write_text(text)
    return path


def print_summary(out_dir: Path) -> None:
    print("\nEstimator accuracy:")
    for row in read_rows(out_dir / "estimator_summary.csv"):
        print(
            f"  {row['estimator']}: med logerr gammaX={finite(row['median_abs_logerr_gamma_x']):.3g}, "
            f"gammaZ={finite(row['median_abs_logerr_gamma_z']):.3g}, "
            f"eta={finite(row['median_abs_logerr_eta']):.3g}, "
            f"Spearman reward={finite(row['spearman_reward']):.3g}, "
            f"settings speedup={finite(row['settings_speedup']):.2g}x, "
            f"wait speedup={finite(row['wait_speedup']):.2g}x"
        )
    opt_rows = aggregate_optimizer_summary(read_rows(out_dir / "optimization_summary.csv"))
    if opt_rows:
        print("\nOptimizer final gold reward:")
        for row in opt_rows:
            print(
                f"  {row['comparison']} / {row['method']}: "
                f"{row['reward_mean']:.4g} +/- {row['reward_sem']:.2g}, "
                f"success={row['success_rate']:.2f}, median_wait={row['median_wait']:.3g}"
            )
    figures = sorted(str(p.relative_to(ROOT)) for p in out_dir.glob("*.png"))
    if figures:
        print("\nGenerated figure PNGs:")
        for fig in figures:
            print(f"  {fig}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=sorted(MODE_CONFIGS), default="quick")
    parser.add_argument("--all", action="store_true", help="run all non-optional no-drift validations plus finite-shot checks")
    parser.add_argument("--simulator-check", action="store_true")
    parser.add_argument("--estimator-scan", action="store_true")
    parser.add_argument("--time-window-scan", action="store_true")
    parser.add_argument("--optimizer-comparison", action="store_true")
    parser.add_argument("--finite-shot", action="store_true")
    parser.add_argument("--z-proxy-comparison", action="store_true")
    parser.add_argument("--n-points", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eta-target", type=float, default=30.0)
    parser.add_argument("--gaussian-sigma", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true", help="append/skip already checkpointed rows instead of overwriting")
    parser.add_argument("--include-large-truncation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "mode": args.mode,
        "n_points": args.n_points,
        "seeds": args.seeds,
        "seed": args.seed,
        "eta_target": args.eta_target,
        "mode_config": mode_config(args.mode).__dict__,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    figure_paths: list[Path] = []
    if args.all or args.simulator_check:
        print("Running simulator checks...")
        run_simulator_check(args, out_dir)
        figure_paths.extend(representative_fit_figures(args, out_dir))
    if args.all or args.estimator_scan:
        print("Running estimator scan...")
        run_estimator_scan(args, out_dir)
    if args.all or args.time_window_scan:
        print("Running BLT time-window scan...")
        run_time_window_scan(args, out_dir)
    if args.all or args.optimizer_comparison:
        print("Running optimizer comparison...")
        run_optimizer_comparison(args, out_dir)
    if args.all or args.finite_shot:
        print("Running finite-shot validation...")
        run_finite_shot_validation(args, out_dir)
    if args.z_proxy_comparison:
        print("Running ideal-Z vs quadrature-proxy comparison...")
        run_z_proxy_comparison(args, out_dir)
    report = write_report(args, out_dir)
    print_summary(out_dir)
    print(f"\nValidation report: {report}")
    print(f"Validation outputs: {out_dir}")


if __name__ == "__main__":
    main()
