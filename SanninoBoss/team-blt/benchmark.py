"""Quick benchmark for Boundary Liouvillian Tracking.

Run:
    python team-blt/benchmark.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = ROOT / "outputs" / ".cache"
_MPL_ROOT = ROOT / "outputs" / ".mplconfig"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_ROOT))

from cat_env import CatLab, CatSystemConfig, default_q_bounds, physical_summary, q_to_knobs
from estimators import (
    EstimateResult,
    RewardConfig,
    blt_full_estimate,
    blt_lite_estimate,
    gold_full_fit_estimate,
)
from optimizers import BLTSPSA, RandomCMAStyle, RandomCMAStyleConfig, SPSAConfig


def make_output_dir(path: str | None) -> Path:
    out = Path(path) if path else ROOT / "outputs" / "quick_benchmark_latest"
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def result_row(method: str, result: EstimateResult, extras: Dict[str, object]) -> Dict[str, object]:
    row: Dict[str, object] = {
        "method": method,
        "gold_reward": result.reward,
        "gold_valid": int(result.valid),
        "gold_T_X": result.t_x,
        "gold_T_Z": result.t_z,
        "gold_eta": result.eta,
        "gold_gamma_x": result.gamma_x,
        "gold_gamma_z": result.gamma_z,
        "gold_settings": result.settings,
        "gold_wait_time_cost": result.wait_time_cost,
    }
    row.update(extras)
    return row


def evaluate_factory(
    lab: CatLab,
    reward_cfg: RewardConfig,
    estimator_name: str,
    *,
    t0: float,
    t1: float,
    k_points: int,
    t_final_x: float,
    t_final_z: float,
    n_shots: int | None,
) -> Callable[[np.ndarray], EstimateResult]:
    def evaluate(q: np.ndarray) -> EstimateResult:
        knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
        if estimator_name == "blt_lite":
            return blt_lite_estimate(
                lab.run_experiment, knobs, t0=t0, t1=t1, cfg=reward_cfg, n_shots=n_shots
            )
        if estimator_name == "blt_full":
            return blt_full_estimate(
                lab.run_experiment, knobs, t0=t0, t1=t1, cfg=reward_cfg, n_shots=n_shots
            )
        if estimator_name == "naive":
            return gold_full_fit_estimate(
                lab.run_experiment,
                knobs,
                k_points=k_points,
                t_final_x=t_final_x,
                t_final_z=t_final_z,
                cfg=reward_cfg,
                style="contrast",
                n_shots=n_shots,
            )
        raise ValueError(f"unknown estimator {estimator_name}")

    return evaluate


def gold_evaluate(
    lab: CatLab,
    reward_cfg: RewardConfig,
    q: np.ndarray,
    *,
    k_points: int,
    t_final_x: float,
    t_final_z: float,
    n_shots: int | None,
) -> EstimateResult:
    knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    return gold_full_fit_estimate(
        lab.run_experiment,
        knobs,
        k_points=k_points,
        t_final_x=t_final_x,
        t_final_z=t_final_z,
        cfg=reward_cfg,
        style="contrast",
        n_shots=n_shots,
    )


def add_physical_columns(row: Dict[str, object], q: np.ndarray, lab: CatLab) -> None:
    knobs = q_to_knobs(q, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    summary = physical_summary(knobs, kappa_b=lab.config.kappa_b, kappa_a=lab.config.kappa_a)
    row.update(
        {
            "kappa_2": summary.kappa_2,
            "nbar": summary.nbar,
            "alpha_real": summary.alpha.real,
            "alpha_imag": summary.alpha.imag,
            "g2_real": summary.g2.real,
            "g2_imag": summary.g2.imag,
            "eps_d_real": summary.eps_d.real,
            "eps_d_imag": summary.eps_d.imag,
        }
    )


def make_plots(out_dir: Path, history_rows: List[Dict[str, object]], summary_rows: List[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    if history_rows:
        center_rows = [r for r in history_rows if r.get("eval_type") in ("center", "mean", "sample0")]
        methods = sorted({str(r["method"]) for r in center_rows})
        for metric, ylabel, fname in (
            ("reward", "estimated reward", "reward_vs_iteration.png"),
            ("eta", "estimated eta", "eta_vs_iteration.png"),
        ):
            fig, ax = plt.subplots(figsize=(6.0, 3.4), dpi=150)
            for method in methods:
                rows = [r for r in center_rows if r["method"] == method]
                ax.plot(
                    [int(r["iteration"]) for r in rows],
                    [float(r[metric]) for r in rows],
                    marker="o",
                    linewidth=1.5,
                    label=method,
                )
            ax.set_xlabel("iteration")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / fname)
            plt.close(fig)

    if summary_rows:
        methods = [str(r["method"]) for r in summary_rows]
        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
        x = np.arange(len(summary_rows))
        width = 0.36
        ax.bar(x - width / 2, [float(r["optimizer_settings"]) for r in summary_rows], width, label="settings")
        ax.bar(
            x + width / 2,
            [float(r["optimizer_wait_time_cost"]) for r in summary_rows],
            width,
            label="wait-time cost",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_ylabel("cost")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "measurement_cost_comparison.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=150)
        ax.plot(methods, [float(r["gold_T_X"]) for r in summary_rows], marker="o", label="T_X")
        ax.plot(methods, [float(r["gold_T_Z"]) for r in summary_rows], marker="o", label="T_Z")
        ax.set_yscale("log")
        ax.set_ylabel("gold fitted lifetime")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "tx_tz_diagnostics.png")
        plt.close(fig)


def run_benchmark(args: argparse.Namespace) -> None:
    out_dir = make_output_dir(args.output_dir)
    (out_dir / ".mplconfig").mkdir(exist_ok=True)

    if args.quick:
        lab_cfg = CatSystemConfig(na=10, nb=3, kappa_b=8.0, kappa_a=0.15, seed=args.seed)
        t0, t1 = 1.0, 6.0
        k_points = 30
        t_final_x, t_final_z = 10.0, 80.0
        spsa_iter = args.spsa_iter if args.spsa_iter is not None else 4
        baseline_iter = args.baseline_iter if args.baseline_iter is not None else 1
        baseline_pop = args.baseline_pop if args.baseline_pop is not None else 4
    else:
        lab_cfg = CatSystemConfig(na=12, nb=4, kappa_b=10.0, kappa_a=0.12, seed=args.seed)
        t0, t1 = 1.0, 8.0
        k_points = 100
        t_final_x, t_final_z = 14.0, 120.0
        spsa_iter = args.spsa_iter if args.spsa_iter is not None else 8
        baseline_iter = args.baseline_iter if args.baseline_iter is not None else 4
        baseline_pop = args.baseline_pop if args.baseline_pop is not None else 6

    lab = CatLab(lab_cfg)
    bounds = default_q_bounds()
    q_init = np.array([np.log(0.75), np.log(1.35), 0.0, 0.0], dtype=float)
    q_init = np.minimum(np.maximum(q_init, bounds[:, 0]), bounds[:, 1])
    reward_cfg = RewardConfig(eta_target=args.eta_target)
    n_shots = args.shots if args.shots and args.shots > 0 else None

    history_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    # Initial gold reference.
    lab.reset_counters()
    lab.clear_cache()
    initial_gold = gold_evaluate(
        lab,
        reward_cfg,
        q_init,
        k_points=k_points,
        t_final_x=t_final_x,
        t_final_z=t_final_z,
        n_shots=n_shots,
    )
    initial_counts = lab.counters()
    init_row = result_row(
        "initial",
        initial_gold,
        {
            "optimizer_reward": initial_gold.reward,
            "optimizer_valid": int(initial_gold.valid),
            "optimizer_settings": 0,
            "optimizer_wait_time_cost": 0.0,
            "optimizer_wall_time_s": 0.0,
            "candidate_evaluations": 0,
            "gold_eval_settings": initial_counts["settings"],
            "gold_eval_wait_time_cost": initial_counts["wait_time_cost"],
        },
    )
    add_physical_columns(init_row, q_init, lab)
    summary_rows.append(init_row)

    methods = []

    baseline = RandomCMAStyle(
        RandomCMAStyleConfig(
            n_iter=baseline_iter,
            population=baseline_pop,
            seed=args.seed + 17,
        ),
        bounds,
        "naive_random_es",
    )
    methods.append(("naive_random_es", baseline, "naive"))

    lite = BLTSPSA(
        SPSAConfig(n_iter=spsa_iter, seed=args.seed + 31),
        bounds,
        "blt_lite_spsa",
    )
    methods.append(("blt_lite_spsa", lite, "blt_lite"))

    if not args.skip_full:
        full = BLTSPSA(
            SPSAConfig(
                n_iter=max(2, min(spsa_iter, 3 if args.quick else spsa_iter)),
                a0=0.14,
                c0=0.08,
                trust_radius=0.16,
                seed=args.seed + 47,
            ),
            bounds,
            "blt_full_spsa",
        )
        methods.append(("blt_full_spsa", full, "blt_full"))

    for method_name, optimizer, estimator_name in methods:
        lab.reset_counters()
        lab.clear_cache()
        evaluate = evaluate_factory(
            lab,
            reward_cfg,
            estimator_name,
            t0=t0,
            t1=t1,
            k_points=k_points,
            t_final_x=t_final_x,
            t_final_z=t_final_z,
            n_shots=n_shots,
        )
        start = time.perf_counter()
        opt_result = optimizer.run(q_init, evaluate)
        wall = time.perf_counter() - start
        optimizer_counts = lab.counters()
        history_rows.extend(opt_result.history)

        lab.reset_counters()
        gold = gold_evaluate(
            lab,
            reward_cfg,
            opt_result.q_best,
            k_points=k_points,
            t_final_x=t_final_x,
            t_final_z=t_final_z,
            n_shots=n_shots,
        )
        gold_counts = lab.counters()
        row = result_row(
            method_name,
            gold,
            {
                "optimizer_reward": opt_result.best.reward,
                "optimizer_valid": int(opt_result.best.valid),
                "optimizer_settings": optimizer_counts["settings"],
                "optimizer_wait_time_cost": optimizer_counts["wait_time_cost"],
                "optimizer_wall_time_s": wall,
                "candidate_evaluations": opt_result.n_evaluations,
                "gold_eval_settings": gold_counts["settings"],
                "gold_eval_wait_time_cost": gold_counts["wait_time_cost"],
            },
        )
        add_physical_columns(row, opt_result.q_best, lab)
        summary_rows.append(row)

    write_csv(out_dir / "history.csv", history_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    make_plots(out_dir, history_rows, summary_rows)

    metadata = {
        "quick": args.quick,
        "lab_config": lab_cfg.__dict__,
        "reward_config": reward_cfg.__dict__,
        "t0": t0,
        "t1": t1,
        "k_points": k_points,
        "t_final_x": t_final_x,
        "t_final_z": t_final_z,
        "n_shots": n_shots,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"Saved outputs to: {out_dir}")
    print("\nSummary:")
    header = (
        "method",
        "gold_reward",
        "gold_T_X",
        "gold_T_Z",
        "gold_eta",
        "candidate_evaluations",
        "optimizer_settings",
        "optimizer_wait_time_cost",
        "optimizer_wall_time_s",
    )
    print(",".join(header))
    for row in summary_rows:
        print(
            ",".join(
                [
                    str(row["method"]),
                    f"{float(row['gold_reward']):.6g}",
                    f"{float(row['gold_T_X']):.6g}",
                    f"{float(row['gold_T_Z']):.6g}",
                    f"{float(row['gold_eta']):.6g}",
                    str(row["candidate_evaluations"]),
                    str(row["optimizer_settings"]),
                    f"{float(row['optimizer_wait_time_cost']):.6g}",
                    f"{float(row['optimizer_wall_time_s']):.3f}",
                ]
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use laptop-friendly dimensions and iterations")
    parser.add_argument("--output-dir", default=None, help="directory for CSVs and plots")
    parser.add_argument("--eta-target", type=float, default=30.0, help="target bias eta = T_Z/T_X")
    parser.add_argument("--shots", type=int, default=0, help="optional finite-shot sampling per setting")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--spsa-iter", type=int, default=None)
    parser.add_argument("--baseline-iter", type=int, default=None)
    parser.add_argument("--baseline-pop", type=int, default=None)
    parser.add_argument("--skip-full", action="store_true", help="skip the 36-setting BLT-full optimizer")
    return parser.parse_args()


if __name__ == "__main__":
    run_benchmark(parse_args())
