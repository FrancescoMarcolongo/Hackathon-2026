"""Compare the original reward with a two-stage lifetime-after-target reward.

This script keeps the simulator, bounds, optimizer, seed, and budget matched
between methods.  It writes direct comparison artifacts for eta, Tx/Tz, reward,
and wall-clock time under ``results/`` and ``figures/``.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
(CACHE_ROOT / "mpl").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "mpl"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

try:
    from cmaes import SepCMA

    OPTIMIZER_NAME = "cmaes.SepCMA"
except ModuleNotFoundError:
    OPTIMIZER_NAME = "local.SimpleSepCMA"

    class SepCMA:
        """Small diagonal evolution-strategy fallback when cmaes is unavailable."""

        def __init__(
            self,
            *,
            mean: np.ndarray,
            sigma: float,
            bounds: np.ndarray,
            population_size: int,
            seed: int,
        ) -> None:
            self.mean = np.asarray(mean, dtype=float)
            self.sigma = float(sigma)
            self.bounds = np.asarray(bounds, dtype=float)
            self.population_size = int(population_size)
            self.rng = np.random.default_rng(int(seed))
            self._best_loss = np.inf

        def ask(self) -> np.ndarray:
            sample = self.mean + self.rng.normal(0.0, self.sigma, size=self.mean.shape)
            return np.minimum(np.maximum(sample, self.bounds[:, 0]), self.bounds[:, 1])

        def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
            ordered = sorted(solutions, key=lambda item: float(item[1]))
            parents = np.asarray([item[0] for item in ordered[: max(1, len(ordered) // 2)]], dtype=float)
            old_mean = self.mean.copy()
            self.mean = np.mean(parents, axis=0)
            self.mean = np.minimum(np.maximum(self.mean, self.bounds[:, 0]), self.bounds[:, 1])
            spread = float(np.mean(np.std(parents, axis=0))) if len(parents) > 1 else 0.0
            improved = float(ordered[0][1]) < self._best_loss
            self._best_loss = min(self._best_loss, float(ordered[0][1]))
            drift = float(np.linalg.norm(self.mean - old_mean) / max(1.0, np.sqrt(self.mean.size)))
            if improved:
                self.sigma = max(0.04, min(1.5, 0.92 * self.sigma + 0.20 * max(spread, drift)))
            else:
                self.sigma = max(0.04, 0.82 * self.sigma)

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from cat_model import SimulationConfig, clear_measure_cache, measure_lifetimes, params_to_complex
from rewards import TwoStageRewardConfig, compute_reward as compute_two_stage_reward
from validation import write_csv, write_json

ORIGINAL_ROOT = ROOT.parent / "team-core-bias-optimizer_sannino"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

TARGET_BIAS = 100.0
BIAS_TOL_REL = 0.03
OPTIMIZATION_START_X = np.array([1.0, 0.0, 2.5, 0.0], dtype=float)
BOUNDS = np.array(
    [
        [0.25, 3.0],
        [-1.0, 1.0],
        [0.50, 8.0],
        [-3.0, 3.0],
    ],
    dtype=float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a laptop-sized smoke comparison.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs per method.")
    parser.add_argument("--population", type=int, default=None, help="Override CMA-ES population.")
    parser.add_argument("--seed", type=int, default=0, help="Shared optimizer seed.")
    parser.add_argument("--sigma0", type=float, default=0.45, help="Shared CMA-ES initial sigma.")
    parser.add_argument("--target-bias", type=float, default=TARGET_BIAS)
    parser.add_argument(
        "--baseline-variant",
        choices=("exact_target", "target_band"),
        default="exact_target",
        help="Original reward variant used as the baseline comparator.",
    )
    parser.add_argument("--plot-only", action="store_true", help="Regenerate figures from saved comparison CSV.")
    return parser.parse_args()


def load_original_rewards():
    rewards_path = ORIGINAL_ROOT / "rewards.py"
    spec = importlib.util.spec_from_file_location("original_sannino_rewards", rewards_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load original rewards from {rewards_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def original_reward_config(original_rewards, args: argparse.Namespace):
    if args.baseline_variant == "target_band":
        return original_rewards.RewardConfig(
            name="original_target_band_strict",
            variant="target_band",
            target_bias=float(args.target_bias),
            bias_tol_rel=BIAS_TOL_REL,
            w_lifetime=0.35,
            w_bias_exact=160.0,
            feasibility_bonus=18.0,
            w_fit=2.0,
        )
    return original_rewards.RewardConfig(
        name="original_exact_target_strict",
        variant="exact_target",
        target_bias=float(args.target_bias),
        bias_tol_rel=BIAS_TOL_REL,
        w_lifetime=0.25,
        w_bias_exact=180.0,
        feasibility_bonus=0.0,
        w_fit=2.0,
    )


def two_stage_config(args: argparse.Namespace) -> TwoStageRewardConfig:
    return TwoStageRewardConfig(
        target_bias=float(args.target_bias),
        bias_tol_rel=BIAS_TOL_REL,
    )


def build_simulation(args: argparse.Namespace) -> tuple[SimulationConfig, SimulationConfig, int, int]:
    if args.quick:
        proxy_cfg = SimulationConfig(na=8, nb=3, t_final_x=1.0, t_final_z=90.0, n_points=22)
        final_cfg = SimulationConfig(na=8, nb=3, t_final_x=1.0, t_final_z=90.0, n_points=24)
        epochs = args.epochs or 8
        population = args.population or 4
    else:
        proxy_cfg = SimulationConfig(na=10, nb=3, t_final_x=1.0, t_final_z=130.0, n_points=28)
        final_cfg = SimulationConfig(na=15, nb=5, t_final_x=1.2, t_final_z=260.0, n_points=60)
        epochs = args.epochs or 34
        population = args.population or 8
    return proxy_cfg, final_cfg, int(epochs), int(population)


def evaluate_x(
    x: np.ndarray,
    sim_cfg: SimulationConfig,
    reward_cfg,
    reward_fn: Callable[[dict, object], dict],
) -> dict:
    x = np.minimum(np.maximum(np.asarray(x, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
    g2, eps_d = params_to_complex(x)
    metrics = measure_lifetimes(g2, eps_d, sim_cfg)
    reward = reward_fn(metrics, reward_cfg)
    row = {
        "x": x,
        "g2": g2,
        "eps_d": eps_d,
        "reward": float(reward["reward"]),
        "loss_to_minimize": float(reward["loss_to_minimize"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_error": float(reward.get("bias_error", np.inf)),
        "bias_rel_error": float(reward.get("bias_rel_error", np.inf)),
        "stage": str(reward.get("stage", "")),
    }
    for key in (
        "T_X",
        "T_Z",
        "bias",
        "geo_lifetime",
        "alpha_abs",
        "nbar",
        "fit_penalty",
        "fit_x_r2",
        "fit_z_r2",
        "valid",
    ):
        row[key] = metrics[key]
    return row


def select_better(candidate: dict, incumbent: dict, mode: str) -> bool:
    c_feasible = bool(candidate["is_feasible"])
    i_feasible = bool(incumbent["is_feasible"])
    if c_feasible and not i_feasible:
        return True
    if i_feasible and not c_feasible:
        return False
    if c_feasible and i_feasible:
        if mode == "two_stage":
            return float(candidate["reward"]) > float(incumbent["reward"])
        c_error = float(candidate["bias_error"])
        i_error = float(incumbent["bias_error"])
        one_percent_log = math.log(1.01)
        if abs(c_error - i_error) > one_percent_log:
            return c_error < i_error
        return float(candidate["geo_lifetime"]) > float(incumbent["geo_lifetime"])
    return float(candidate["reward"]) > float(incumbent["reward"])


def run_method(
    *,
    method: str,
    sim_cfg: SimulationConfig,
    reward_cfg,
    reward_fn: Callable[[dict, object], dict],
    epochs: int,
    population: int,
    sigma0: float,
    seed: int,
) -> tuple[list[dict], dict]:
    clear_measure_cache()
    optimizer = SepCMA(
        mean=OPTIMIZATION_START_X.copy(),
        sigma=float(sigma0),
        bounds=BOUNDS,
        population_size=int(population),
        seed=int(seed),
    )
    start_eval = evaluate_x(OPTIMIZATION_START_X.copy(), sim_cfg, reward_cfg, reward_fn)
    incumbent = dict(start_eval)
    history = [
        history_row(method, 0, start_eval, start_eval, incumbent, optimizer.mean, 0.0, 0.0)
    ]
    total_eval_seconds = 0.0
    start = time.time()
    for epoch in range(1, epochs + 1):
        xs = [np.asarray(optimizer.ask(), dtype=float) for _ in range(optimizer.population_size)]
        if len(xs) >= 2:
            xs[0] = np.asarray(incumbent["x"], dtype=float)
            xs[1] = OPTIMIZATION_START_X.copy()

        eval_start = time.time()
        evaluated = [evaluate_x(x, sim_cfg, reward_cfg, reward_fn) for x in xs]
        epoch_eval_seconds = time.time() - eval_start
        total_eval_seconds += epoch_eval_seconds

        optimizer.tell([(np.asarray(item["x"], dtype=float), float(item["loss_to_minimize"])) for item in evaluated])
        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        for item in evaluated:
            if select_better(item, incumbent, "two_stage" if method == "two_stage" else "baseline"):
                incumbent = dict(item)

        mean_eval = evaluate_x(np.asarray(optimizer.mean, dtype=float), sim_cfg, reward_cfg, reward_fn)
        row = history_row(
            method,
            epoch,
            epoch_best,
            mean_eval,
            incumbent,
            optimizer.mean,
            epoch_eval_seconds,
            total_eval_seconds,
        )
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(
                f"{method} epoch={epoch:03d}/{epochs} "
                f"Tx={row['incumbent_T_X']:.4g} Tz={row['incumbent_T_Z']:.4g} "
                f"eta={row['incumbent_bias']:.4g} reward={row['incumbent_reward']:.3f} "
                f"wall={time.time() - start:.1f}s",
                flush=True,
            )
    return history, incumbent


def history_row(
    method: str,
    epoch: int,
    epoch_best: dict,
    mean_eval: dict,
    incumbent: dict,
    optimizer_mean: np.ndarray,
    epoch_eval_seconds: float,
    total_eval_seconds: float,
) -> dict:
    row = {
        "method": method,
        "epoch": int(epoch),
        "epoch_eval_seconds": float(epoch_eval_seconds),
        "cumulative_eval_seconds": float(total_eval_seconds),
    }
    for prefix, item in (("epoch_best", epoch_best), ("mean", mean_eval), ("incumbent", incumbent)):
        row[f"{prefix}_reward"] = float(item["reward"])
        row[f"{prefix}_loss"] = float(item["loss_to_minimize"])
        row[f"{prefix}_T_X"] = float(item["T_X"])
        row[f"{prefix}_T_Z"] = float(item["T_Z"])
        row[f"{prefix}_bias"] = float(item["bias"])
        row[f"{prefix}_geo_lifetime"] = float(item["geo_lifetime"])
        row[f"{prefix}_bias_error"] = float(item["bias_error"])
        row[f"{prefix}_fit_penalty"] = float(item["fit_penalty"])
        row[f"{prefix}_is_feasible"] = int(bool(item["is_feasible"]))
        row[f"{prefix}_stage"] = str(item.get("stage", ""))
        x = np.asarray(item["x"], dtype=float)
        row[f"{prefix}_g2_real"] = float(x[0])
        row[f"{prefix}_g2_imag"] = float(x[1])
        row[f"{prefix}_eps_d_real"] = float(x[2])
        row[f"{prefix}_eps_d_imag"] = float(x[3])
    for i, value in enumerate(np.asarray(optimizer_mean, dtype=float)):
        row[f"optimizer_mean_x{i}"] = float(value)
    return row


def validate_final(method: str, incumbent: dict, final_cfg: SimulationConfig, reward_cfg, reward_fn) -> dict:
    evaluated = evaluate_x(np.asarray(incumbent["x"], dtype=float), final_cfg, reward_cfg, reward_fn)
    return summary_row(method, evaluated)


def summary_row(method: str, item: dict) -> dict:
    x = np.asarray(item["x"], dtype=float)
    return {
        "method": method,
        "g2_real": float(x[0]),
        "g2_imag": float(x[1]),
        "eps_d_real": float(x[2]),
        "eps_d_imag": float(x[3]),
        "T_X": float(item["T_X"]),
        "T_Z": float(item["T_Z"]),
        "eta": float(item["bias"]),
        "geo_lifetime": float(item["geo_lifetime"]),
        "reward": float(item["reward"]),
        "target_achieved": int(bool(item["is_feasible"])),
        "bias_error": float(item["bias_error"]),
        "fit_penalty": float(item["fit_penalty"]),
    }


def plot_comparison(history: list[dict], target_bias: float) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    _set_style()
    methods = list(dict.fromkeys(str(row["method"]) for row in history))

    fig, ax = _styled_figure("Validated eta")
    for method in methods:
        rows = [row for row in history if row["method"] == method]
        style = _method_style(method)
        ax.plot(
            _series(rows, "epoch"),
            _series(rows, "incumbent_bias"),
            color=style["color"],
            lw=4.2,
            marker=style["marker"],
            markevery=_marker_stride(rows),
            ms=8,
            mfc="white",
            mec=style["color"],
            mew=2.4,
            label=_pretty_method(method),
        )
    ax.axhline(target_bias, color="#c44e52", lw=2.3, ls="--", label="target")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\eta = T_Z/T_X$")
    _finish_axis(ax)
    _save(fig, FIGURES / "eta_vs_epoch_comparison.png")

    fig, ax = _styled_figure("Validated lifetimes")
    for method in methods:
        rows = [row for row in history if row["method"] == method]
        method_ls = "-" if method == "two_stage" else "--"
        method_alpha = 1.0 if method == "two_stage" else 0.70
        label_prefix = _pretty_method(method)
        epochs = _series(rows, "epoch")
        ax.plot(
            epochs,
            _series(rows, "incumbent_T_X"),
            color="#008b8b",
            lw=4.0,
            ls=method_ls,
            alpha=method_alpha,
            marker="o",
            markevery=_marker_stride(rows),
            ms=8,
            mfc="white",
            mec="#008b8b",
            mew=2.4,
            label=rf"{label_prefix} $T_X$",
        )
        ax.plot(
            epochs,
            _series(rows, "incumbent_T_Z"),
            color="#6f3db8",
            lw=4.0,
            ls=method_ls,
            alpha=method_alpha,
            marker="s",
            markevery=_marker_stride(rows),
            ms=8,
            mfc="white",
            mec="#6f3db8",
            mew=2.4,
            label=rf"{label_prefix} $T_Z$",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lifetime (us)")
    ax.set_yscale("log")
    _apply_lifetime_ticks(ax, history)
    _finish_axis(ax)
    _save(fig, FIGURES / "lifetimes_vs_epoch_comparison.png")

    fig, ax = _styled_figure("Reward convergence")
    for method in methods:
        rows = [row for row in history if row["method"] == method]
        style = _method_style(method)
        ax.plot(
            _series(rows, "epoch"),
            _series(rows, "incumbent_reward"),
            color=style["color"],
            lw=4.2,
            marker=style["marker"],
            markevery=_marker_stride(rows),
            ms=8,
            mfc="white",
            mec=style["color"],
            mew=2.4,
            label=_pretty_method(method),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Method reward")
    _finish_axis(ax)
    _save(fig, FIGURES / "reward_vs_epoch_comparison.png")

    fig, ax = _styled_figure("Runtime convergence")
    for method in methods:
        rows = [row for row in history if row["method"] == method]
        style = _method_style(method)
        ax.plot(
            _series(rows, "epoch"),
            _series(rows, "cumulative_eval_seconds"),
            color=style["color"],
            lw=4.2,
            marker=style["marker"],
            markevery=_marker_stride(rows),
            ms=8,
            mfc="white",
            mec=style["color"],
            mew=2.4,
            label=_pretty_method(method),
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cumulative evaluation seconds")
    _finish_axis(ax)
    _save(fig, FIGURES / "runtime_vs_epoch_comparison.png")


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.labelsize": 24,
            "axes.titlesize": 32,
            "axes.titleweight": "bold",
            "legend.fontsize": 18,
            "figure.dpi": 200,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#273238",
            "axes.edgecolor": "#273238",
            "xtick.color": "#273238",
            "ytick.color": "#273238",
            "text.color": "#273238",
            "font.family": "DejaVu Sans",
        }
    )


def _styled_figure(title: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(1, 1, figsize=(13.5, 7.1), dpi=200)
    ax.set_title(title, loc="left", pad=24)
    return fig, ax


def _finish_axis(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="#d7dde5", lw=1.5, alpha=0.78)
    ax.grid(False, which="minor")
    ax.tick_params(axis="both", which="major", labelsize=21, length=9, width=2.0, pad=10)
    ax.tick_params(axis="both", which="minor", length=5, width=1.4)
    ax.spines["left"].set_linewidth(2.2)
    ax.spines["bottom"].set_linewidth(2.2)
    ax.legend(
        loc="center right",
        frameon=True,
        fancybox=False,
        facecolor="white",
        edgecolor="#dfe3e8",
        framealpha=0.95,
        borderpad=0.55,
        handlelength=2.0,
    )


def _apply_lifetime_ticks(ax: plt.Axes, history: list[dict]) -> None:
    values = []
    for row in history:
        values.extend([float(row["incumbent_T_X"]), float(row["incumbent_T_Z"])])
    positive = [value for value in values if np.isfinite(value) and value > 0]
    if not positive:
        return
    base_ticks = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    lo = min(positive)
    hi = max(positive)
    ticks = base_ticks[(base_ticks >= lo * 0.65) & (base_ticks <= hi * 1.35)]
    if len(ticks):
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{tick:g}" for tick in ticks])


def _method_style(method: str) -> dict:
    if method == "two_stage":
        return {"color": "#6f3db8", "marker": "s"}
    return {"color": "#008b8b", "marker": "o"}


def _pretty_method(method: str) -> str:
    return "Two-stage" if method == "two_stage" else "Baseline"


def _series(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def _marker_stride(rows: list[dict]) -> int:
    return max(1, len(rows) // 10)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def read_history_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_readme_summary(summary_rows: list[dict], args: argparse.Namespace) -> None:
    lines = [
        "# Two-stage reward comparison",
        "",
        "This folder compares the original `team-core-bias-optimizer_sannino` reward against a two-stage reward.",
        "The simulator and optimizer setup are matched; only the reward and feasible-candidate selection differ.",
        "",
        "## Run",
        "",
        "```bash",
        "python run_two_stage_comparison.py --quick",
        "python run_two_stage_comparison.py",
        "```",
        "",
        "## Current summary",
        "",
        "| method | T_X | T_Z | eta | geo lifetime | target achieved | reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['T_X']:.4g} | {row['T_Z']:.4g} | "
            f"{row['eta']:.4g} | {row['geo_lifetime']:.4g} | "
            f"{bool(row['target_achieved'])} | {row['reward']:.4g} |"
        )
    lines += [
        "",
        "Artifacts are written to `results/` and `figures/`.",
        f"Last command options: `quick={args.quick}`, `epochs={args.epochs}`, `population={args.population}`, `seed={args.seed}`.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        history_path = RESULTS / "epoch_history_comparison.csv"
        history = read_history_csv(history_path)
        plot_comparison(history, float(args.target_bias))
        print(f"Regenerated comparison figures from {history_path}")
        return

    original_rewards = load_original_rewards()
    baseline_cfg = original_reward_config(original_rewards, args)
    staged_cfg = two_stage_config(args)
    proxy_cfg, final_cfg, epochs, population = build_simulation(args)

    print(f"Proxy simulation: {asdict(proxy_cfg)}")
    print(f"Final simulation: {asdict(final_cfg)}")
    print(f"epochs={epochs}, population={population}, seed={args.seed}, sigma0={args.sigma0}")

    baseline_history, baseline_incumbent = run_method(
        method="baseline_original",
        sim_cfg=proxy_cfg,
        reward_cfg=baseline_cfg,
        reward_fn=original_rewards.compute_reward,
        epochs=epochs,
        population=population,
        sigma0=float(args.sigma0),
        seed=int(args.seed),
    )
    two_stage_history, two_stage_incumbent = run_method(
        method="two_stage",
        sim_cfg=proxy_cfg,
        reward_cfg=staged_cfg,
        reward_fn=compute_two_stage_reward,
        epochs=epochs,
        population=population,
        sigma0=float(args.sigma0),
        seed=int(args.seed),
    )
    history = baseline_history + two_stage_history
    write_csv(RESULTS / "epoch_history_comparison.csv", history)

    baseline_final = validate_final(
        "baseline_original",
        baseline_incumbent,
        final_cfg,
        baseline_cfg,
        original_rewards.compute_reward,
    )
    two_stage_final = validate_final(
        "two_stage",
        two_stage_incumbent,
        final_cfg,
        staged_cfg,
        compute_two_stage_reward,
    )
    summary_rows = [baseline_final, two_stage_final]
    write_csv(RESULTS / "final_summary_comparison.csv", summary_rows)
    write_json(
        RESULTS / "comparison_config.json",
        {
            "optimizer": OPTIMIZER_NAME,
            "baseline_reward_config": asdict(baseline_cfg),
            "two_stage_reward_config": asdict(staged_cfg),
            "proxy_simulation": asdict(proxy_cfg),
            "final_simulation": asdict(final_cfg),
            "epochs": epochs,
            "population": population,
            "seed": int(args.seed),
            "sigma0": float(args.sigma0),
            "original_project_compared": str(ORIGINAL_ROOT),
        },
    )
    plot_comparison(history, float(args.target_bias))
    write_readme_summary(summary_rows, args)

    print("\nFinal validation comparison")
    for row in summary_rows:
        print(
            f"{row['method']:>18s}: Tx={row['T_X']:.5g} Tz={row['T_Z']:.5g} "
            f"eta={row['eta']:.5g} geo={row['geo_lifetime']:.5g} "
            f"target={bool(row['target_achieved'])} reward={row['reward']:.4g}"
        )
    print(f"\nSaved comparison outputs to {RESULTS} and {FIGURES}")


if __name__ == "__main__":
    main()
