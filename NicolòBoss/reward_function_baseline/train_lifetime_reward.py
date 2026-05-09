"""Run CMA-ES training for the simple lifetime reward baseline.

Example:
    python reward_function_baseline/train_lifetime_reward.py \
        --generations 5 \
        --population-size 4 \
        --sigma 0.25 \
        --x0 1.0 0.0 4.0 0.0
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(THIS_DIR / ".matplotlib-cache"))

PROJECT_ROOT = THIS_DIR.parent
PROJECT_VENV = PROJECT_ROOT / ".venv"
PROJECT_VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"

if (
    PROJECT_VENV_PYTHON.exists()
    and Path(sys.prefix).resolve() != PROJECT_VENV.resolve()
    and os.environ.get("LIFETIME_REWARD_NO_VENV_REEXEC") != "1"
):
    print(f"Re-running with project virtualenv: {PROJECT_VENV_PYTHON}", flush=True)
    os.execv(str(PROJECT_VENV_PYTHON), [str(PROJECT_VENV_PYTHON), *sys.argv])

import jax.numpy as jnp
import numpy as np
from cmaes import CMA, SepCMA

from simple_lifetime_reward import cat_lifetime_metrics


# Edit this block when you want to run experiments directly from this file.
# Command-line flags still work and override these defaults.
USER_CONFIG = {
    # Optimizer hyperparameters
    "optimizer": "sep-cma",
    "generations": 10,
    "population_size": 5,
    "sigma": 0.1,
    "seed": 0,
    "x0": [1.0, 0.0, 4.0, 0.0],
    "lower_bounds": None,
    "upper_bounds": None,
    # Reward hyperparameters
    "eta_target": 500.0,
    "lambda_bias": 2.0,
    # Simulation/model parameters
    "na": 15,
    "nb": 5,
    "kappa_b": 10.0,
    "kappa_a": 1.0,
    "tfinal_z": 200.0,
    "tfinal_x": 2.0,
    "nsave": 100,
    # Optional drift placeholder. Set either value to something like 0.95
    # or 1.05 to test simple multiplicative drift.
    "g2_prefactor": 1.0,
    "epsd_prefactor": 1.0,
    # Output settings
    "out_dir": None,
    "plot_name": "loss_curve.png",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train cat-qubit knobs with the simple lifetime reward loss.",
    )
    parser.add_argument(
        "--optimizer",
        choices=["sep-cma", "cma"],
        default=USER_CONFIG["optimizer"],
        help="CMA-ES variant to use.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=USER_CONFIG["generations"],
        help="Number of CMA-ES generations to run.",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=USER_CONFIG["population_size"],
        help="Number of candidates evaluated per generation.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=USER_CONFIG["sigma"],
        help="Initial CMA-ES sampling standard deviation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=USER_CONFIG["seed"],
        help="Random seed for CMA-ES.",
    )
    parser.add_argument(
        "--x0",
        nargs=4,
        type=float,
        default=USER_CONFIG["x0"],
        metavar=("RE_G2", "IM_G2", "RE_EPSD", "IM_EPSD"),
        help="Initial knobs: Re(g2) Im(g2) Re(eps_d) Im(eps_d).",
    )
    parser.add_argument(
        "--lower-bounds",
        nargs=4,
        type=float,
        default=USER_CONFIG["lower_bounds"],
        metavar=("RE_G2", "IM_G2", "RE_EPSD", "IM_EPSD"),
        help="Optional lower bounds for the four knobs.",
    )
    parser.add_argument(
        "--upper-bounds",
        nargs=4,
        type=float,
        default=USER_CONFIG["upper_bounds"],
        metavar=("RE_G2", "IM_G2", "RE_EPSD", "IM_EPSD"),
        help="Optional upper bounds for the four knobs.",
    )
    parser.add_argument(
        "--eta-target",
        type=float,
        default=USER_CONFIG["eta_target"],
        help="Target bias eta = Tz / Tx used by the reward.",
    )
    parser.add_argument(
        "--lambda-bias",
        type=float,
        default=USER_CONFIG["lambda_bias"],
        help="Penalty weight for eta-target mismatch.",
    )
    parser.add_argument(
        "--na",
        type=int,
        default=USER_CONFIG["na"],
        help="Storage Hilbert space dimension.",
    )
    parser.add_argument(
        "--nb",
        type=int,
        default=USER_CONFIG["nb"],
        help="Buffer Hilbert space dimension.",
    )
    parser.add_argument(
        "--kappa-b",
        type=float,
        default=USER_CONFIG["kappa_b"],
        help="Buffer dissipation rate.",
    )
    parser.add_argument(
        "--kappa-a",
        type=float,
        default=USER_CONFIG["kappa_a"],
        help="Storage single-photon loss rate.",
    )
    parser.add_argument(
        "--tfinal-z",
        type=float,
        default=USER_CONFIG["tfinal_z"],
        help="Final simulation time for the Z lifetime fit.",
    )
    parser.add_argument(
        "--tfinal-x",
        type=float,
        default=USER_CONFIG["tfinal_x"],
        help="Final simulation time for the X lifetime fit.",
    )
    parser.add_argument(
        "--nsave",
        type=int,
        default=USER_CONFIG["nsave"],
        help="Number of saved time points in each lifetime simulation.",
    )
    parser.add_argument(
        "--g2-prefactor",
        type=float,
        default=USER_CONFIG["g2_prefactor"],
        help="Optional multiplicative drift factor for g2.",
    )
    parser.add_argument(
        "--epsd-prefactor",
        type=float,
        default=USER_CONFIG["epsd_prefactor"],
        help="Optional multiplicative drift factor for eps_d.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=USER_CONFIG["out_dir"],
        help="Output directory for history, summary, and plot.",
    )
    parser.add_argument(
        "--plot-name",
        default=USER_CONFIG["plot_name"],
        help="Filename for the saved loss plot.",
    )
    return parser.parse_args()


def build_bounds(args: argparse.Namespace) -> np.ndarray | None:
    if args.lower_bounds is None and args.upper_bounds is None:
        return None
    if args.lower_bounds is None or args.upper_bounds is None:
        raise ValueError("Pass both --lower-bounds and --upper-bounds, or neither.")

    lower = np.array(args.lower_bounds, dtype=float)
    upper = np.array(args.upper_bounds, dtype=float)
    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be smaller than its upper bound.")

    return np.column_stack([lower, upper])


def make_output_dir(out_dir: Path | str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = THIS_DIR / "runs" / f"simple_lifetime_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_simulation_kwargs(args: argparse.Namespace) -> dict[str, object]:
    drift = {
        "g2_prefactor": args.g2_prefactor,
        "epsd_prefactor": args.epsd_prefactor,
    }

    return {
        "tfinal_z": args.tfinal_z,
        "tfinal_x": args.tfinal_x,
        "na": args.na,
        "nb": args.nb,
        "kappa_b": args.kappa_b,
        "kappa_a": args.kappa_a,
        "nsave": args.nsave,
        "drift": drift,
    }


def evaluate_candidate(
    x: np.ndarray,
    eta_target: float,
    lambda_bias: float,
    simulation_kwargs: dict[str, object],
) -> dict[str, float | list[float]]:
    try:
        metrics = cat_lifetime_metrics(
            jnp.array(x),
            eta_target=eta_target,
            lambda_bias=lambda_bias,
            **simulation_kwargs,
        )

        return {
            "loss": metrics["loss"],
            "reward": metrics["reward"],
            "Tx": metrics["Tx"],
            "Tz": metrics["Tz"],
            "eta": metrics["eta"],
            "failed": False,
        }
    except Exception as exc:
        print("candidate failed:", repr(exc))
        return {
            "loss": 1e6,
            "reward": -1e6,
            "Tx": 0.0,
            "Tz": 0.0,
            "eta": 0.0,
            "failed": True,
        }


def write_history_csv(history: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "generation",
        "candidate",
        "evaluation",
        "loss",
        "best_loss_so_far",
        "reward",
        "Tx",
        "best_Tx_so_far",
        "Tz",
        "best_Tz_so_far",
        "eta",
        "best_eta_so_far",
        "x0",
        "x1",
        "x2",
        "x3",
        "failed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            x = row["x"]
            writer.writerow(
                {
                    "generation": row["generation"],
                    "candidate": row["candidate"],
                    "evaluation": row["evaluation"],
                    "loss": row["loss"],
                    "best_loss_so_far": row["best_loss_so_far"],
                    "reward": row["reward"],
                    "Tx": row["Tx"],
                    "best_Tx_so_far": row["best_Tx_so_far"],
                    "Tz": row["Tz"],
                    "best_Tz_so_far": row["best_Tz_so_far"],
                    "eta": row["eta"],
                    "best_eta_so_far": row["best_eta_so_far"],
                    "x0": x[0],
                    "x1": x[1],
                    "x2": x[2],
                    "x3": x[3],
                    "failed": row["failed"],
                }
            )


def plot_loss(history: list[dict[str, object]], path: Path, eta_target: float) -> None:
    from matplotlib import pyplot as plt

    evaluations = np.array([row["evaluation"] for row in history], dtype=int)
    losses = np.array([row["loss"] for row in history], dtype=float)
    best_losses = np.array([row["best_loss_so_far"] for row in history], dtype=float)
    etas = np.array([row["eta"] for row in history], dtype=float)
    best_etas = np.array([row["best_eta_so_far"] for row in history], dtype=float)
    txs = np.array([row["Tx"] for row in history], dtype=float)
    best_txs = np.array([row["best_Tx_so_far"] for row in history], dtype=float)
    tzs = np.array([row["Tz"] for row in history], dtype=float)
    best_tzs = np.array([row["best_Tz_so_far"] for row in history], dtype=float)

    fig, axes = plt.subplots(4, 1, figsize=(8.5, 11), sharex=True)

    ax = axes[0]
    ax.plot(evaluations, losses, marker="o", linewidth=1.0, alpha=0.45, label="loss")
    ax.plot(
        evaluations,
        best_losses,
        marker="o",
        linewidth=2.0,
        color="#b84a62",
        label="best loss so far",
    )
    ax.set_ylabel("Loss")
    ax.set_title("Simple lifetime reward training")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    ax.plot(evaluations, etas, marker="o", linewidth=1.0, alpha=0.45, label="eta")
    ax.plot(
        evaluations,
        best_etas,
        marker="o",
        linewidth=2.0,
        color="#2f6f73",
        label="eta at best loss so far",
    )
    ax.axhline(
        eta_target,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label="eta target",
    )
    ax.set_ylabel("Eta")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[2]
    ax.plot(evaluations, txs, marker="o", linewidth=1.0, alpha=0.45, label="Tx")
    ax.plot(
        evaluations,
        best_txs,
        marker="o",
        linewidth=2.0,
        color="#7a5cbd",
        label="Tx at best loss so far",
    )
    ax.set_ylabel("Tx")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[3]
    ax.plot(evaluations, tzs, marker="o", linewidth=1.0, alpha=0.45, label="Tz")
    ax.plot(
        evaluations,
        best_tzs,
        marker="o",
        linewidth=2.0,
        color="#b87333",
        label="Tz at best loss so far",
    )
    ax.set_xlabel("Evaluation")
    ax.set_ylabel("Tz")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = make_output_dir(args.out_dir)
    bounds = build_bounds(args)
    simulation_kwargs = build_simulation_kwargs(args)
    mean = np.array(args.x0, dtype=float)

    optimizer_cls = SepCMA if args.optimizer == "sep-cma" else CMA
    optimizer = optimizer_cls(
        mean=mean,
        sigma=args.sigma,
        bounds=bounds,
        seed=args.seed,
        population_size=args.population_size,
    )

    config = vars(args).copy()
    config["out_dir"] = str(out_dir)
    config["bounds"] = None if bounds is None else bounds.tolist()
    config["simulation_kwargs"] = simulation_kwargs
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    history: list[dict[str, object]] = []
    best_loss = float("inf")
    best_x = mean.copy()
    best_Tx = float("nan")
    best_Tz = float("nan")
    best_eta = float("nan")
    evaluation = 0

    print("Starting training")
    print("output directory:", out_dir)
    print("initial x:", mean.tolist())

    for generation in range(args.generations):
        solutions = []
        generation_losses = []

        for candidate in range(args.population_size):
            x = optimizer.ask()
            metrics = evaluate_candidate(
                x,
                args.eta_target,
                args.lambda_bias,
                simulation_kwargs,
            )
            loss = float(metrics["loss"])
            solutions.append((x, loss))
            generation_losses.append(loss)

            evaluation += 1
            if loss < best_loss:
                best_loss = loss
                best_x = x.copy()
                best_Tx = float(metrics["Tx"])
                best_Tz = float(metrics["Tz"])
                best_eta = float(metrics["eta"])

            row = {
                "generation": generation,
                "candidate": candidate,
                "evaluation": evaluation,
                "x": x.tolist(),
                "loss": loss,
                "best_loss_so_far": best_loss,
                "reward": float(metrics["reward"]),
                "Tx": float(metrics["Tx"]),
                "best_Tx_so_far": best_Tx,
                "Tz": float(metrics["Tz"]),
                "best_Tz_so_far": best_Tz,
                "eta": float(metrics["eta"]),
                "best_eta_so_far": best_eta,
                "failed": bool(metrics["failed"]),
            }
            history.append(row)

            print(
                "gen",
                generation,
                "cand",
                candidate,
                "loss",
                f"{loss:.6g}",
                "Tx",
                f"{row['Tx']:.6g}",
                "Tz",
                f"{row['Tz']:.6g}",
                "eta",
                f"{row['eta']:.6g}",
                "best",
                f"{best_loss:.6g}",
            )

        optimizer.tell(solutions)
        print(
            "generation",
            generation,
            "done: mean loss",
            f"{np.mean(generation_losses):.6g}",
            "best loss",
            f"{best_loss:.6g}",
        )

    csv_path = out_dir / "history.csv"
    json_path = out_dir / "summary.json"
    plot_path = out_dir / args.plot_name

    write_history_csv(history, csv_path)
    plot_loss(history, plot_path, args.eta_target)

    summary = {
        "best_loss": best_loss,
        "best_x": best_x.tolist(),
        "best_Tx": best_Tx,
        "best_Tz": best_Tz,
        "best_eta": best_eta,
        "evaluations": evaluation,
        "history_csv": str(csv_path),
        "loss_plot": str(plot_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))

    print("Training complete")
    print("best loss:", best_loss)
    print("best x:", best_x.tolist())
    print("best Tx:", best_Tx)
    print("best Tz:", best_Tz)
    print("best eta:", best_eta)
    print("history:", csv_path)
    print("summary:", json_path)
    print("loss plot:", plot_path)


if __name__ == "__main__":
    main()
