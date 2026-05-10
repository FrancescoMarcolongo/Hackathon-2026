"""Compare SepCMA and BLT-style online tracking under complex g2-gain drift."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from cmaes import SepCMA

from cat_model import SimulationConfig, clear_measure_cache, measure_lifetimes, params_to_complex
from g2_gain_drift import (
    G2GainDriftConfig,
    apply_g2_gain_drift,
    g2_gain_components,
    true_g2_gain_optimum,
    verify_or_scale_g2_gain_path,
)
from plotting import plot_decay_fit, set_style
from rewards import RewardConfig, compute_reward
from run_core_bias_optimization import BIAS_TOL_REL, BOUNDS, TARGET_BIAS
from validation import write_csv, write_json


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CONTROL_SUFFIXES = ("g2_real", "g2_imag", "eps_d_real", "eps_d_imag")
CONTROL_LABELS = [
    ("g2_real", "Re(g2)", "#165a96"),
    ("g2_imag", "Im(g2)", "#1f7a4d"),
    ("eps_d_real", "Re(epsilon_d)", "#8a5a00"),
    ("eps_d_imag", "Im(epsilon_d)", "#7a3f98"),
]
BLT_BASE_STEPS = np.array([0.34, 0.25, 0.58, 0.48], dtype=float)
WARMUP_EPOCH = 18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Run a short smoke test.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--baseline-population", type=int, default=8)
    parser.add_argument("--blt-population", type=int, default=28)
    parser.add_argument("--baseline-sigma0", type=float, default=0.42)
    parser.add_argument("--baseline-sigma-floor", type=float, default=0.045)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--start-seed", type=int, default=17)
    parser.add_argument("--target-bias", type=float, default=TARGET_BIAS)
    parser.add_argument("--warmup-epoch", type=int, default=WARMUP_EPOCH)
    parser.add_argument("--period-epochs", type=float, default=72.0)
    parser.add_argument("--amplitude-scale", type=float, default=1.0)
    parser.add_argument("--reference-signature-weight", type=float, default=80.0)
    parser.add_argument("--blt-initial-trust", type=float, default=0.64)
    parser.add_argument("--blt-min-trust", type=float, default=0.08)
    parser.add_argument("--blt-max-trust", type=float, default=0.68)
    parser.add_argument("--blt-continuity-weight", type=float, default=2.0)
    parser.add_argument(
        "--x-reference",
        type=float,
        nargs=4,
        default=None,
        metavar=("G2_RE", "G2_IM", "EPS_RE", "EPS_IM"),
        help="Stationary no-drift reference. Defaults to the calibrated medium reference when available.",
    )
    parser.add_argument(
        "--sim-preset",
        choices=("quick", "medium", "final"),
        default=None,
        help="Simulation preset. Defaults to quick with --quick and medium otherwise.",
    )
    parser.add_argument("--no-decay-snapshots", action="store_true")
    return parser.parse_args()


def build_simulation_config(args: argparse.Namespace) -> SimulationConfig:
    preset = args.sim_preset or ("quick" if args.quick else "medium")
    if preset == "quick":
        return SimulationConfig(na=8, nb=3, t_final_x=1.0, t_final_z=90.0, n_points=24)
    if preset == "medium":
        return SimulationConfig(na=12, nb=4, t_final_x=1.2, t_final_z=220.0, n_points=36)
    return SimulationConfig(na=15, nb=5, t_final_x=1.2, t_final_z=260.0, n_points=60)


def build_reward_config(target_bias: float) -> RewardConfig:
    return RewardConfig(
        name="g2_gain_reference_signature",
        variant="target_band",
        target_bias=float(target_bias),
        bias_tol_rel=BIAS_TOL_REL,
        w_lifetime=0.35,
        w_bias_exact=160.0,
        feasibility_bonus=18.0,
        w_fit=2.0,
    )


def load_reference_from_existing_results() -> tuple[float, float, float, float] | None:
    summary_path = RESULTS / "storage_detuning_summary_metrics.json"
    if not summary_path.exists():
        return None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        values = data["storage_detuning_config"]["x_reference"]
        if len(values) != 4:
            return None
        return tuple(float(value) for value in values)
    except Exception:
        return None


def build_gain_config(args: argparse.Namespace) -> tuple[G2GainDriftConfig, str]:
    kwargs = {
        "period_epochs": float(args.period_epochs),
        "amplitude_scale": float(args.amplitude_scale),
    }
    if args.x_reference is not None:
        kwargs["x_reference"] = tuple(float(value) for value in args.x_reference)
        source = "command_line"
    else:
        reference = load_reference_from_existing_results()
        if reference is not None:
            kwargs["x_reference"] = reference
            source = "results/storage_detuning_summary_metrics.json"
        else:
            source = "G2GainDriftConfig default"
    return G2GainDriftConfig(**kwargs), source


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("dynamiqs", "jax", "cmaes", "scipy", "matplotlib", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return versions


def reference_signature_penalty(
    metrics: dict[str, object],
    reference_metrics: dict[str, object],
    *,
    weight: float,
) -> float:
    if weight <= 0:
        return 0.0
    penalty = 0.0
    for key in ("T_X", "T_Z", "alpha_abs"):
        value = float(metrics.get(key, np.nan))
        reference = float(reference_metrics.get(key, np.nan))
        if not np.isfinite(value) or not np.isfinite(reference) or value <= 0 or reference <= 0:
            return 1.0e6
        penalty += math.log(value / reference) ** 2
    return float(weight * penalty)


def evaluate_x_gain(
    x_cmd: np.ndarray,
    *,
    epoch: int,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    gain_cfg: G2GainDriftConfig,
    reference_metrics: dict[str, object],
    reference_signature_weight: float,
) -> dict:
    state = apply_g2_gain_drift(x_cmd, epoch, gain_cfg, BOUNDS)
    x_eff = np.asarray(state["x_eff"], dtype=float)
    g2_eff, eps_eff = params_to_complex(x_eff)
    metrics = measure_lifetimes(g2_eff, eps_eff, sim_cfg)
    reward = dict(compute_reward(metrics, reward_cfg))
    signature_penalty = reference_signature_penalty(
        metrics,
        reference_metrics,
        weight=float(reference_signature_weight),
    )
    reward["reward"] = float(reward["reward"]) - signature_penalty
    reward["loss_to_minimize"] = -float(reward["reward"])
    row = {
        "x": np.asarray(state["x_cmd_clipped"], dtype=float),
        "x_cmd": np.asarray(state["x_cmd_clipped"], dtype=float),
        "x_eff": x_eff,
        "x_true_opt": np.asarray(state["x_true_opt"], dtype=float),
        "x_reference_eff": np.asarray(state["x_reference_eff"], dtype=float),
        "tracking_error": np.asarray(state["tracking_error"], dtype=float),
        "effective_error": np.asarray(state["effective_error"], dtype=float),
        "command_offset": np.asarray(state["command_offset"], dtype=float),
        "gain_real": float(state["gain_real"]),
        "gain_imag": float(state["gain_imag"]),
        "gain_amplitude_delta": float(state["gain_amplitude_delta"]),
        "gain_amplitude_factor": float(state["gain_amplitude_factor"]),
        "gain_phase_rad": float(state["gain_phase_rad"]),
        "reward": float(reward["reward"]),
        "loss_to_minimize": float(reward["loss_to_minimize"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_shortfall": float(reward["bias_shortfall"]),
        "bias_error": float(reward["bias_error"]),
        "bias_rel_error": float(reward["bias_rel_error"]),
        "reference_signature_penalty": float(signature_penalty),
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
        "fit_ok",
        "valid",
    ):
        row[key] = metrics[key]
    return row


def random_initial_command(
    *,
    seed: int,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    gain_cfg: G2GainDriftConfig,
    reference_metrics: dict[str, object],
    reference_signature_weight: float,
    target_bias: float,
) -> tuple[np.ndarray, dict, int]:
    rng = np.random.default_rng(int(seed))
    low = np.array([0.35, -0.85, 1.0, -2.0], dtype=float)
    high = np.array([2.45, 0.85, 5.8, 2.0], dtype=float)
    fallback: tuple[np.ndarray, dict, int] | None = None
    for attempt in range(1, 200 + 1):
        x = rng.uniform(low, high)
        evaluated = evaluate_x_gain(
            x,
            epoch=0,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            gain_cfg=gain_cfg,
            reference_metrics=reference_metrics,
            reference_signature_weight=reference_signature_weight,
        )
        if not bool(evaluated.get("valid", False)):
            continue
        if fallback is None:
            fallback = (x, evaluated, attempt)
        bias = float(evaluated["bias"])
        if not (target_bias * (1.0 - BIAS_TOL_REL) <= bias <= target_bias * (1.0 + BIAS_TOL_REL)):
            return x, evaluated, attempt
    if fallback is not None:
        return fallback
    raise RuntimeError("Could not find a valid seeded random initial command.")


def optimizer_effective_sigma(opt: SepCMA) -> float:
    try:
        opt._eigen_decomposition()
    except Exception:
        pass
    sigma = float(getattr(opt, "_sigma", np.nan))
    diagonal_raw = getattr(opt, "_D", None)
    if diagonal_raw is None:
        return sigma
    diagonal = np.asarray(diagonal_raw, dtype=float)
    if diagonal.size == 0 or not np.all(np.isfinite(diagonal)):
        return sigma
    return float(sigma * np.min(diagonal))


def maybe_reheat_optimizer(
    opt: SepCMA,
    *,
    population: int,
    seed: int,
    epoch: int,
    sigma_floor: float,
) -> tuple[SepCMA, bool]:
    if sigma_floor <= 0:
        return opt, False
    effective_sigma = optimizer_effective_sigma(opt)
    if np.isfinite(effective_sigma) and effective_sigma >= sigma_floor and not opt.should_stop():
        return opt, False
    mean = np.minimum(np.maximum(np.asarray(opt.mean, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
    return (
        SepCMA(
            mean=mean,
            sigma=float(sigma_floor),
            bounds=BOUNDS,
            population_size=int(population),
            seed=int(seed + 300_000 + epoch),
        ),
        True,
    )


def run_sepcma_baseline(
    *,
    x0: np.ndarray,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    gain_cfg: G2GainDriftConfig,
    reference_metrics: dict[str, object],
    reference_signature_weight: float,
    epochs: int,
    population: int,
    sigma0: float,
    sigma_floor: float,
    seed: int,
) -> tuple[list[dict], list[dict], dict]:
    clear_measure_cache()
    optimizer = SepCMA(
        mean=np.minimum(np.maximum(np.asarray(x0, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1]),
        sigma=float(sigma0),
        bounds=BOUNDS,
        population_size=int(population),
        seed=int(seed),
    )
    history: list[dict] = []
    samples: list[dict] = []
    previous_epoch_best_x = np.asarray(x0, dtype=float).copy()
    reheats = 0
    start = time.time()
    start_eval = evaluate_x_gain(
        x0,
        epoch=0,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        gain_cfg=gain_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=reference_signature_weight,
    )
    history.append(
        _history_row(
            0,
            "sepcma_baseline",
            command_eval=start_eval,
            epoch_best=start_eval,
            population_stats=_population_stats([start_eval]),
            seed=seed,
            update_type="initial_random_command",
            optimizer_sigma=optimizer_effective_sigma(optimizer),
            trust_radius=np.nan,
            reheated=False,
        )
    )
    samples.append(_sample_row(0, 0, "initial_random_command", "sepcma_baseline", start_eval))

    for epoch in range(1, int(epochs) + 1):
        xs = [np.asarray(optimizer.ask(), dtype=float) for _ in range(int(population))]
        roles = ["sample"] * len(xs)
        if xs:
            xs[0] = np.asarray(optimizer.mean, dtype=float)
            roles[0] = "optimizer_mean"
        if len(xs) >= 2:
            xs[1] = np.asarray(previous_epoch_best_x, dtype=float)
            roles[1] = "previous_epoch_best"
        evaluated = [
            evaluate_x_gain(
                x,
                epoch=epoch,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                gain_cfg=gain_cfg,
                reference_metrics=reference_metrics,
                reference_signature_weight=reference_signature_weight,
            )
            for x in xs
        ]
        for idx, (role, item) in enumerate(zip(roles, evaluated)):
            samples.append(_sample_row(epoch, idx, role, "sepcma_baseline", item))
        optimizer.tell([(np.asarray(item["x"], dtype=float), float(item["loss_to_minimize"])) for item in evaluated])
        optimizer, reheated = maybe_reheat_optimizer(
            optimizer,
            population=population,
            seed=seed,
            epoch=epoch,
            sigma_floor=sigma_floor,
        )
        if reheated:
            reheats += 1
        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        previous_epoch_best_x = np.asarray(epoch_best["x"], dtype=float)
        mean_eval = evaluate_x_gain(
            np.asarray(optimizer.mean, dtype=float),
            epoch=epoch,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            gain_cfg=gain_cfg,
            reference_metrics=reference_metrics,
            reference_signature_weight=reference_signature_weight,
        )
        history.append(
            _history_row(
                epoch,
                "sepcma_baseline",
                command_eval=mean_eval,
                epoch_best=epoch_best,
                population_stats=_population_stats(evaluated),
                seed=seed,
                update_type="sepcma_mean",
                optimizer_sigma=optimizer_effective_sigma(optimizer),
                trust_radius=np.nan,
                reheated=reheated,
            )
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = history[-1]
            print(
                f"g2-gain SepCMA epoch={epoch:03d}/{epochs} "
                f"bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"track_l2={row['tracking_error_l2']:.4g} "
                f"sigma_eff={row['optimizer_effective_sigma']:.4g} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return history, samples, {
        "optimizer": "SepCMA",
        "initial_command": np.asarray(x0, dtype=float).tolist(),
        "population": int(population),
        "sigma0": float(sigma0),
        "sigma_floor": float(sigma_floor),
        "seed": int(seed),
        "reheat_count": int(reheats),
    }


def run_blt_tracker(
    *,
    x0: np.ndarray,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    gain_cfg: G2GainDriftConfig,
    reference_metrics: dict[str, object],
    reference_signature_weight: float,
    epochs: int,
    population: int,
    seed: int,
    initial_trust: float,
    min_trust: float,
    max_trust: float,
    continuity_weight: float,
) -> tuple[list[dict], list[dict], dict]:
    clear_measure_cache()
    rng = np.random.default_rng(int(seed + 10_000))
    current = np.minimum(np.maximum(np.asarray(x0, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
    previous = current.copy()
    archive: list[dict] = []
    trust = float(initial_trust)
    history: list[dict] = []
    samples: list[dict] = []
    start = time.time()
    start_eval = evaluate_x_gain(
        current,
        epoch=0,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        gain_cfg=gain_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=reference_signature_weight,
    )
    archive.append(start_eval)
    history.append(
        _history_row(
            0,
            "blt_optimized",
            command_eval=start_eval,
            epoch_best=start_eval,
            population_stats=_population_stats([start_eval]),
            seed=seed,
            update_type="initial_random_command",
            optimizer_sigma=np.nan,
            trust_radius=trust,
            reheated=False,
        )
    )
    samples.append(_sample_row(0, 0, "initial_random_command", "blt_optimized", start_eval))

    for epoch in range(1, int(epochs) + 1):
        current_eval = evaluate_x_gain(
            current,
            epoch=epoch,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            gain_cfg=gain_cfg,
            reference_metrics=reference_metrics,
            reference_signature_weight=reference_signature_weight,
        )
        candidates, roles = _blt_candidates(
            current=current,
            previous=previous,
            archive=archive,
            trust_radius=trust,
            population=int(population),
            rng=rng,
            epoch=epoch,
        )
        evaluated = [
            evaluate_x_gain(
                candidate,
                epoch=epoch,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                gain_cfg=gain_cfg,
                reference_metrics=reference_metrics,
                reference_signature_weight=reference_signature_weight,
            )
            for candidate in candidates
        ]
        for idx, (role, item) in enumerate(zip(roles, evaluated)):
            samples.append(_sample_row(epoch, idx, role, "blt_optimized", item))
        epoch_best = _select_blt_candidate(
            evaluated,
            current=current,
            continuity_weight=float(continuity_weight),
        )
        archive.extend(evaluated)
        archive = sorted(archive, key=lambda item: float(item["reward"]), reverse=True)[:24]

        improved = (
            float(epoch_best["reward"]) > float(current_eval["reward"]) + 0.02
            or float(epoch_best["bias_rel_error"]) < 0.85 * float(current_eval["bias_rel_error"])
        )
        previous = current.copy()
        current = np.asarray(epoch_best["x"], dtype=float)
        if improved:
            trust = max(float(min_trust), 0.90 * trust)
        else:
            trust = min(float(max_trust), 1.08 * trust)

        command_eval = evaluate_x_gain(
            current,
            epoch=epoch,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            gain_cfg=gain_cfg,
            reference_metrics=reference_metrics,
            reference_signature_weight=reference_signature_weight,
        )
        history.append(
            _history_row(
                epoch,
                "blt_optimized",
                command_eval=command_eval,
                epoch_best=epoch_best,
                population_stats=_population_stats(evaluated),
                seed=seed,
                update_type="blt_trust_region_scan",
                optimizer_sigma=np.nan,
                trust_radius=trust,
                reheated=False,
            )
        )
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            row = history[-1]
            print(
                f"g2-gain BLT epoch={epoch:03d}/{epochs} "
                f"bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"track_l2={row['tracking_error_l2']:.4g} "
                f"trust={trust:.3g} elapsed={time.time() - start:.1f}s",
                flush=True,
            )
    return history, samples, {
        "optimizer": "BLT_TRUST_REGION_COORDINATE_SCAN",
        "initial_command": np.asarray(x0, dtype=float).tolist(),
        "population": int(population),
        "seed": int(seed),
        "initial_trust": float(initial_trust),
        "min_trust": float(min_trust),
        "max_trust": float(max_trust),
        "continuity_weight": float(continuity_weight),
        "notes": "No x_reference, gain, drift vector, or true optimum is injected as a candidate.",
    }


def _select_blt_candidate(
    evaluated: list[dict],
    *,
    current: np.ndarray,
    continuity_weight: float,
) -> dict:
    current = np.asarray(current, dtype=float)

    def score(item: dict) -> float:
        move = np.asarray(item["x"], dtype=float) - current
        return float(item["reward"]) - float(continuity_weight) * float(move @ move)

    return max(evaluated, key=score)


def _blt_candidates(
    *,
    current: np.ndarray,
    previous: np.ndarray,
    archive: list[dict],
    trust_radius: float,
    population: int,
    rng: np.random.Generator,
    epoch: int,
) -> tuple[list[np.ndarray], list[str]]:
    candidates: list[np.ndarray] = []
    roles: list[str] = []

    def add(x: np.ndarray, role: str) -> None:
        clipped = np.minimum(np.maximum(np.asarray(x, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
        if not any(np.allclose(clipped, existing, atol=1e-10, rtol=0.0) for existing in candidates):
            candidates.append(clipped)
            roles.append(role)

    current = np.asarray(current, dtype=float)
    previous = np.asarray(previous, dtype=float)
    add(current, "current_command")
    add(current + 0.75 * (current - previous), "momentum_prediction")

    steps = float(trust_radius) * BLT_BASE_STEPS
    for dim, step in enumerate(steps):
        for sign in (-1.0, 1.0):
            trial = current.copy()
            trial[dim] += sign * float(step)
            add(trial, f"coordinate_{dim}_{'plus' if sign > 0 else 'minus'}")

    for item in archive[: min(5, len(archive))]:
        elite = np.asarray(item["x"], dtype=float)
        add(elite, "archive_elite")
        add(0.70 * current + 0.30 * elite, "archive_recombine")

    global_probability = max(0.08, 0.35 * math.exp(-float(epoch) / 24.0))
    low = np.array([0.25, -1.0, 0.50, -3.0], dtype=float)
    high = np.array([3.0, 1.0, 8.0, 3.0], dtype=float)
    while len(candidates) < max(int(population), 4):
        if rng.random() < global_probability:
            trial = rng.uniform(low, high)
            role = "global_random"
        else:
            trial = current + rng.normal(0.0, steps)
            role = "trust_region_random"
        add(trial, role)
    return candidates[:population], roles[:population]


def _population_stats(evaluated: Iterable[dict]) -> dict:
    data = list(evaluated)
    reward = np.array([float(item["reward"]) for item in data], dtype=float)
    bias = np.array([float(item["bias"]) for item in data], dtype=float)
    return {
        "population_reward_mean": _finite_stat(np.nanmean, reward),
        "population_reward_std": _finite_stat(np.nanstd, reward),
        "population_bias_mean": _finite_stat(np.nanmean, bias),
        "population_bias_std": _finite_stat(np.nanstd, bias),
    }


def _finite_stat(func, values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(func(finite))


def _flatten_vector(row: dict, prefix: str, values: np.ndarray) -> None:
    for suffix, value in zip(CONTROL_SUFFIXES, np.asarray(values, dtype=float)):
        row[f"{prefix}_{suffix}"] = float(value)


def _history_row(
    epoch: int,
    algorithm: str,
    *,
    command_eval: dict,
    epoch_best: dict,
    population_stats: dict,
    seed: int,
    update_type: str,
    optimizer_sigma: float,
    trust_radius: float,
    reheated: bool,
) -> dict:
    row: dict[str, object] = {
        "epoch": int(epoch),
        "algorithm": algorithm,
        "update_type": update_type,
        "reward": float(command_eval["reward"]),
        "loss": float(command_eval["loss_to_minimize"]),
        "T_X": float(command_eval["T_X"]),
        "T_Z": float(command_eval["T_Z"]),
        "bias": float(command_eval["bias"]),
        "geo_lifetime": float(command_eval["geo_lifetime"]),
        "is_feasible": int(bool(command_eval["is_feasible"])),
        "bias_error": float(command_eval["bias_error"]),
        "bias_rel_error": float(command_eval["bias_rel_error"]),
        "fit_penalty": float(command_eval["fit_penalty"]),
        "fit_x_r2": float(command_eval["fit_x_r2"]),
        "fit_z_r2": float(command_eval["fit_z_r2"]),
        "reference_signature_penalty": float(command_eval["reference_signature_penalty"]),
        "epoch_best_reward": float(epoch_best["reward"]),
        "epoch_best_loss": float(epoch_best["loss_to_minimize"]),
        "epoch_best_T_X": float(epoch_best["T_X"]),
        "epoch_best_T_Z": float(epoch_best["T_Z"]),
        "epoch_best_bias": float(epoch_best["bias"]),
        "epoch_best_is_feasible": int(bool(epoch_best["is_feasible"])),
        "population_reward_mean": float(population_stats["population_reward_mean"]),
        "population_reward_std": float(population_stats["population_reward_std"]),
        "population_bias_mean": float(population_stats["population_bias_mean"]),
        "population_bias_std": float(population_stats["population_bias_std"]),
        "seed": int(seed),
        "optimizer_effective_sigma": float(optimizer_sigma),
        "trust_radius": float(trust_radius),
        "reheated": int(bool(reheated)),
        "gain_real": float(command_eval["gain_real"]),
        "gain_imag": float(command_eval["gain_imag"]),
        "gain_amplitude_delta": float(command_eval["gain_amplitude_delta"]),
        "gain_amplitude_factor": float(command_eval["gain_amplitude_factor"]),
        "gain_phase_rad": float(command_eval["gain_phase_rad"]),
    }
    _flatten_vector(row, "command", np.asarray(command_eval["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(command_eval["x_eff"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(command_eval["x_true_opt"], dtype=float))
    _flatten_vector(row, "reference", np.asarray(command_eval["x_reference_eff"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(command_eval["tracking_error"], dtype=float))
    _flatten_vector(row, "effective_error", np.asarray(command_eval["effective_error"], dtype=float))
    _flatten_vector(row, "drift", np.asarray(command_eval["command_offset"], dtype=float))
    _flatten_vector(row, "epoch_best_command", np.asarray(epoch_best["x_cmd"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(command_eval["tracking_error"], dtype=float)))
    row["effective_error_l2"] = float(np.linalg.norm(np.asarray(command_eval["effective_error"], dtype=float)))
    return row


def _sample_row(epoch: int, candidate_id: int, role: str, algorithm: str, item: dict) -> dict:
    row: dict[str, object] = {
        "epoch": int(epoch),
        "candidate_id": int(candidate_id),
        "algorithm": algorithm,
        "role": role,
        "reward": float(item["reward"]),
        "loss": float(item["loss_to_minimize"]),
        "T_X": float(item["T_X"]),
        "T_Z": float(item["T_Z"]),
        "bias": float(item["bias"]),
        "is_feasible": int(bool(item["is_feasible"])),
        "fit_penalty": float(item["fit_penalty"]),
        "fit_x_r2": float(item["fit_x_r2"]),
        "fit_z_r2": float(item["fit_z_r2"]),
        "gain_amplitude_factor": float(item["gain_amplitude_factor"]),
        "gain_phase_rad": float(item["gain_phase_rad"]),
    }
    _flatten_vector(row, "command", np.asarray(item["x_cmd"], dtype=float))
    _flatten_vector(row, "effective", np.asarray(item["x_eff"], dtype=float))
    _flatten_vector(row, "true_opt", np.asarray(item["x_true_opt"], dtype=float))
    _flatten_vector(row, "tracking_error", np.asarray(item["tracking_error"], dtype=float))
    row["tracking_error_l2"] = float(np.linalg.norm(np.asarray(item["tracking_error"], dtype=float)))
    return row


def compute_summary_metrics(
    history: list[dict],
    *,
    warmup_epoch: int,
    target_bias: float,
    bias_tol_rel: float,
) -> dict:
    post = [row for row in history if int(row["epoch"]) >= int(warmup_epoch)]
    if not post:
        post = list(history)
    bias = np.array([float(row["bias"]) for row in post], dtype=float)
    tx = np.array([float(row["T_X"]) for row in post], dtype=float)
    tz = np.array([float(row["T_Z"]) for row in post], dtype=float)
    tracking = np.array([float(row["tracking_error_l2"]) for row in post], dtype=float)
    effective = np.array([float(row["effective_error_l2"]) for row in post], dtype=float)
    reward = np.array([float(row["reward"]) for row in post], dtype=float)
    fit_penalty = np.array([float(row["fit_penalty"]) for row in post], dtype=float)
    fit_x = np.array([float(row["fit_x_r2"]) for row in post], dtype=float)
    fit_z = np.array([float(row["fit_z_r2"]) for row in post], dtype=float)
    lower = target_bias * (1.0 - bias_tol_rel)
    upper = target_bias * (1.0 + bias_tol_rel)
    in_band = (bias >= lower) & (bias <= upper)
    final = history[-1]
    return {
        "final_bias": float(final["bias"]),
        "median_bias_after_warmup": _finite_median(bias),
        "fraction_post_warmup_in_success_band": float(np.mean(in_band[np.isfinite(bias)])) if np.any(np.isfinite(bias)) else float("nan"),
        "success_band": [float(lower), float(upper)],
        "median_tracking_error_l2_after_warmup": _finite_median(tracking),
        "final_tracking_error_l2": float(final["tracking_error_l2"]),
        "median_effective_error_l2_after_warmup": _finite_median(effective),
        "final_effective_error_l2": float(final["effective_error_l2"]),
        "median_reward_after_warmup": _finite_median(reward),
        "median_T_X_after_warmup": _finite_median(tx),
        "median_T_Z_after_warmup": _finite_median(tz),
        "median_fit_penalty_after_warmup": _finite_median(fit_penalty),
        "median_fit_x_r2_after_warmup": _finite_median(fit_x),
        "median_fit_z_r2_after_warmup": _finite_median(fit_z),
        "final_fit_penalty": float(final["fit_penalty"]),
        "final_fit_x_r2": float(final["fit_x_r2"]),
        "final_fit_z_r2": float(final["fit_z_r2"]),
    }


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float("nan")
    return float(np.median(finite))


def _series(rows: list[dict], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def _epochs(rows: list[dict]) -> np.ndarray:
    return np.array([int(row["epoch"]) for row in rows], dtype=int)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_comparison_family(
    baseline: list[dict],
    blt: list[dict],
    *,
    target_bias: float,
    bias_tol_rel: float,
    warmup_epoch: int,
    prefix: str,
) -> dict[str, str]:
    set_style()
    figures: dict[str, str] = {}
    datasets = [("SepCMA baseline", baseline, "#165a96"), ("BLT optimized", blt, "#c04f15")]

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        ax.plot(_epochs(rows), _series(rows, "bias"), lw=2.0, color=color, label=label)
    ax.axhspan(target_bias * (1 - bias_tol_rel), target_bias * (1 + bias_tol_rel), color="#165a96", alpha=0.10, label="success band")
    ax.axhline(target_bias, color="#1f1f1f", ls="--", lw=1.1, label="target")
    ax.axvline(warmup_epoch, color="#5f5f5f", ls=":", lw=1.0, label="warm-up")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"Bias $\eta$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_bias_vs_epoch.png"
    _save(fig, path)
    figures["bias_vs_epoch"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        ax.plot(_epochs(rows), _series(rows, "T_X"), lw=1.8, color=color, ls="-", label=f"{label} T_X")
        ax.plot(_epochs(rows), _series(rows, "T_Z"), lw=1.8, color=color, ls="--", label=f"{label} T_Z")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lifetime (us)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    path = FIGURES / f"{prefix}_lifetimes_vs_epoch.png"
    _save(fig, path)
    figures["lifetimes_vs_epoch"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        ax.plot(_epochs(rows), _series(rows, "reward"), lw=2.0, color=color, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward (higher is better)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_reward_or_loss_vs_epoch.png"
    _save(fig, path)
    figures["reward_or_loss_vs_epoch"] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=200, sharex=True)
    for ax, (suffix, label, color) in zip(axes.ravel(), CONTROL_LABELS):
        ax.plot(_epochs(baseline), _series(baseline, f"command_{suffix}"), color="#165a96", lw=1.8, label="SepCMA command")
        ax.plot(_epochs(blt), _series(blt, f"command_{suffix}"), color="#c04f15", lw=1.8, label="BLT command")
        ax.plot(_epochs(blt), _series(blt, f"true_opt_{suffix}"), color="#1f1f1f", lw=1.4, ls="--", label="true optimum")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Command")
    axes[1, 0].set_ylabel("Command")
    axes[0, 0].legend(frameon=False, fontsize=8)
    path = FIGURES / f"{prefix}_parameters_vs_epoch.png"
    _save(fig, path)
    figures["parameters_vs_epoch"] = str(path)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), dpi=200, sharex=True)
    epochs = _epochs(blt)
    axes[0].plot(epochs, _series(blt, "gain_amplitude_delta"), color="#165a96", lw=2.0, label="gain amplitude delta")
    axes[0].axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.4)
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, _series(blt, "gain_phase_rad"), color="#c04f15", lw=2.0, label="gain phase drift")
    axes[1].axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.4)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Phase (rad)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)
    path = FIGURES / f"{prefix}_signal_vs_epoch.png"
    _save(fig, path)
    figures["signal_vs_epoch"] = str(path)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4), dpi=200)
    for label, rows, color in datasets:
        ax.plot(_epochs(rows), _series(rows, "tracking_error_l2"), color=color, lw=2.0, label=f"{label} command L2")
        ax.plot(_epochs(rows), _series(rows, "effective_error_l2"), color=color, lw=1.4, ls="--", label=f"{label} effective L2")
    ax.axhline(0.0, color="#1f1f1f", lw=0.8, alpha=0.4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    path = FIGURES / f"{prefix}_tracking_error_vs_epoch.png"
    _save(fig, path)
    figures["tracking_error_vs_epoch"] = str(path)

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), dpi=200, sharex=True)
    for ax, (suffix, label, _color) in zip(axes.ravel(), CONTROL_LABELS):
        ax.plot(_epochs(baseline), _series(baseline, f"effective_{suffix}"), color="#165a96", lw=1.8, label="SepCMA effective")
        ax.plot(_epochs(blt), _series(blt, f"effective_{suffix}"), color="#c04f15", lw=1.8, label="BLT effective")
        ax.plot(_epochs(blt), _series(blt, f"reference_{suffix}"), color="#1f1f1f", lw=1.4, ls="--", label="reference")
        ax.set_title(label)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Effective")
    axes[1, 0].set_ylabel("Effective")
    axes[0, 0].legend(frameon=False, fontsize=8)
    path = FIGURES / f"{prefix}_effective_parameters_vs_epoch.png"
    _save(fig, path)
    figures["effective_parameters_vs_epoch"] = str(path)
    return figures


def write_decay_snapshot(rows: list[dict], sim_cfg: SimulationConfig, path: Path, title: str) -> dict[str, str]:
    row = rows[-1]
    x_eff = np.array([row[f"effective_{suffix}"] for suffix in CONTROL_SUFFIXES], dtype=float)
    g2, eps_d = params_to_complex(x_eff)
    result = measure_lifetimes(g2, eps_d, sim_cfg, return_curves=True)
    plot_decay_fit(result, path, title)
    return {path.stem: str(path)}


def write_report(
    path: Path,
    *,
    command: str,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    gain_cfg: G2GainDriftConfig,
    reference_source: str,
    start_info: dict,
    bounds_check: dict,
    baseline_summary: dict,
    blt_summary: dict,
    baseline_metadata: dict,
    blt_metadata: dict,
    figures: dict[str, str],
) -> None:
    lines = [
        "# Complex g2-gain drift: SepCMA vs BLT",
        "",
        "## Reproduction command",
        "",
        f"```bash\n{command}\n```",
        "",
        "## Drift model",
        "",
        "This model is the challenge notebook's suggested amplitude drift in the buffer/two-photon control chain, implemented as a complex transfer function on the driven controls:",
        "",
        "```text",
        "G(e) = (1 + r(e)) exp(i theta(e))",
        "g2_eff(e) = G(e) g2_cmd(e)",
        "epsilon_d_eff(e) = G(e) epsilon_d_cmd(e)",
        "r(e) = A_r [0.65 sin(2 pi e/P + 0.15) + 0.35 sin(4 pi e/P + 1.40)]",
        "theta(e) = A_phi [0.60 sin(2 pi e/P + 1.10) + 0.40 sin(4 pi e/P + 2.65)]",
        "x_opt_true(e) = [Re(g2_ref/G(e)), Im(g2_ref/G(e)), Re(epsilon_ref/G(e)), Im(epsilon_ref/G(e))]",
        "```",
        "",
        f"- P: `{gain_cfg.period_epochs:g}` epochs",
        f"- highest harmonic period: `{gain_cfg.period_epochs / gain_cfg.bandwidth:g}` epochs",
        f"- amplitude modulation A_r: `{gain_cfg.amplitude_modulation:g}`",
        f"- phase modulation A_phi: `{gain_cfg.phase_modulation:g}` rad",
        f"- x_reference source: `{reference_source}`",
        "",
        "Both optimizers start from the same seeded random valid command. The shared reward is the target-band lifetime reward plus a stationary reference-signature penalty on `T_X`, `T_Z`, and `alpha_abs`. The BLT candidate generator does not receive `G(e)`, `x_ref`, or `x_opt_true(e)` as candidate inputs; those are used by the simulator and by the offline error plots.",
        "",
        "## Random start",
        "",
        "```json",
        json.dumps(start_info, indent=2),
        "```",
        "",
        "## Results after warm-up",
        "",
        "| metric | SepCMA baseline | BLT optimized |",
        "|---|---:|---:|",
        f"| final bias | {baseline_summary['final_bias']:.6g} | {blt_summary['final_bias']:.6g} |",
        f"| median bias | {baseline_summary['median_bias_after_warmup']:.6g} | {blt_summary['median_bias_after_warmup']:.6g} |",
        f"| success-band fraction | {baseline_summary['fraction_post_warmup_in_success_band']:.4g} | {blt_summary['fraction_post_warmup_in_success_band']:.4g} |",
        f"| median tracking L2 | {baseline_summary['median_tracking_error_l2_after_warmup']:.6g} | {blt_summary['median_tracking_error_l2_after_warmup']:.6g} |",
        f"| median effective L2 | {baseline_summary['median_effective_error_l2_after_warmup']:.6g} | {blt_summary['median_effective_error_l2_after_warmup']:.6g} |",
        f"| median T_X (us) | {baseline_summary['median_T_X_after_warmup']:.6g} | {blt_summary['median_T_X_after_warmup']:.6g} |",
        f"| median T_Z (us) | {baseline_summary['median_T_Z_after_warmup']:.6g} | {blt_summary['median_T_Z_after_warmup']:.6g} |",
        "",
        "Interpretation: lower tracking/effective error means the commanded controls better compensate the drifting complex gain and keep the physical Hamiltonian near the stationary cat optimum.",
        "",
        "## Figures",
        "",
    ]
    for label, fig_path in figures.items():
        lines.append(f"- {label}: `{fig_path}`")
    lines += [
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(
            {
                "simulation": asdict(sim_cfg),
                "reward": asdict(reward_cfg),
                "g2_gain_drift": asdict(gain_cfg),
                "bounds_check": bounds_check,
                "baseline_metadata": baseline_metadata,
                "blt_metadata": blt_metadata,
                "package_versions": package_versions(),
            },
            indent=2,
        ),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    epochs = int(args.epochs if args.epochs is not None else (10 if args.quick else 72))
    sim_cfg = build_simulation_config(args)
    reward_cfg = build_reward_config(float(args.target_bias))
    gain_cfg_raw, reference_source = build_gain_config(args)
    gain_cfg, bounds_check = verify_or_scale_g2_gain_path(gain_cfg_raw, BOUNDS, epochs)
    x_ref = np.asarray(gain_cfg.x_reference, dtype=float)
    ref_g2, ref_eps = params_to_complex(x_ref)
    reference_metrics = measure_lifetimes(ref_g2, ref_eps, sim_cfg)
    x0, x0_eval, start_attempt = random_initial_command(
        seed=int(args.start_seed),
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        gain_cfg=gain_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=float(args.reference_signature_weight),
        target_bias=float(args.target_bias),
    )
    start_info = {
        "start_seed": int(args.start_seed),
        "attempt": int(start_attempt),
        "x0": np.asarray(x0, dtype=float).tolist(),
        "bias_at_epoch0": float(x0_eval["bias"]),
        "T_X_at_epoch0": float(x0_eval["T_X"]),
        "T_Z_at_epoch0": float(x0_eval["T_Z"]),
        "tracking_error_l2_at_epoch0": float(np.linalg.norm(np.asarray(x0_eval["tracking_error"], dtype=float))),
    }

    baseline_history, baseline_samples, baseline_metadata = run_sepcma_baseline(
        x0=x0,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        gain_cfg=gain_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=float(args.reference_signature_weight),
        epochs=epochs,
        population=int(args.baseline_population if not args.quick else min(args.baseline_population, 4)),
        sigma0=float(args.baseline_sigma0),
        sigma_floor=float(args.baseline_sigma_floor),
        seed=int(args.seed),
    )
    blt_history, blt_samples, blt_metadata = run_blt_tracker(
        x0=x0,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        gain_cfg=gain_cfg,
        reference_metrics=reference_metrics,
        reference_signature_weight=float(args.reference_signature_weight),
        epochs=epochs,
        population=int(args.blt_population if not args.quick else min(args.blt_population, 8)),
        seed=int(args.seed),
        initial_trust=float(args.blt_initial_trust),
        min_trust=float(args.blt_min_trust),
        max_trust=float(args.blt_max_trust),
        continuity_weight=float(args.blt_continuity_weight),
    )
    baseline_summary = compute_summary_metrics(
        baseline_history,
        warmup_epoch=int(args.warmup_epoch),
        target_bias=float(args.target_bias),
        bias_tol_rel=BIAS_TOL_REL,
    )
    blt_summary = compute_summary_metrics(
        blt_history,
        warmup_epoch=int(args.warmup_epoch),
        target_bias=float(args.target_bias),
        bias_tol_rel=BIAS_TOL_REL,
    )
    summary = {
        "target_bias": float(args.target_bias),
        "warmup_epoch": int(args.warmup_epoch),
        "epochs": epochs,
        "reference_source": reference_source,
        "start_info": start_info,
        "baseline": baseline_summary,
        "blt": blt_summary,
        "simulation": asdict(sim_cfg),
        "reward_config": asdict(reward_cfg),
        "g2_gain_drift_config": asdict(gain_cfg),
        "bounds_check": bounds_check,
        "reference_metrics": reference_metrics,
        "reference_signature_weight": float(args.reference_signature_weight),
        "baseline_metadata": baseline_metadata,
        "blt_metadata": blt_metadata,
    }

    prefix = "g2_gain_drift_comparison"
    write_csv(RESULTS / f"{prefix}_baseline_history.csv", baseline_history)
    write_csv(RESULTS / f"{prefix}_blt_history.csv", blt_history)
    write_csv(RESULTS / f"{prefix}_combined_history.csv", baseline_history + blt_history)
    write_csv(RESULTS / f"{prefix}_baseline_samples.csv", baseline_samples)
    write_csv(RESULTS / f"{prefix}_blt_samples.csv", blt_samples)
    write_json(RESULTS / f"{prefix}_summary_metrics.json", summary)
    write_json(
        RESULTS / f"{prefix}_final_candidates.json",
        {
            "baseline_final": baseline_history[-1],
            "blt_final": blt_history[-1],
            "true_optimum_final": true_g2_gain_optimum(epochs, gain_cfg).tolist(),
            "gain_components_final": g2_gain_components(epochs, gain_cfg),
        },
    )
    figures = plot_comparison_family(
        baseline_history,
        blt_history,
        target_bias=float(args.target_bias),
        bias_tol_rel=BIAS_TOL_REL,
        warmup_epoch=int(args.warmup_epoch),
        prefix=prefix,
    )
    if not args.no_decay_snapshots:
        figures.update(
            write_decay_snapshot(
                baseline_history,
                sim_cfg,
                FIGURES / f"{prefix}_baseline_decay_fits_final.png",
                "Complex g2-gain drift SepCMA final decay fits",
            )
        )
        figures.update(
            write_decay_snapshot(
                blt_history,
                sim_cfg,
                FIGURES / f"{prefix}_blt_decay_fits_final.png",
                "Complex g2-gain drift BLT final decay fits",
            )
        )

    command = "python run_g2_gain_drift_comparison.py"
    if args.quick:
        command += " --quick"
    if args.epochs is not None:
        command += f" --epochs {args.epochs}"
    if args.baseline_population != 8:
        command += f" --baseline-population {args.baseline_population}"
    if args.blt_population != 28:
        command += f" --blt-population {args.blt_population}"
    if args.sim_preset is not None:
        command += f" --sim-preset {args.sim_preset}"
    if args.period_epochs != 72.0:
        command += f" --period-epochs {args.period_epochs}"
    if args.reference_signature_weight != 80.0:
        command += f" --reference-signature-weight {args.reference_signature_weight}"
    if args.blt_continuity_weight != 2.0:
        command += f" --blt-continuity-weight {args.blt_continuity_weight}"
    if args.no_decay_snapshots:
        command += " --no-decay-snapshots"
    write_report(
        ROOT / "g2_gain_drift_comparison_report.md",
        command=command,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        gain_cfg=gain_cfg,
        reference_source=reference_source,
        start_info=start_info,
        bounds_check=bounds_check,
        baseline_summary=baseline_summary,
        blt_summary=blt_summary,
        baseline_metadata=baseline_metadata,
        blt_metadata=blt_metadata,
        figures=figures,
    )
    print("\nComplex g2-gain drift comparison summary")
    print(f"SepCMA final bias={baseline_summary['final_bias']:.6g}")
    print(f"BLT final bias={blt_summary['final_bias']:.6g}")
    print(f"SepCMA median tracking L2={baseline_summary['median_tracking_error_l2_after_warmup']:.6g}")
    print(f"BLT median tracking L2={blt_summary['median_tracking_error_l2_after_warmup']:.6g}")
    print(f"Saved report to {ROOT / 'g2_gain_drift_comparison_report.md'}")


if __name__ == "__main__":
    main()
