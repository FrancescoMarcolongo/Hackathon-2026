"""Drift-aware variant of the static team-core cat-qubit bias optimizer.

This script keeps the static baseline's physical evaluation path:
`cat_model.measure_lifetimes(...)` performs the Dynamiqs Lindblad simulations,
fits T_X and T_Z, and `rewards.compute_reward(...)` computes the challenge
reward.  The only conceptual addition is external epoch drift.

The optimizer state is the controllable 4-vector

    x_control = [Re(g2), Im(g2), Re(eps_d), Im(eps_d)].

At epoch e the physical Hamiltonian is evaluated at

    x_eff(e) = x_control(e) - drift(e)

when the default drift sign is -1.  Therefore the ideal compensation trajectory
is `x_control_target(e) = x_static_ref + drift(e)`, so that the effective
Hamiltonian remains near `x_static_ref`.  Drift periods are measured in
optimization epochs, not in Lindblad simulation time.

Run:
    python cat_bias_optimization_with_drift.py --output-dir drift_results
    python plot_cat_bias_drift_results.py --input-dir drift_results
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent
BASELINE_DIR = ROOT / "team-core-bias-optimizer"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

try:
    from cmaes import SepCMA
except Exception:  # pragma: no cover - fallback for non-notebook smoke tests.
    SepCMA = None

from cat_model import SimulationConfig, clear_measure_cache, measure_lifetimes, params_to_complex
from rewards import RewardConfig, compute_reward, default_reward_sweep
from validation import write_csv, write_json


PARAMETER_NAMES = ["re_g2", "im_g2", "re_eps_d", "im_eps_d"]
BOUNDS = np.array(
    [
        [0.25, 3.0],
        [-1.0, 1.0],
        [0.50, 8.0],
        [-3.0, 3.0],
    ],
    dtype=float,
)

CONFIG = {
    "epochs": 30,
    "population": 8,
    "seed": 0,
    "sigma0": 0.45,
    "target_bias": 100.0,
    "bias_tol_rel": 0.03,
    "output_dir": "drift_results/",
    "init_mean": [1.0, 0.0, 2.5, 0.0],
    "static_ref": None,
    "sim": {
        "na": 8,
        "nb": 3,
        "t_final_x": 1.0,
        "t_final_z": 90.0,
        "n_points": 22,
    },
    # Keep this true for the hackathon demo: with 70 epochs, a 24-epoch period
    # gives almost three full oscillations while still leaving many samples per
    # cycle.  Set this false only for slow production-style robustness tests.
    "visible_drift_demo": False,
    "reward_index": 0,
    "clip_effective_to_bounds": True,
    "anchor_incumbent": True,
    "anchor_static_ref": True,
    # For the visible drift demo, always include the ideal compensation
    # x_static_ref + drift(epoch) as one candidate.  This is a diagnostic anchor:
    # it verifies that the reward/physics prefer drift compensation when that
    # candidate is available, and it makes the tracking plots interpretable in
    # short runs.  Disable for a fully blind online optimizer test.
    "anchor_target_control": True,
    "drift": {
        "enabled": True,
        "kind": "fourier",
        "sign": -1.0,
        "seed": 123,
        "num_terms": 1,
        "periods_epochs": [70.0],
        "amplitudes": [0.0, 0.0, 0.0, 0.0],
        "offset": [0.0, 0.0, 0.0, 0.0],
        "linear_per_epoch": [0.0, 0.0, 0.0, 0.0],
        "phases": None,
        "normalize_terms": False,
    },
}

VISIBLE_DRIFT_DEMO = {
    "periods_epochs": [0.0],
    "num_terms": 1,
    "amplitudes": [0.0, 0.0, 0.0, 0.0],
    "normalize_terms": False,
}


def package_versions() -> dict:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("dynamiqs", "jax", "cmaes", "scipy", "matplotlib", "numpy"):
        try:
            module = __import__(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            versions[name] = f"unavailable: {exc}"
    return versions


def make_drift_state(config: dict) -> dict:
    drift_cfg = dict(config["drift"])
    periods = np.asarray(drift_cfg["periods_epochs"], dtype=float)
    amplitudes = np.asarray(drift_cfg["amplitudes"], dtype=float)
    offset = np.asarray(drift_cfg["offset"], dtype=float)
    linear = np.asarray(drift_cfg["linear_per_epoch"], dtype=float)
    num_terms = int(drift_cfg.get("num_terms", len(periods)))
    if len(periods) != num_terms:
        periods = periods[:num_terms]
    if len(periods) != num_terms:
        raise ValueError("drift.periods_epochs must contain num_terms values")
    if amplitudes.shape != (4,) or offset.shape != (4,) or linear.shape != (4,):
        raise ValueError("drift amplitudes/offset/linear_per_epoch must be length-4 vectors")

    phases = drift_cfg.get("phases")
    if phases is None:
        rng = np.random.default_rng(int(drift_cfg["seed"]))
        phases_arr = rng.uniform(0.0, 2.0 * np.pi, size=(4, num_terms))
    else:
        phases_arr = np.asarray(phases, dtype=float)
        if phases_arr.shape != (4, num_terms):
            raise ValueError("drift.phases must have shape [4, num_terms]")

    per_term_amplitudes = amplitudes[:, None]
    if bool(drift_cfg.get("normalize_terms", True)):
        per_term_amplitudes = per_term_amplitudes / float(num_terms)

    return {
        "enabled": bool(drift_cfg.get("enabled", True)),
        "sign": float(drift_cfg.get("sign", -1.0)),
        "periods": periods,
        "amplitudes": amplitudes,
        "per_term_amplitudes": per_term_amplitudes,
        "offset": offset,
        "linear": linear,
        "phases": phases_arr,
        "config": drift_cfg,
    }


def fourier_drift(epoch: int | float, drift_state: dict) -> np.ndarray:
    if not drift_state["enabled"]:
        return np.zeros(4, dtype=float)
    e = float(epoch)
    drift = drift_state["offset"] + drift_state["linear"] * e
    for m, period in enumerate(drift_state["periods"]):
        drift += drift_state["per_term_amplitudes"][:, 0] * np.sin(
            2.0 * np.pi * e / float(period) + drift_state["phases"][:, m]
        )
    return np.asarray(drift, dtype=float)


def apply_drift(x_control: np.ndarray, drift: np.ndarray, sign: float = -1.0) -> np.ndarray:
    return np.asarray(x_control, dtype=float) + float(sign) * np.asarray(drift, dtype=float)


def clip_to_bounds(x: np.ndarray) -> tuple[np.ndarray, bool]:
    clipped = np.minimum(np.maximum(np.asarray(x, dtype=float), BOUNDS[:, 0]), BOUNDS[:, 1])
    return clipped, bool(np.max(np.abs(clipped - np.asarray(x, dtype=float))) > 1e-12)


def select_better(candidate: dict, incumbent: dict, reward_cfg: RewardConfig) -> bool:
    """Same incumbent-selection semantics as the static baseline."""
    if reward_cfg.variant == "lower_bound":
        c_feasible = bool(candidate["is_feasible"] and candidate["bias"] >= reward_cfg.target_bias)
        i_feasible = bool(incumbent["is_feasible"] and incumbent["bias"] >= reward_cfg.target_bias)
        if c_feasible and not i_feasible:
            return True
        if i_feasible and not c_feasible:
            return False
        if c_feasible and i_feasible:
            return float(candidate["geo_lifetime"]) > float(incumbent["geo_lifetime"])
        return float(candidate["reward"]) > float(incumbent["reward"])

    c_feasible = bool(candidate["is_feasible"])
    i_feasible = bool(incumbent["is_feasible"])
    if c_feasible and not i_feasible:
        return True
    if i_feasible and not c_feasible:
        return False
    if c_feasible and i_feasible:
        c_error = float(candidate["bias_error"])
        i_error = float(incumbent["bias_error"])
        if abs(c_error - i_error) > math.log(1.01):
            return c_error < i_error
        return float(candidate["geo_lifetime"]) > float(incumbent["geo_lifetime"])
    return float(candidate["reward"]) > float(incumbent["reward"])


def evaluate_control_x(
    x_control: np.ndarray,
    *,
    epoch: int,
    drift_state: dict,
    sim_cfg: SimulationConfig,
    reward_cfg: RewardConfig,
    clip_effective: bool,
) -> dict:
    x_control, control_clipped = clip_to_bounds(x_control)
    drift = fourier_drift(epoch, drift_state)
    x_eff_raw = apply_drift(x_control, drift, drift_state["sign"])
    x_eff, eff_clipped = clip_to_bounds(x_eff_raw) if clip_effective else (x_eff_raw, False)

    g2_eff, eps_d_eff = params_to_complex(x_eff)
    metrics = measure_lifetimes(g2_eff, eps_d_eff, sim_cfg)
    reward = compute_reward(metrics, reward_cfg)
    row = {
        "x_control": x_control,
        "x_eff": x_eff,
        "x_eff_raw": x_eff_raw,
        "drift": drift,
        "control_clipped": control_clipped,
        "effective_clipped": eff_clipped,
        "g2": g2_eff,
        "eps_d": eps_d_eff,
        "reward": float(reward["reward"]),
        "loss_to_minimize": float(reward["loss_to_minimize"]),
        "is_feasible": bool(reward["is_feasible"]),
        "bias_shortfall": float(reward["bias_shortfall"]),
        "bias_error": float(reward["bias_error"]),
        "bias_rel_error": float(reward["bias_rel_error"]),
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


class RandomElite:
    def __init__(self, mean: np.ndarray, sigma: float, population: int, seed: int):
        self.mean = np.asarray(mean, dtype=float)
        self.sigma = float(sigma)
        self.population_size = int(population)
        self.rng = np.random.default_rng(int(seed))

    def ask(self) -> np.ndarray:
        x = self.rng.normal(self.mean, self.sigma, size=4)
        return np.minimum(np.maximum(x, BOUNDS[:, 0]), BOUNDS[:, 1])

    def tell(self, solutions: list[tuple[np.ndarray, float]]) -> None:
        ordered = sorted(solutions, key=lambda item: float(item[1]))
        elite_n = max(2, int(round(0.35 * len(ordered))))
        elite_x = np.asarray([item[0] for item in ordered[:elite_n]], dtype=float)
        elite_loss = np.asarray([item[1] for item in ordered[:elite_n]], dtype=float)
        weights = elite_loss.max() - elite_loss + 1e-9
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones(elite_n) / elite_n
        self.mean = np.minimum(np.maximum(np.sum(elite_x * weights[:, None], axis=0), BOUNDS[:, 0]), BOUNDS[:, 1])
        self.sigma = float(np.clip(0.98 * self.sigma, 0.04, 1.0))


def make_optimizer(config: dict):
    mean = np.asarray(config["init_mean"], dtype=float)
    if SepCMA is not None:
        return SepCMA(
            mean=mean.copy(),
            sigma=float(config["sigma0"]),
            bounds=BOUNDS,
            population_size=int(config["population"]),
            seed=int(config["seed"]),
        )
    return RandomElite(mean, float(config["sigma0"]), int(config["population"]), int(config["seed"]))


def optimizer_ask(optimizer) -> list[np.ndarray]:
    return [np.asarray(optimizer.ask(), dtype=float) for _ in range(int(optimizer.population_size))]


def optimizer_mean(optimizer) -> np.ndarray:
    return np.asarray(optimizer.mean, dtype=float)


def target_vectors(epoch: int, static_ref: np.ndarray, drift_state: dict) -> tuple[np.ndarray, np.ndarray]:
    drift = fourier_drift(epoch, drift_state)
    return static_ref + drift, static_ref.copy()


def add_vector_columns(row: dict, prefix: str, x: np.ndarray) -> None:
    for name, value in zip(PARAMETER_NAMES, np.asarray(x, dtype=float)):
        row[f"{prefix}_{name}"] = float(value)


def history_row(
    *,
    epoch: int,
    epoch_best: dict,
    mean_eval: dict,
    incumbent: dict,
    optimizer_mean_x: np.ndarray,
    static_ref: np.ndarray,
    drift_state: dict,
    reward_cfg: RewardConfig,
    seed: int,
    sigma0: float,
) -> dict:
    drift = fourier_drift(epoch, drift_state)
    target_control, target_effective = target_vectors(epoch, static_ref, drift_state)
    control = np.asarray(incumbent["x_control"], dtype=float)
    effective = np.asarray(incumbent["x_eff"], dtype=float)
    epoch_best_control = np.asarray(epoch_best["x_control"], dtype=float)
    epoch_best_effective = np.asarray(epoch_best["x_eff"], dtype=float)
    mean_control = optimizer_mean_x
    mean_effective = np.asarray(mean_eval["x_eff"], dtype=float)
    control_error = control - target_control
    effective_error = effective - target_effective
    mean_control_error = mean_control - target_control
    mean_effective_error = mean_effective - target_effective

    row: Dict[str, object] = {
        "epoch": int(epoch),
        "reward": float(incumbent["reward"]),
        "train_reward": float(epoch_best["reward"]),
        "validation_reward": float(mean_eval["reward"]),
        "T_X": float(incumbent["T_X"]),
        "T_Z": float(incumbent["T_Z"]),
        "bias": float(incumbent["bias"]),
        "geo_lifetime": float(incumbent["geo_lifetime"]),
        "fit_penalty": float(incumbent["fit_penalty"]),
        "is_feasible": int(bool(incumbent["is_feasible"])),
        "epoch_best_reward": float(epoch_best["reward"]),
        "epoch_best_T_X": float(epoch_best["T_X"]),
        "epoch_best_T_Z": float(epoch_best["T_Z"]),
        "epoch_best_bias": float(epoch_best["bias"]),
        "mean_reward": float(mean_eval["reward"]),
        "mean_T_X": float(mean_eval["T_X"]),
        "mean_T_Z": float(mean_eval["T_Z"]),
        "mean_bias": float(mean_eval["bias"]),
        "control_error_norm": float(np.linalg.norm(control_error)),
        "effective_error_norm": float(np.linalg.norm(effective_error)),
        "mean_control_error_norm": float(np.linalg.norm(mean_control_error)),
        "mean_effective_error_norm": float(np.linalg.norm(mean_effective_error)),
        "effective_clipped": int(bool(incumbent["effective_clipped"])),
        "control_clipped": int(bool(incumbent["control_clipped"])),
        "reward_config": reward_cfg.name,
        "reward_variant": reward_cfg.variant,
        "sigma0": float(sigma0),
        "seed": int(seed),
    }
    add_vector_columns(row, "control", control)
    add_vector_columns(row, "epoch_best_control", epoch_best_control)
    add_vector_columns(row, "mean_control", mean_control)
    add_vector_columns(row, "drift", drift)
    add_vector_columns(row, "effective", effective)
    add_vector_columns(row, "epoch_best_effective", epoch_best_effective)
    add_vector_columns(row, "mean_effective", mean_effective)
    add_vector_columns(row, "effective_raw", np.asarray(incumbent["x_eff_raw"], dtype=float))
    add_vector_columns(row, "target_control", target_control)
    add_vector_columns(row, "target_effective", target_effective)
    add_vector_columns(row, "abs_error_control", np.abs(control_error))
    add_vector_columns(row, "abs_error_effective", np.abs(effective_error))
    add_vector_columns(row, "abs_error_mean_control", np.abs(mean_control_error))
    add_vector_columns(row, "abs_error_mean_effective", np.abs(mean_effective_error))
    add_vector_columns(row, "optimizer_mean", optimizer_mean_x)
    return row


def build_sim_config(config: dict) -> SimulationConfig:
    return SimulationConfig(**dict(config["sim"]))


def build_reward_config(config: dict) -> RewardConfig:
    reward_cfgs = default_reward_sweep(float(config["target_bias"]), float(config["bias_tol_rel"]))
    return reward_cfgs[int(config.get("reward_index", 0))]


def run(config: dict) -> tuple[list[dict], dict]:
    clear_measure_cache()
    sim_cfg = build_sim_config(config)
    reward_cfg = build_reward_config(config)
    drift_state = make_drift_state(config)
    static_ref = np.asarray(config["static_ref"] if config["static_ref"] is not None else config["init_mean"], dtype=float)

    print(
        "Effective drift config: "
        f"enabled={drift_state['enabled']} sign={drift_state['sign']} "
        f"periods={drift_state['periods'].tolist()} "
        f"amplitudes={drift_state['amplitudes'].tolist()} "
        f"normalize_terms={config['drift'].get('normalize_terms')}"
    )
    print(f"Static reference: {np.array2string(static_ref, precision=4)}")
    if bool(config.get("anchor_target_control", False)):
        print("Diagnostic anchor enabled: sampling target_control = static_ref + drift(epoch).")

    optimizer = make_optimizer(config)
    start_eval = evaluate_control_x(
        np.asarray(config["init_mean"], dtype=float),
        epoch=0,
        drift_state=drift_state,
        sim_cfg=sim_cfg,
        reward_cfg=reward_cfg,
        clip_effective=bool(config["clip_effective_to_bounds"]),
    )
    incumbent = dict(start_eval)
    history = [
        history_row(
            epoch=0,
            epoch_best=start_eval,
            mean_eval=start_eval,
            incumbent=incumbent,
            optimizer_mean_x=optimizer_mean(optimizer),
            static_ref=static_ref,
            drift_state=drift_state,
            reward_cfg=reward_cfg,
            seed=int(config["seed"]),
            sigma0=float(config["sigma0"]),
        )
    ]

    start = time.time()
    for epoch in range(1, int(config["epochs"]) + 1):
        # In the drift setting, yesterday's incumbent control vector has a new
        # physical meaning today.  Re-evaluate it under the current epoch drift
        # before comparing it to newly sampled controls.
        incumbent = evaluate_control_x(
            np.asarray(incumbent["x_control"], dtype=float),
            epoch=epoch,
            drift_state=drift_state,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            clip_effective=bool(config["clip_effective_to_bounds"]),
        )
        xs = optimizer_ask(optimizer)
        if bool(config.get("anchor_incumbent", True)) and len(xs) >= 1:
            xs[0] = np.asarray(incumbent["x_control"], dtype=float)
        if bool(config.get("anchor_target_control", True)) and len(xs) >= 2:
            xs[1] = target_vectors(epoch, static_ref, drift_state)[0]
        elif bool(config.get("anchor_static_ref", True)) and len(xs) >= 2:
            xs[1] = static_ref.copy()

        evaluated = [
            evaluate_control_x(
                x,
                epoch=epoch,
                drift_state=drift_state,
                sim_cfg=sim_cfg,
                reward_cfg=reward_cfg,
                clip_effective=bool(config["clip_effective_to_bounds"]),
            )
            for x in xs
        ]
        optimizer.tell([(np.asarray(item["x_control"], dtype=float), float(item["loss_to_minimize"])) for item in evaluated])

        epoch_best = max(evaluated, key=lambda item: float(item["reward"]))
        for item in evaluated:
            if select_better(item, incumbent, reward_cfg):
                incumbent = dict(item)

        mean_eval = evaluate_control_x(
            optimizer_mean(optimizer),
            epoch=epoch,
            drift_state=drift_state,
            sim_cfg=sim_cfg,
            reward_cfg=reward_cfg,
            clip_effective=bool(config["clip_effective_to_bounds"]),
        )
        row = history_row(
            epoch=epoch,
            epoch_best=epoch_best,
            mean_eval=mean_eval,
            incumbent=incumbent,
            optimizer_mean_x=optimizer_mean(optimizer),
            static_ref=static_ref,
            drift_state=drift_state,
            reward_cfg=reward_cfg,
            seed=int(config["seed"]),
            sigma0=float(config["sigma0"]),
        )
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == int(config["epochs"]):
            print(
                f"drift epoch={epoch:03d}/{config['epochs']} "
                f"incumbent Tx={row['T_X']:.4g} Tz={row['T_Z']:.4g} "
                f"bias={row['bias']:.4g} reward={row['reward']:.3f} "
                f"|control-target|={row['control_error_norm']:.3g} "
                f"|eff-ref|={row['effective_error_norm']:.3g} "
                f"elapsed={time.time() - start:.1f}s",
                flush=True,
            )

    best = {
        "reward": float(incumbent["reward"]),
        "x_control": np.asarray(incumbent["x_control"], dtype=float),
        "x_effective": np.asarray(incumbent["x_eff"], dtype=float),
        "x_effective_raw": np.asarray(incumbent["x_eff_raw"], dtype=float),
        "drift_at_best_epoch": np.asarray(incumbent["drift"], dtype=float),
        "metrics": {k: incumbent[k] for k in ("T_X", "T_Z", "bias", "geo_lifetime", "fit_penalty", "fit_ok", "valid")},
        "reward_config": asdict(reward_cfg),
        "simulation_config": asdict(sim_cfg),
        "static_ref": static_ref,
    }
    return history, best


def write_results(config: dict, history: list[dict], best: dict) -> None:
    out = Path(str(config["output_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "drift_optimization_history.csv", history)
    write_json(
        out / "metadata.json",
        {
            "config": config,
            "drift_config": config["drift"],
            "parameter_names": PARAMETER_NAMES,
            "bounds": BOUNDS,
            "sign_convention": "x_eff = x_control + sign * drift; default sign=-1 gives x_eff=x_control-drift",
            "baseline_source": str(BASELINE_DIR / "run_core_bias_optimization.py"),
            "package_versions": package_versions(),
        },
    )
    write_json(out / "best_candidate.json", best)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--drift-enabled", dest="drift_enabled", action="store_true")
    parser.add_argument("--no-drift", dest="drift_enabled", action="store_false")
    parser.set_defaults(drift_enabled=None)
    parser.add_argument("--drift-periods-epochs", nargs="+", type=float)
    parser.add_argument("--drift-amplitudes", nargs=4, type=float)
    parser.add_argument("--drift-seed", type=int)
    parser.add_argument("--drift-sign", type=float)
    parser.add_argument("--static-ref", nargs=4, type=float)
    parser.add_argument("--quick", action="store_true", help="Tiny smoke-test settings.")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> dict:
    config = dict(CONFIG)
    config["drift"] = dict(CONFIG["drift"])
    config["sim"] = dict(CONFIG["sim"])
    if bool(config.get("visible_drift_demo", False)):
        config["drift"].update(VISIBLE_DRIFT_DEMO)
    for key in ("epochs", "population", "seed"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.drift_enabled is not None:
        config["drift"]["enabled"] = bool(args.drift_enabled)
    if args.drift_periods_epochs:
        config["drift"]["periods_epochs"] = list(args.drift_periods_epochs)
        config["drift"]["num_terms"] = len(args.drift_periods_epochs)
    if args.drift_amplitudes:
        config["drift"]["amplitudes"] = list(args.drift_amplitudes)
    if args.drift_seed is not None:
        config["drift"]["seed"] = int(args.drift_seed)
    if args.drift_sign is not None:
        config["drift"]["sign"] = float(args.drift_sign)
    if args.static_ref:
        config["static_ref"] = list(args.static_ref)
    if args.quick:
        config["epochs"] = args.epochs or 2
        config["population"] = args.population or 3
        config["sim"] = {"na": 8, "nb": 3, "t_final_x": 0.5, "t_final_z": 20.0, "n_points": 10}
    return config


def main() -> None:
    config = config_from_args(parse_args())
    history, best = run(config)
    write_results(config, history, best)
    print(f"\nSaved drift optimization results to {config['output_dir']}")
    print("Next:")
    print(f"  python plot_cat_bias_drift_results.py --input-dir {config['output_dir']}")


if __name__ == "__main__":
    main()
