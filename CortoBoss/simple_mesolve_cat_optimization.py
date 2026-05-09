"""
Simple mesolve-based cat-qubit bias optimization.

Run in the notebook environment:
    python simple_mesolve_cat_optimization.py

The reward is computed directly from fitted logical decay lifetimes T_X and T_Z.
No RK4 implementation and no parity-as-X shortcut are used here.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import dynamiqs as dq
import jax.numpy as jnp
import numpy as np

try:
    from cmaes import SepCMA
except ImportError:
    SepCMA = None


CONFIG = {
    "epochs": 60,
    "population": 12,
    "seed": 0,
    "output_dir": "simple_mesolve_results/",
    "na": 8,
    "nb": 3,
    "bias_target": 100.0,
    "init_mean": [1.0, 0.0, 4.0, 0.0],
    "init_sigma": 0.2,
    "bounds": [
        [0.15, 2.0],
        [-0.8, 0.8],
        [0.5, 8.0],
        [-3.0, 3.0],
    ],
    "times": [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40],
}


KAPPA_B = 10.0
KAPPA_A = 1.0
LAMBDA_X = 0.5
LAMBDA_Z = 0.5
PENALTY_WEIGHT = 5.0
ELITE_FRACTION = 0.35
EPS = 1e-9


def as_float_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def params_to_complex(x: np.ndarray) -> Tuple[complex, complex]:
    return complex(float(x[0]), float(x[1])), complex(float(x[2]), float(x[3]))


def estimate_alpha(g2: complex, eps_d: complex, kappa_b: float = KAPPA_B, kappa_a: float = KAPPA_A) -> float:
    """Same adiabatic-elimination estimate used in the challenge notebook."""
    kappa_2 = 4.0 * abs(g2) ** 2 / kappa_b
    if kappa_2 < EPS:
        return 0.2
    eps_2 = 2.0 * g2 * eps_d / kappa_b
    alpha_sq = 2.0 * max(abs(eps_2) - kappa_a / 4.0, 0.02) / kappa_2
    return float(math.sqrt(alpha_sq))


def build_system(x: np.ndarray, config: Dict[str, object]):
    na, nb = int(config["na"]), int(config["nb"])
    g2, eps_d = params_to_complex(x)
    alpha = estimate_alpha(g2, eps_d)

    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    H = (
        np.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - np.conj(eps_d) * b
    )
    c_ops = [jnp.sqrt(KAPPA_B) * b, jnp.sqrt(KAPPA_A) * a]

    return a, b, H, c_ops, alpha


def logical_states_and_ops(alpha: float, config: Dict[str, object]):
    """Approximate coherent-lobe logical convention.

    |0_L> = |+alpha>, |1_L> = |-alpha>
    X_L ~= |+alpha><-alpha| + |-alpha><+alpha|
    Z_L ~= |+alpha><+alpha| - |-alpha><-alpha|

    This X_L is not photon-number parity.  It is a direct lobe-basis logical
    coherence observable, used only because we are in simulation.
    """
    na, nb = int(config["na"]), int(config["nb"])
    g = dq.coherent(na, alpha)
    e = dq.coherent(na, -alpha)
    overlap = float(np.exp(-2.0 * alpha**2))
    plus_x = (g + e) / jnp.sqrt(2.0 + 2.0 * overlap)

    x_storage = g @ e.dag() + e @ g.dag()
    z_storage = g @ g.dag() - e @ e.dag()
    x_op = dq.tensor(x_storage, dq.eye(nb))
    z_op = dq.tensor(z_storage, dq.eye(nb))
    n_op = dq.tensor(dq.destroy(na).dag() @ dq.destroy(na), dq.eye(nb))

    buffer_vac = dq.fock(nb, 0)
    psi_x = dq.tensor(plus_x, buffer_vac)
    psi_z = dq.tensor(g, buffer_vac)
    return psi_x, psi_z, x_op, z_op, n_op


def fit_lifetime(times: List[float], values: np.ndarray) -> Tuple[float, bool]:
    times_arr = np.asarray(times, dtype=float)
    vals = np.abs(np.asarray(values, dtype=float))
    mask = np.isfinite(vals) & (vals > EPS)
    if int(mask.sum()) < 3:
        return float("nan"), False
    try:
        slope, _intercept = np.polyfit(times_arr[mask], np.log(vals[mask]), 1)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan"), False
    if not np.isfinite(slope) or slope >= -EPS:
        return float("nan"), False
    return float(-1.0 / slope), True


def simulate_candidate(x: np.ndarray, config: Dict[str, object]) -> Dict[str, float]:
    a, _b, H, c_ops, alpha = build_system(x, config)
    psi_x, psi_z, x_op, z_op, n_op = logical_states_and_ops(alpha, config)
    times = list(map(float, config["times"]))
    tsave = jnp.array(times)
    opts = dq.Options(progress_meter=False)

    res_x = dq.mesolve(H, c_ops, psi_x, tsave, exp_ops=[x_op, n_op], options=opts)
    res_z = dq.mesolve(H, c_ops, psi_z, tsave, exp_ops=[z_op], options=opts)

    x_curve = as_float_array(res_x.expects[0].real)
    z_curve = as_float_array(res_z.expects[0].real)
    n_curve = as_float_array(res_x.expects[1].real)

    T_X, ok_x = fit_lifetime(times, x_curve)
    T_Z, ok_z = fit_lifetime(times, z_curve)
    bias = T_Z / T_X if np.isfinite(T_X) and T_X > 0 and np.isfinite(T_Z) else float("nan")

    return {
        "T_X": T_X,
        "T_Z": T_Z,
        "bias": bias,
        "alpha_est": alpha,
        "n_mean": float(n_curve[-1]),
        "fit_ok": bool(ok_x and ok_z),
    }


def reward(metrics: Dict[str, float], baseline: Dict[str, float], bias_target: float) -> float:
    T_X, T_Z, bias = metrics["T_X"], metrics["T_Z"], metrics["bias"]
    if not (np.isfinite(T_X) and np.isfinite(T_Z) and np.isfinite(bias) and T_X > 0 and T_Z > 0 and bias > 0):
        return -1e9

    floor_penalty = PENALTY_WEIGHT * (
        max(0.0, math.log(baseline["T_X"] / T_X)) ** 2
        + max(0.0, math.log(baseline["T_Z"] / T_Z)) ** 2
    )
    return float(
        -(math.log(bias / bias_target)) ** 2
        + LAMBDA_X * math.log(T_X / baseline["T_X"])
        + LAMBDA_Z * math.log(T_Z / baseline["T_Z"])
        - floor_penalty
    )


def hit_bounds(x: np.ndarray, bounds: np.ndarray, tol_frac: float = 0.03) -> bool:
    width = bounds[:, 1] - bounds[:, 0]
    return bool(np.any((x <= bounds[:, 0] + tol_frac * width) | (x >= bounds[:, 1] - tol_frac * width)))


class RandomElite:
    def __init__(self, mean: np.ndarray, sigma: float, bounds: np.ndarray, seed: int, population: int):
        self.mean = mean.copy()
        self.sigma = float(sigma)
        self.bounds = bounds
        self.population = int(population)
        self.rng = np.random.default_rng(seed)

    def ask(self) -> List[np.ndarray]:
        xs = self.rng.normal(self.mean, self.sigma, size=(self.population, len(self.mean)))
        return [np.clip(x, self.bounds[:, 0], self.bounds[:, 1]) for x in xs]

    def tell(self, xs: List[np.ndarray], rewards: List[float]) -> None:
        rewards_arr = np.asarray(rewards, dtype=float)
        order = np.argsort(rewards_arr)[::-1]
        elite_n = max(2, int(round(ELITE_FRACTION * len(xs))))
        elite_xs = np.asarray([xs[i] for i in order[:elite_n]])
        elite_rs = rewards_arr[order[:elite_n]]
        shifted = elite_rs - elite_rs.min() + 1e-9
        weights = shifted / shifted.sum() if shifted.sum() > 0 else np.ones(elite_n) / elite_n
        self.mean = np.clip(np.sum(elite_xs * weights[:, None], axis=0), self.bounds[:, 0], self.bounds[:, 1])
        self.sigma = float(np.clip(0.98 * self.sigma, 0.04, 0.8))


def make_optimizer(config: Dict[str, object], bounds: np.ndarray):
    mean = np.asarray(config["init_mean"], dtype=float)
    sigma = float(config["init_sigma"])
    if SepCMA is not None:
        return SepCMA(
            mean=mean,
            sigma=sigma,
            bounds=bounds,
            population_size=int(config["population"]),
            seed=int(config["seed"]),
        )
    return RandomElite(mean, sigma, bounds, int(config["seed"]), int(config["population"]))


def optimizer_mean(opt) -> np.ndarray:
    return np.asarray(opt.mean, dtype=float)


def ask(opt) -> List[np.ndarray]:
    if SepCMA is not None and isinstance(opt, SepCMA):
        return [np.asarray(opt.ask(), dtype=float) for _ in range(opt.population_size)]
    return opt.ask()


def tell(opt, xs: List[np.ndarray], rewards: List[float]) -> None:
    if SepCMA is not None and isinstance(opt, SepCMA):
        opt.tell([(x, -float(r)) for x, r in zip(xs, rewards)])
    else:
        opt.tell(xs, rewards)


def write_outputs(config: Dict[str, object], history: List[Dict[str, float]], best: Dict[str, object]) -> None:
    out = Path(str(config["output_dir"]))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with (out / "best_candidate.json").open("w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)


def main() -> None:
    config = dict(CONFIG)
    bounds = np.asarray(config["bounds"], dtype=float)
    bias_target = float(config["bias_target"])

    baseline = simulate_candidate(np.asarray(config["init_mean"], dtype=float), config)
    config["baseline"] = {
        "T_X": baseline["T_X"],
        "T_Z": baseline["T_Z"],
        "bias": baseline["bias"],
    }
    print(
        "Baseline: "
        f"T_X={baseline['T_X']:.4g}, T_Z={baseline['T_Z']:.4g}, "
        f"bias={baseline['bias']:.4g}, alpha={baseline['alpha_est']:.3g}"
    )

    opt = make_optimizer(config, bounds)
    history: List[Dict[str, float]] = []
    best = {"reward": -1e9, "x": list(config["init_mean"]), "metrics": baseline}
    previous_mean_bias = baseline["bias"]
    bias_down_streak = 0

    for epoch in range(int(config["epochs"])):
        xs = ask(opt)
        metrics = [simulate_candidate(x, config) for x in xs]
        rewards = [reward(m, baseline, bias_target) for m in metrics]
        tell(opt, xs, rewards)

        best_i = int(np.argmax(rewards))
        if rewards[best_i] > best["reward"]:
            best = {"reward": float(rewards[best_i]), "x": xs[best_i].tolist(), "metrics": metrics[best_i]}

        mean_x = optimizer_mean(opt)
        mean_metrics = simulate_candidate(mean_x, config)
        mean_reward = reward(mean_metrics, baseline, bias_target)

        if np.isfinite(mean_metrics["bias"]) and np.isfinite(previous_mean_bias) and mean_metrics["bias"] < previous_mean_bias:
            bias_down_streak += 1
        else:
            bias_down_streak = 0
        previous_mean_bias = mean_metrics["bias"]

        failed_fits = sum(0 if m["fit_ok"] else 1 for m in metrics)
        if bias_down_streak >= 5:
            print("WARNING: mean validation bias decreased for many epochs.")
        if mean_metrics["T_X"] < baseline["T_X"] or mean_metrics["T_Z"] < baseline["T_Z"]:
            print("WARNING: mean T_X or T_Z is below baseline.")
        if hit_bounds(mean_x, bounds):
            print("WARNING: optimizer mean is close to bounds.")
        if mean_metrics["bias"] > baseline["bias"] and mean_metrics["T_X"] < 0.7 * baseline["T_X"]:
            print("WARNING: bias improved mainly because T_X collapsed.")
        if failed_fits > len(metrics) // 3:
            print(f"WARNING: many fits failed this epoch ({failed_fits}/{len(metrics)}).")

        row = {
            "epoch": epoch,
            "reward_best": float(rewards[best_i]),
            "reward_mean": float(mean_reward),
            "mean_g2_re": float(mean_x[0]),
            "mean_g2_im": float(mean_x[1]),
            "mean_eps_d_re": float(mean_x[2]),
            "mean_eps_d_im": float(mean_x[3]),
            "best_g2_re": float(xs[best_i][0]),
            "best_g2_im": float(xs[best_i][1]),
            "best_eps_d_re": float(xs[best_i][2]),
            "best_eps_d_im": float(xs[best_i][3]),
            "T_X_best": float(metrics[best_i]["T_X"]),
            "T_Z_best": float(metrics[best_i]["T_Z"]),
            "bias_best": float(metrics[best_i]["bias"]),
            "T_X_mean": float(mean_metrics["T_X"]),
            "T_Z_mean": float(mean_metrics["T_Z"]),
            "bias_mean": float(mean_metrics["bias"]),
            "baseline_T_X": float(baseline["T_X"]),
            "baseline_T_Z": float(baseline["T_Z"]),
            "baseline_bias": float(baseline["bias"]),
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} reward_best={row['reward_best']:.3f} "
            f"best: T_X={row['T_X_best']:.3g}, T_Z={row['T_Z_best']:.3g}, bias={row['bias_best']:.3g} | "
            f"mean: T_X={row['T_X_mean']:.3g}, T_Z={row['T_Z_mean']:.3g}, bias={row['bias_mean']:.3g}"
        )

    write_outputs(config, history, best)
    print(f"\nBest reward={best['reward']:.4g}")
    print(f"Best x={np.array2string(np.asarray(best['x']), precision=4)}")
    print(f"Saved to {config['output_dir']}")


if __name__ == "__main__":
    main()
