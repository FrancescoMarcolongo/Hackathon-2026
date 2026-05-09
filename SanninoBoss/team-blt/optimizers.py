"""Lightweight optimizers for black-box cat-qubit rewards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from estimators import EstimateResult


Evaluator = Callable[[np.ndarray], EstimateResult]


@dataclass
class OptimizerResult:
    q_best: np.ndarray
    best: EstimateResult
    q_final: np.ndarray
    history: List[Dict[str, float | int | str]]
    n_evaluations: int


def clip_to_bounds(q: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(np.asarray(q, dtype=float), bounds[:, 0]), bounds[:, 1])


def _record(
    method: str,
    iteration: int,
    q: np.ndarray,
    result: EstimateResult,
    eval_type: str,
) -> Dict[str, float | int | str]:
    row: Dict[str, float | int | str] = {
        "method": method,
        "iteration": int(iteration),
        "eval_type": eval_type,
        "reward": float(result.reward),
        "valid": int(result.valid),
        "gamma_x": float(result.gamma_x),
        "gamma_z": float(result.gamma_z),
        "T_X": float(result.t_x),
        "T_Z": float(result.t_z),
        "eta": float(result.eta),
        "settings": int(result.settings),
        "wait_time_cost": float(result.wait_time_cost),
    }
    for i, value in enumerate(np.asarray(q, dtype=float)):
        row[f"q{i}"] = float(value)
    return row


@dataclass
class SPSAConfig:
    n_iter: int = 5
    a0: float = 0.20
    c0: float = 0.10
    alpha: float = 0.602
    gamma: float = 0.101
    stability_offset: float = 2.0
    trust_radius: float = 0.22
    seed: int = 123


class BLTSPSA:
    """Antithetic SPSA with a bounded trust-region update."""

    def __init__(self, config: SPSAConfig, bounds: np.ndarray, method_name: str):
        self.config = config
        self.bounds = np.asarray(bounds, dtype=float)
        self.method_name = method_name
        self.rng = np.random.default_rng(config.seed)

    def run(self, q_init: np.ndarray, evaluate: Evaluator) -> OptimizerResult:
        q = clip_to_bounds(q_init, self.bounds)
        history: List[Dict[str, float | int | str]] = []
        n_eval = 0

        best = evaluate(q)
        n_eval += 1
        q_best = q.copy()
        history.append(_record(self.method_name, 0, q, best, "center"))

        for k in range(1, self.config.n_iter + 1):
            ak = self.config.a0 / (k + self.config.stability_offset) ** self.config.alpha
            ck = self.config.c0 / k**self.config.gamma
            delta = self.rng.choice(np.array([-1.0, 1.0]), size=q.shape)
            q_plus = clip_to_bounds(q + ck * delta, self.bounds)
            q_minus = clip_to_bounds(q - ck * delta, self.bounds)

            r_plus = evaluate(q_plus)
            r_minus = evaluate(q_minus)
            n_eval += 2
            history.append(_record(self.method_name, k, q_plus, r_plus, "plus"))
            history.append(_record(self.method_name, k, q_minus, r_minus, "minus"))

            grad = (r_plus.reward - r_minus.reward) / (2.0 * ck * delta)
            if not np.all(np.isfinite(grad)):
                grad = np.zeros_like(q)
            step = ak * grad
            step_norm = float(np.linalg.norm(step))
            if step_norm > self.config.trust_radius:
                step *= self.config.trust_radius / step_norm

            q = clip_to_bounds(q + step, self.bounds)
            center = evaluate(q)
            n_eval += 1
            history.append(_record(self.method_name, k, q, center, "center"))

            for candidate_q, candidate in (
                (q_plus, r_plus),
                (q_minus, r_minus),
                (q, center),
            ):
                if candidate.reward > best.reward:
                    best = candidate
                    q_best = candidate_q.copy()

        return OptimizerResult(
            q_best=q_best,
            best=best,
            q_final=q.copy(),
            history=history,
            n_evaluations=n_eval,
        )


@dataclass
class RandomCMAStyleConfig:
    n_iter: int = 2
    population: int = 4
    sigma0: float = 0.22
    sigma_decay: float = 0.72
    elite_frac: float = 0.5
    seed: int = 321


class RandomCMAStyle:
    """Tiny evolution-strategy baseline used when full CMA-ES is overkill."""

    def __init__(self, config: RandomCMAStyleConfig, bounds: np.ndarray, method_name: str):
        self.config = config
        self.bounds = np.asarray(bounds, dtype=float)
        self.method_name = method_name
        self.rng = np.random.default_rng(config.seed)

    def run(self, q_init: np.ndarray, evaluate: Evaluator) -> OptimizerResult:
        mean = clip_to_bounds(q_init, self.bounds)
        sigma = float(self.config.sigma0)
        history: List[Dict[str, float | int | str]] = []
        n_eval = 0

        best = evaluate(mean)
        n_eval += 1
        q_best = mean.copy()
        history.append(_record(self.method_name, 0, mean, best, "mean"))

        for k in range(1, self.config.n_iter + 1):
            candidates = [mean]
            for _ in range(max(0, self.config.population - 1)):
                candidates.append(clip_to_bounds(mean + sigma * self.rng.normal(size=mean.shape), self.bounds))

            scored = []
            for j, q in enumerate(candidates):
                result = evaluate(q)
                n_eval += 1
                scored.append((result.reward, q, result))
                history.append(_record(self.method_name, k, q, result, f"sample{j}"))
                if result.reward > best.reward:
                    best = result
                    q_best = q.copy()

            scored.sort(key=lambda item: item[0], reverse=True)
            n_elite = max(1, int(np.ceil(self.config.elite_frac * len(scored))))
            elites = np.array([item[1] for item in scored[:n_elite]])
            weights = np.linspace(1.0, 0.4, n_elite)
            weights = weights / weights.sum()
            mean = clip_to_bounds(np.sum(elites * weights[:, None], axis=0), self.bounds)
            sigma *= self.config.sigma_decay

        return OptimizerResult(
            q_best=q_best,
            best=best,
            q_final=mean,
            history=history,
            n_evaluations=n_eval,
        )

