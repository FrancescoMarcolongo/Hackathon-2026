"""Online optimization loops around surrogate rewards."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .reward import surrogate_reward

LOGGER = logging.getLogger(__name__)


class AskTellOptimizer(Protocol):
    """Minimal ask/tell optimizer interface."""

    def ask(self) -> list[np.ndarray]:
        """Return candidate theta vectors."""

    def tell(self, candidates: list[np.ndarray], rewards: list[float]) -> None:
        """Update optimizer state from evaluated rewards."""

    def best(self) -> tuple[np.ndarray, float]:
        """Return best theta and reward seen so far."""


class RandomSearchOptimizer:
    """Simple baseline optimizer useful before installing CMA-ES."""

    def __init__(
        self,
        center: np.ndarray | None = None,
        sigma: float = 0.05,
        population_size: int = 16,
        seed: int = 1234,
    ) -> None:
        self.center = np.zeros(4, dtype=float) if center is None else np.asarray(center, dtype=float).reshape(4)
        self.sigma = float(sigma)
        self.population_size = int(population_size)
        self.rng = np.random.default_rng(seed)
        self._best_theta = self.center.copy()
        self._best_reward = -np.inf

    def ask(self) -> list[np.ndarray]:
        return [self.center + self.sigma * self.rng.normal(size=4) for _ in range(self.population_size)]

    def tell(self, candidates: list[np.ndarray], rewards: list[float]) -> None:
        if not rewards:
            return
        best_idx = int(np.argmax(rewards))
        if rewards[best_idx] > self._best_reward:
            self._best_reward = float(rewards[best_idx])
            self._best_theta = np.asarray(candidates[best_idx], dtype=float).copy()
            self.center = self._best_theta.copy()
        self.sigma *= 0.995

    def best(self) -> tuple[np.ndarray, float]:
        return self._best_theta.copy(), float(self._best_reward)


class CMAESOptimizer:
    """Optional wrapper for ``cmaes.CMA`` when the dependency is installed."""

    def __init__(self, mean: np.ndarray, sigma: float, population_size: int = 16, seed: int = 1234) -> None:
        try:
            from cmaes import CMA
        except ImportError as exc:
            raise ImportError("Install cmaes to use CMAESOptimizer") from exc
        self._optimizer = CMA(
            mean=np.asarray(mean, dtype=float).reshape(4),
            sigma=float(sigma),
            population_size=int(population_size),
            seed=seed,
        )
        self._best_theta = np.asarray(mean, dtype=float).reshape(4)
        self._best_reward = -np.inf

    def ask(self) -> list[np.ndarray]:
        return [self._optimizer.ask() for _ in range(self._optimizer.population_size)]

    def tell(self, candidates: list[np.ndarray], rewards: list[float]) -> None:
        self._optimizer.tell([(candidate, -float(reward)) for candidate, reward in zip(candidates, rewards)])
        best_idx = int(np.argmax(rewards))
        if rewards[best_idx] > self._best_reward:
            self._best_reward = float(rewards[best_idx])
            self._best_theta = np.asarray(candidates[best_idx], dtype=float).copy()

    def best(self) -> tuple[np.ndarray, float]:
        return self._best_theta.copy(), float(self._best_reward)


def _theta_log(prefix: str, theta: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_re_g2": float(theta[0]),
        f"{prefix}_im_g2": float(theta[1]),
        f"{prefix}_re_eps_d": float(theta[2]),
        f"{prefix}_im_eps_d": float(theta[3]),
    }


def hybrid_online_optimization_loop(
    optimizer: AskTellOptimizer,
    config: Any,
    surrogate_bundle: dict[str, Any],
    adapter: Any,
    n_epochs: int | None = None,
    validation_every: int | None = None,
    retrain_callback: Any | None = None,
) -> pd.DataFrame:
    """Run an ask/tell optimization loop with periodic expensive validation."""

    n_epochs = int(n_epochs if n_epochs is not None else getattr(config, "n_epochs", 100))
    validation_every = int(validation_every if validation_every is not None else getattr(config, "validation_every", 10))
    retrain_every = int(getattr(config, "retrain_every", 0))
    rows: list[dict[str, float]] = []

    for epoch in range(n_epochs):
        candidates = optimizer.ask()
        rewards: list[float] = []
        infos: list[dict[str, float]] = []
        for candidate in candidates:
            reward, info = surrogate_reward(candidate, config, surrogate_bundle, adapter, return_info=True)
            rewards.append(float(reward))
            infos.append(info)

        optimizer.tell(candidates, rewards)
        best_theta, best_reward = optimizer.best()
        _, best_info = surrogate_reward(best_theta, config, surrogate_bundle, adapter, return_info=True)
        row: dict[str, float] = {
            "epoch": float(epoch),
            "surrogate_reward": float(best_reward),
            **_theta_log("best_theta", best_theta),
            "predicted_log_T_X": float(best_info["mean_log_T_X"]),
            "predicted_log_T_Z": float(best_info["mean_log_T_Z"]),
            "predicted_eta": float(best_info["mean_eta"]),
            "uncertainty": float(best_info["uncertainty"]),
            "physics_penalty": float(best_info["physics_penalty"]),
        }

        if validation_every > 0 and (epoch % validation_every == 0 or epoch == n_epochs - 1):
            try:
                truth = adapter.expensive_lifetime_benchmark(best_theta, config)
                row.update(
                    {
                        "true_T_X": float(truth["T_X"]),
                        "true_T_Z": float(truth["T_Z"]),
                        "true_eta": float(truth["eta"]),
                        "true_log_T_X": float(truth["log_T_X"]),
                        "true_log_T_Z": float(truth["log_T_Z"]),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Validation benchmark failed at epoch %s: %s", epoch, exc)

        rows.append(row)
        if retrain_callback is not None and retrain_every > 0 and (epoch + 1) % retrain_every == 0:
            surrogate_bundle = retrain_callback(pd.DataFrame(rows), surrogate_bundle)

    return pd.DataFrame(rows)
