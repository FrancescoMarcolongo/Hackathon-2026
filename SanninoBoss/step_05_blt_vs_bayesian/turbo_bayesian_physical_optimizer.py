"""Step 4B revision: trust-region Bayesian optimizer in physical coordinates."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from physical_coordinates_reward import (
    ImprovedRewardConfig,
    PhysicalBounds,
    evaluate_candidate_physical,
    history_row,
    physical_to_raw,
    raw_bounds_ok,
    raw_to_physical,
    select_better_physical,
    wrap_angle,
)
from two_points_with_noise import (
    NoiseConfig,
    OPTIMIZATION_START_X,
    TwoPointConfig,
    cleanup_local_qutip_cache,
    clear_measure_cache,
)


@dataclass(frozen=True)
class TurboBayesianOptConfig:
    max_epochs: int = 45
    n_init: int = 4
    beta: float = 0.8
    n_pool: int = 4096
    global_fraction: float = 0.10
    initial_trust_region_length: float = 0.44
    length_min: float = 0.04
    length_max: float = 0.90
    success_tolerance: int = 2
    failure_tolerance: int = 3
    improvement_tol: float = 1.0e-3
    expand_factor: float = 1.6
    shrink_factor: float = 1.7
    warm_start_length: float = 0.34
    length_scale_logs: float = 0.35
    length_scale_phases: float = 0.70
    gp_noise_floor: float = 2.5e-4
    random_seed: int = 9
    noise_seed: int = 11
    noise_sigma: float = 0.03
    use_alpha_correction: bool = True


def embed_physical(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return np.asarray(
        [
            v[0],
            v[1],
            math.sin(v[2]),
            math.cos(v[2]),
            math.sin(v[3]),
            math.cos(v[3]),
        ],
        dtype=float,
    )


def _matern52_kernel(xa: np.ndarray, xb: np.ndarray, length_scales: np.ndarray) -> np.ndarray:
    xa = np.asarray(xa, dtype=float) / length_scales
    xb = np.asarray(xb, dtype=float) / length_scales
    diff = xa[:, None, :] - xb[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    sqrt5 = math.sqrt(5.0)
    return (1.0 + sqrt5 * r + 5.0 * r * r / 3.0) * np.exp(-sqrt5 * r)


class TurboBayesianPhysicalOptimizer:
    def __init__(
        self,
        *,
        bounds: PhysicalBounds,
        config: TurboBayesianOptConfig,
        sim_cfg: TwoPointConfig,
        v0: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.bounds = bounds
        self.config = config
        self.sim_cfg = sim_cfg
        self.v0 = np.asarray(v0, dtype=float)
        self.rng = rng
        self.bounds_array = bounds.as_array()
        self.X_v: list[np.ndarray] = []
        self.X_z: list[np.ndarray] = []
        self.y: list[float] = []
        self.results: list[dict] = []
        self.best_result: dict | None = None
        self.best_v = self.v0.copy()
        self.center = self.v0.copy()
        self.trust_region_length = float(config.initial_trust_region_length)
        self.success_counter = 0
        self.failure_counter = 0
        self.restart_count = 0
        self.best_reward = -np.inf
        self._length_scales = np.asarray(
            [
                config.length_scale_logs,
                config.length_scale_logs,
                config.length_scale_phases,
                config.length_scale_phases,
                config.length_scale_phases,
                config.length_scale_phases,
            ],
            dtype=float,
        )

    def ask(self) -> np.ndarray:
        if len(self.X_v) < self.config.n_init:
            return self._ask_warm_start()
        pool = self._candidate_pool()
        if len(pool) == 0:
            return self.center.copy()
        mu, sigma = self._predict(pool)
        acquisition = mu + self.config.beta * sigma
        return pool[int(np.argmax(acquisition))]

    def tell(self, v: np.ndarray, result: dict) -> None:
        score = float(result.get("reward_safe", result.get("reward", -1.0e6)))
        previous_best_reward = self.best_reward
        self.X_v.append(np.asarray(v, dtype=float).copy())
        self.X_z.append(embed_physical(v))
        self.y.append(score)
        self.results.append(result)

        reward_improved = score > previous_best_reward + self.config.improvement_tol
        if reward_improved:
            self.success_counter += 1
            self.failure_counter = 0
            self.best_reward = score
        else:
            self.failure_counter += 1
            self.success_counter = 0

        if self.best_result is None or select_better_physical(result, self.best_result):
            self.best_result = dict(result)
            self.best_v = np.asarray(v, dtype=float).copy()
            self.center = self.best_v.copy()

        if self.success_counter >= self.config.success_tolerance:
            self.trust_region_length = min(
                self.config.length_max,
                self.trust_region_length * self.config.expand_factor,
            )
            self.success_counter = 0
        if self.failure_counter >= self.config.failure_tolerance:
            self.trust_region_length = self.trust_region_length / self.config.shrink_factor
            self.failure_counter = 0

        if self.trust_region_length < self.config.length_min:
            self.restart_count += 1
            self.trust_region_length = float(self.config.initial_trust_region_length)
            self.center = self.best_v.copy()
            self.success_counter = 0
            self.failure_counter = 0

    def state(self) -> dict[str, float | int | str]:
        return {
            "optimizer_type": "trust-region Bayesian / TUrBO-like",
            "trust_region_length": float(self.trust_region_length),
            "success_counter": int(self.success_counter),
            "failure_counter": int(self.failure_counter),
            "best_reward": float(self.best_reward),
            "restart_count": int(self.restart_count),
        }

    def _ask_warm_start(self) -> np.ndarray:
        attempts = 0
        center_u = self._to_unit(self.v0)
        while attempts < 300:
            attempts += 1
            if len(self.X_v) == 0:
                v = self.v0.copy()
            else:
                u = center_u.copy()
                delta = self.rng.uniform(
                    -0.5 * self.config.warm_start_length,
                    0.5 * self.config.warm_start_length,
                    size=4,
                )
                u[:2] = np.clip(u[:2] + delta[:2], 0.0, 1.0)
                u[2:] = (u[2:] + delta[2:]) % 1.0
                v = self._from_unit(u)
            if self._valid_raw(v):
                return v
        return self.v0.copy()

    def _candidate_pool(self) -> np.ndarray:
        n_global = int(round(self.config.n_pool * self.config.global_fraction))
        n_local = max(0, self.config.n_pool - n_global)
        center_u = self._to_unit(self.center)

        local_delta = self.rng.uniform(
            -0.5 * self.trust_region_length,
            0.5 * self.trust_region_length,
            size=(n_local, 4),
        )
        local_u = np.empty((n_local, 4), dtype=float)
        local_u[:, :2] = np.clip(center_u[:2] + local_delta[:, :2], 0.0, 1.0)
        local_u[:, 2:] = (center_u[2:] + local_delta[:, 2:]) % 1.0

        global_u = self.rng.random((n_global, 4))
        units = np.vstack([local_u, global_u, center_u[None, :], self._to_unit(self.v0)[None, :]])
        pool = np.asarray([self._from_unit(u) for u in units], dtype=float)
        valid = [v for v in pool if self._valid_raw(v)]
        return np.asarray(valid, dtype=float)

    def _to_unit(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        low = self.bounds_array[:, 0]
        high = self.bounds_array[:, 1]
        u = np.empty(4, dtype=float)
        u[:2] = (v[:2] - low[:2]) / (high[:2] - low[:2])
        wrapped = np.asarray([wrap_angle(v[2]), wrap_angle(v[3])])
        u[2:] = (wrapped - low[2:]) / (high[2:] - low[2:])
        return np.clip(u, 0.0, 1.0)

    def _from_unit(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float).copy()
        low = self.bounds_array[:, 0]
        high = self.bounds_array[:, 1]
        v = low + u * (high - low)
        v[2] = float(wrap_angle(v[2]))
        v[3] = float(wrap_angle(v[3]))
        return v

    def _valid_raw(self, v: np.ndarray) -> bool:
        raw = physical_to_raw(
            v,
            self.sim_cfg,
            use_alpha_correction=self.config.use_alpha_correction,
        )
        return raw_bounds_ok(raw)

    def _predict(self, pool_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_train = np.vstack(self.X_z)
        y = np.asarray(self.y, dtype=float)
        y_mean = float(np.mean(y))
        y_std = float(np.std(y)) if len(y) > 1 else 1.0
        if y_std < 1.0e-9:
            y_std = 1.0
        y_scaled = (y - y_mean) / y_std
        x_pool = np.vstack([embed_physical(v) for v in pool_v])
        k_train = _matern52_kernel(x_train, x_train, self._length_scales)
        k_train += np.eye(len(x_train)) * self.config.gp_noise_floor
        try:
            c, lower = cho_factor(k_train, lower=True, check_finite=False)
            alpha = cho_solve((c, lower), y_scaled, check_finite=False)
            k_cross = _matern52_kernel(x_pool, x_train, self._length_scales)
            mu = k_cross @ alpha
            v = cho_solve((c, lower), k_cross.T, check_finite=False)
            var = np.maximum(1.0 - np.sum(k_cross * v.T, axis=1), 1.0e-9)
            sigma = np.sqrt(var)
        except Exception:
            mu = np.zeros(len(pool_v), dtype=float)
            sigma = np.ones(len(pool_v), dtype=float)
        return mu, sigma


def run_turbo_bayesian_physical_optimizer(
    *,
    verbose: bool = True,
    turbo_cfg: TurboBayesianOptConfig | None = None,
    reward_cfg: ImprovedRewardConfig | None = None,
    bounds: PhysicalBounds | None = None,
) -> dict:
    turbo_cfg = turbo_cfg or TurboBayesianOptConfig()
    reward_cfg = reward_cfg or ImprovedRewardConfig()
    bounds = bounds or PhysicalBounds()
    sim_cfg = TwoPointConfig()
    noise_cfg = NoiseConfig(sigma=turbo_cfg.noise_sigma, seed=turbo_cfg.noise_seed)
    noise_rng = np.random.default_rng(turbo_cfg.noise_seed)
    opt_rng = np.random.default_rng(turbo_cfg.random_seed)
    clear_measure_cache()

    v0 = raw_to_physical(
        OPTIMIZATION_START_X.copy(),
        sim_cfg,
        use_alpha_correction=turbo_cfg.use_alpha_correction,
    )
    optimizer = TurboBayesianPhysicalOptimizer(
        bounds=bounds,
        config=turbo_cfg,
        sim_cfg=sim_cfg,
        v0=v0,
        rng=opt_rng,
    )
    run_id = f"step04B_turbo_seed{turbo_cfg.random_seed}_beta{turbo_cfg.beta:g}"
    history: list[dict] = []
    start_time = time.time()
    incumbent: dict | None = None
    for epoch in range(0, turbo_cfg.max_epochs + 1):
        candidate_v = optimizer.ask()
        observed = evaluate_candidate_physical(
            candidate_v,
            sim_cfg=sim_cfg,
            noise_cfg=noise_cfg,
            reward_cfg=reward_cfg,
            bounds=bounds,
            rng=noise_rng,
            use_alpha_correction=turbo_cfg.use_alpha_correction,
        )
        optimizer.tell(candidate_v, observed)
        incumbent = dict(optimizer.best_result)
        row = history_row(run_id, epoch, observed, incumbent)
        row.update(optimizer.state())
        row["center_log_kappa2"] = float(optimizer.center[0])
        row["center_log_abs_alpha"] = float(optimizer.center[1])
        row["center_theta_alpha"] = float(optimizer.center[2])
        row["center_theta_g"] = float(optimizer.center[3])
        history.append(row)
        if verbose and (epoch == 0 or epoch == 1 or epoch % 5 == 0 or epoch == turbo_cfg.max_epochs):
            elapsed = time.time() - start_time
            print(
                f"{run_id} epoch={epoch:03d}/{turbo_cfg.max_epochs} "
                f"incumbent Tx={float(incumbent['T_X']):.4g} "
                f"Tz={float(incumbent['T_Z']):.4g} "
                f"bias={float(incumbent['bias']):.4g} "
                f"reward={float(incumbent['reward_safe']):.3f} "
                f"tr={optimizer.trust_region_length:.3f} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    cleanup_local_qutip_cache()
    return {
        "history": history,
        "incumbent": incumbent,
        "v0": v0,
        "x0": OPTIMIZATION_START_X.copy(),
        "sim_config": asdict(sim_cfg),
        "noise_config": asdict(noise_cfg),
        "reward_config": asdict(reward_cfg),
        "turbo_config": asdict(turbo_cfg),
        "physical_bounds": asdict(bounds),
    }
