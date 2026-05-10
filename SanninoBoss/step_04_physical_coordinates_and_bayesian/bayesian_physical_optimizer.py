"""Step 4B: Bayesian optimizer in physical coordinates."""

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
class BayesianOptConfig:
    max_epochs: int = 30
    n_init: int = 6
    beta: float = 1.25
    n_pool: int = 3072
    local_pool_fraction: float = 0.65
    local_scale_logs: float = 0.24
    local_scale_phases: float = 0.42
    global_scale_shrink: float = 0.92
    length_scale_logs: float = 0.35
    length_scale_phases: float = 0.70
    gp_noise_floor: float = 2.5e-4
    random_seed: int = 3
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


def _wrap_phases(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).copy()
    v[..., 2] = wrap_angle(v[..., 2])
    v[..., 3] = wrap_angle(v[..., 3])
    return v


def _matern52_kernel(xa: np.ndarray, xb: np.ndarray, length_scales: np.ndarray) -> np.ndarray:
    xa = np.asarray(xa, dtype=float) / length_scales
    xb = np.asarray(xb, dtype=float) / length_scales
    diff = xa[:, None, :] - xb[None, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    sqrt5 = math.sqrt(5.0)
    return (1.0 + sqrt5 * r + 5.0 * r * r / 3.0) * np.exp(-sqrt5 * r)


class BayesianPhysicalOptimizer:
    def __init__(
        self,
        *,
        bounds: PhysicalBounds,
        config: BayesianOptConfig,
        sim_cfg: TwoPointConfig,
        v0: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        self.bounds = bounds
        self.config = config
        self.sim_cfg = sim_cfg
        self.v0 = np.asarray(v0, dtype=float)
        self.rng = rng
        self.X_v: list[np.ndarray] = []
        self.X_z: list[np.ndarray] = []
        self.y: list[float] = []
        self.results: list[dict] = []
        self.best_result: dict | None = None
        self.best_v = self.v0.copy()
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
            return self._ask_warm_start()
        mu, sigma = self._predict(pool)
        acquisition = mu + self.config.beta * sigma
        return pool[int(np.argmax(acquisition))]

    def tell(self, v: np.ndarray, result: dict) -> None:
        score = float(result.get("reward_safe", result.get("reward", -1.0e6)))
        self.X_v.append(np.asarray(v, dtype=float).copy())
        self.X_z.append(embed_physical(v))
        self.y.append(score)
        self.results.append(result)
        if self.best_result is None or select_better_physical(result, self.best_result):
            self.best_result = dict(result)
            self.best_v = np.asarray(v, dtype=float).copy()

    def _ask_warm_start(self) -> np.ndarray:
        bounds_array = self.bounds.as_array()
        attempts = 0
        while attempts < 200:
            attempts += 1
            if len(self.X_v) == 0:
                v = self.v0.copy()
            else:
                scale = np.asarray(
                    [
                        self.config.local_scale_logs,
                        self.config.local_scale_logs,
                        self.config.local_scale_phases,
                        self.config.local_scale_phases,
                    ],
                    dtype=float,
                )
                v = self.v0 + self.rng.normal(0.0, scale)
            v[:2] = np.minimum(np.maximum(v[:2], bounds_array[:2, 0]), bounds_array[:2, 1])
            v = _wrap_phases(v)
            if self._valid_raw(v):
                return v
        return self.v0.copy()

    def _candidate_pool(self) -> np.ndarray:
        n_local = int(round(self.config.n_pool * self.config.local_pool_fraction))
        n_global = max(0, self.config.n_pool - n_local)
        bounds_array = self.bounds.as_array()

        local_scale = np.asarray(
            [
                self.config.local_scale_logs,
                self.config.local_scale_logs,
                self.config.local_scale_phases,
                self.config.local_scale_phases,
            ],
            dtype=float,
        )
        local = self.best_v + self.rng.normal(0.0, local_scale, size=(n_local, 4))
        global_unit = self.rng.random((n_global, 4))
        low = bounds_array[:, 0]
        high = bounds_array[:, 1]
        center = self.v0
        span = (high - low) * self.config.global_scale_shrink
        global_low = np.maximum(low, center - 0.5 * span)
        global_high = np.minimum(high, center + 0.5 * span)
        global_pool = global_low + global_unit * (global_high - global_low)
        pool = np.vstack([local, global_pool, self.best_v[None, :], self.v0[None, :]])
        pool[:, :2] = np.minimum(np.maximum(pool[:, :2], low[:2]), high[:2])
        pool = _wrap_phases(pool)
        valid = [v for v in pool if self._valid_raw(v)]
        return np.asarray(valid, dtype=float)

    def _valid_raw(self, v: np.ndarray) -> bool:
        u = physical_to_raw(
            v,
            self.sim_cfg,
            use_alpha_correction=self.config.use_alpha_correction,
        )
        return raw_bounds_ok(u)

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


def run_bayesian_physical_optimizer(
    *,
    verbose: bool = True,
    bo_cfg: BayesianOptConfig | None = None,
    reward_cfg: ImprovedRewardConfig | None = None,
    bounds: PhysicalBounds | None = None,
) -> dict:
    bo_cfg = bo_cfg or BayesianOptConfig()
    reward_cfg = reward_cfg or ImprovedRewardConfig()
    bounds = bounds or PhysicalBounds()
    sim_cfg = TwoPointConfig()
    noise_cfg = NoiseConfig(sigma=bo_cfg.noise_sigma, seed=bo_cfg.noise_seed)
    noise_rng = np.random.default_rng(bo_cfg.noise_seed)
    opt_rng = np.random.default_rng(bo_cfg.random_seed)
    clear_measure_cache()

    v0 = raw_to_physical(
        OPTIMIZATION_START_X.copy(),
        sim_cfg,
        use_alpha_correction=bo_cfg.use_alpha_correction,
    )
    optimizer = BayesianPhysicalOptimizer(bounds=bounds, config=bo_cfg, sim_cfg=sim_cfg, v0=v0, rng=opt_rng)
    run_id = f"step04B_bayesian_seed{bo_cfg.random_seed}_beta{bo_cfg.beta:g}"
    history: list[dict] = []
    start_time = time.time()
    incumbent: dict | None = None
    for epoch in range(0, bo_cfg.max_epochs + 1):
        v = optimizer.ask()
        observed = evaluate_candidate_physical(
            v,
            sim_cfg=sim_cfg,
            noise_cfg=noise_cfg,
            reward_cfg=reward_cfg,
            bounds=bounds,
            rng=noise_rng,
            use_alpha_correction=bo_cfg.use_alpha_correction,
        )
        optimizer.tell(v, observed)
        incumbent = dict(optimizer.best_result)
        history.append(history_row(run_id, epoch, observed, incumbent))
        if verbose and (epoch == 0 or epoch == 1 or epoch % 5 == 0 or epoch == bo_cfg.max_epochs):
            elapsed = time.time() - start_time
            print(
                f"{run_id} epoch={epoch:03d}/{bo_cfg.max_epochs} "
                f"incumbent Tx={float(incumbent['T_X']):.4g} "
                f"Tz={float(incumbent['T_Z']):.4g} "
                f"bias={float(incumbent['bias']):.4g} "
                f"reward={float(incumbent['reward_safe']):.3f} "
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
        "bo_config": asdict(bo_cfg),
        "physical_bounds": asdict(bounds),
    }
