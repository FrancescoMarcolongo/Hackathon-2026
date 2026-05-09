"""Configuration dataclasses for cat-qubit surrogate training and search."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SurrogateConfig:
    """Shared configuration for feature generation, training, and rewards."""

    eta_target: float = 100.0
    kappa_b: float = 1.0
    na: int = 40
    alpha_min: float = 1.0
    alpha_max: float = 4.0
    g2_over_kappa_b_max: float = 0.2
    w_lifetime: float = 1.0
    w_bias: float = 1.0
    w_uncertainty: float = 0.25
    w_physics: float = 10.0
    probe_times_X: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    probe_times_Z: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    random_seed: int = 1234
    device: str = "cpu"
    batch_size: int = 64
    max_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 30


@dataclass
class OptimizationConfig:
    """Configuration for online optimizer orchestration."""

    n_epochs: int = 100
    validation_every: int = 10
    retrain_every: int = 25
    population_size: int = 16
    sigma0: float = 0.05
