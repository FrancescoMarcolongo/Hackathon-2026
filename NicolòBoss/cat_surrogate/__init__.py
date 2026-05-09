"""Modular learned surrogate rewards for cat-qubit stabilization."""

from .adapters import DummyAdapter, SimulationAdapter
from .config import OptimizationConfig, SurrogateConfig
from .features import compute_cheap_observables
from .model import LifetimeSurrogate, load_surrogate_bundle, predict_log_lifetimes, save_surrogate_bundle
from .params import derived_physics_features, pack_params, unpack_params
from .reward import physics_penalty, short_time_proxy_reward, surrogate_reward, true_lifetime_reward

__all__ = [
    "DummyAdapter",
    "LifetimeSurrogate",
    "OptimizationConfig",
    "SimulationAdapter",
    "SurrogateConfig",
    "compute_cheap_observables",
    "derived_physics_features",
    "load_surrogate_bundle",
    "pack_params",
    "physics_penalty",
    "predict_log_lifetimes",
    "save_surrogate_bundle",
    "short_time_proxy_reward",
    "surrogate_reward",
    "true_lifetime_reward",
    "unpack_params",
]
