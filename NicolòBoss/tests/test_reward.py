from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from cat_surrogate.adapters import DummyAdapter
from cat_surrogate.config import SurrogateConfig
from cat_surrogate.features import compute_cheap_observables
from cat_surrogate.model import LifetimeSurrogate
from cat_surrogate.params import pack_params
from cat_surrogate.reward import physics_penalty, surrogate_reward


def _dummy_bundle(theta: np.ndarray, config: SurrogateConfig):
    features = compute_cheap_observables(theta, config, DummyAdapter())
    feature_columns = sorted(features)
    scaler = StandardScaler().fit(np.zeros((2, len(feature_columns)), dtype=np.float32))
    model = LifetimeSurrogate(input_dim=len(feature_columns))
    return {"models": [model], "scaler": scaler, "feature_columns": feature_columns}


def test_physics_penalty_finite_outputs() -> None:
    config = SurrogateConfig()
    theta = pack_params(0.05 + 0.01j, 0.02 - 0.03j)
    assert np.isfinite(physics_penalty(theta, config))


def test_model_forward_pass_shape() -> None:
    model = LifetimeSurrogate(input_dim=7)
    x = np.zeros((3, 7), dtype=np.float32)
    y = model.forward(__import__("torch").as_tensor(x))
    assert tuple(y.shape) == (3, 2)


def test_surrogate_reward_returns_finite_float_with_dummy_adapter() -> None:
    config = SurrogateConfig()
    theta = pack_params(0.05 + 0.01j, 0.02 - 0.03j)
    bundle = _dummy_bundle(theta, config)
    reward = surrogate_reward(theta, config, bundle, DummyAdapter())
    assert isinstance(reward, float)
    assert np.isfinite(reward)
