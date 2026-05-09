from __future__ import annotations

import numpy as np

from cat_surrogate.adapters import DummyAdapter
from cat_surrogate.config import SurrogateConfig
from cat_surrogate.dataset import generate_surrogate_dataset
from cat_surrogate.features import compute_cheap_observables, flatten_feature_dict
from cat_surrogate.params import pack_params


def test_flatten_feature_dict_converts_complex_scalars() -> None:
    flat = flatten_feature_dict({"z": 1.0 + 2.0j, "nested": {"x": 3.0}})
    assert flat["z_real"] == 1.0
    assert flat["z_imag"] == 2.0
    assert flat["z_abs"] == abs(1.0 + 2.0j)
    assert flat["nested_x"] == 3.0


def test_compute_cheap_observables_dummy_adapter() -> None:
    config = SurrogateConfig()
    theta = pack_params(0.05 + 0.01j, 0.02 - 0.03j)
    features = compute_cheap_observables(theta, config, DummyAdapter())
    assert features
    assert all(np.isfinite(value) for value in features.values())


def test_dummy_adapter_dataset_generation_works() -> None:
    config = SurrogateConfig()
    samples = iter(
        [
            pack_params(0.05 + 0.01j, 0.02 - 0.03j),
            pack_params(0.03 + 0.02j, -0.01 + 0.02j),
        ]
    )
    df = generate_surrogate_dataset(samples, 2, config, DummyAdapter())
    assert len(df) == 2
    assert {"log_T_X", "log_T_Z", "eta"}.issubset(df.columns)
