from __future__ import annotations

import numpy as np

from cat_surrogate.params import derived_physics_features, pack_params, unpack_params


def test_pack_unpack_roundtrip() -> None:
    g2 = 0.1 + 0.2j
    eps_d = -0.3 + 0.4j
    theta = pack_params(g2, eps_d)
    recovered_g2, recovered_eps_d = unpack_params(theta)
    assert recovered_g2 == g2
    assert recovered_eps_d == eps_d


def test_derived_physics_features_finite_outputs() -> None:
    features = derived_physics_features(0.1 + 0.2j, -0.3 + 0.4j, kappa_b=1.0, na=40)
    assert "alpha_est" in features
    assert "truncation_margin" in features
    assert all(np.isfinite(value) for value in features.values())
