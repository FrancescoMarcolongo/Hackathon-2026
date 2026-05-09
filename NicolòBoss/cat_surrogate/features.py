"""Feature extraction and validation for learned lifetime surrogates."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .params import ensure_numpy_theta, theta_feature_dict

LOGGER = logging.getLogger(__name__)
BAD_VALUE_SENTINEL = 0.0


def _scalarize_value(key: str, value: Any) -> dict[str, float]:
    if isinstance(value, complex):
        return {
            f"{key}_real": float(value.real),
            f"{key}_imag": float(value.imag),
            f"{key}_abs": float(abs(value)),
        }

    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        if isinstance(scalar, complex):
            return {
                f"{key}_real": float(scalar.real),
                f"{key}_imag": float(scalar.imag),
                f"{key}_abs": float(abs(scalar)),
            }
        return {key: float(scalar)}

    if arr.size == 1:
        scalar = arr.reshape(-1)[0]
        if np.iscomplexobj(arr):
            return {
                f"{key}_real": float(np.real(scalar)),
                f"{key}_imag": float(np.imag(scalar)),
                f"{key}_abs": float(abs(scalar)),
            }
        return {key: float(scalar)}

    raise ValueError(f"Feature '{key}' is not scalar: shape={arr.shape}")


def flatten_feature_dict(d: Mapping[str, Any]) -> dict[str, float]:
    """Flatten a mapping into real-valued scalar features."""

    flat: dict[str, float] = {}
    for key, value in d.items():
        if isinstance(value, Mapping):
            nested = flatten_feature_dict(value)
            flat.update({f"{key}_{nested_key}": nested_value for nested_key, nested_value in nested.items()})
            continue
        flat.update(_scalarize_value(str(key), value))
    return flat


def validate_feature_dict(d: Mapping[str, Any]) -> dict[str, float]:
    """Return finite scalar numeric features, replacing bad values with sentinels."""

    flat = flatten_feature_dict(d)
    clean: dict[str, float] = {}
    for key, value in flat.items():
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            LOGGER.warning("Dropping non-numeric feature %s=%r: %s", key, value, exc)
            continue
        if not math.isfinite(value_f):
            LOGGER.warning("Replacing non-finite feature %s=%r with %s", key, value_f, BAD_VALUE_SENTINEL)
            value_f = BAD_VALUE_SENTINEL
        clean[key] = value_f
    return clean


def compute_cheap_observables(theta: np.ndarray, config: Any, adapter: Any) -> dict[str, float]:
    """Compute flat cheap observables from parameters plus adapter-provided probes."""

    theta_arr = ensure_numpy_theta(theta)
    features: dict[str, Any] = theta_feature_dict(theta_arr, kappa_b=config.kappa_b, na=config.na)

    static = adapter.compute_static_observables(theta_arr, config)
    short_time = adapter.compute_short_time_observables(theta_arr, config)
    features.update({f"static_{key}": value for key, value in static.items()})
    features.update({f"short_{key}": value for key, value in short_time.items()})

    return validate_feature_dict(features)
