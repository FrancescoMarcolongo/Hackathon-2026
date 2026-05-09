"""Dataset construction and feature matrix utilities."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .features import compute_cheap_observables
from .params import ensure_numpy_theta

LOGGER = logging.getLogger(__name__)

TARGET_COLUMNS = ("log_T_X", "log_T_Z")
METADATA_COLUMNS = {
    "sample_id",
    "failed",
    "error",
    "T_X",
    "T_Z",
    "eta",
    "log_eta",
    "theta",
    "g2",
    "eps_d",
}


def _draw_theta(param_sampler: Callable[[], np.ndarray] | Iterable[np.ndarray]) -> np.ndarray:
    if callable(param_sampler):
        return ensure_numpy_theta(param_sampler())
    return ensure_numpy_theta(next(param_sampler))  # type: ignore[arg-type]


def _row_from_theta(sample_id: int, theta: np.ndarray) -> dict[str, float]:
    return {
        "sample_id": float(sample_id),
        "theta_re_g2": float(theta[0]),
        "theta_im_g2": float(theta[1]),
        "theta_re_eps_d": float(theta[2]),
        "theta_im_eps_d": float(theta[3]),
    }


def generate_surrogate_dataset(
    param_sampler: Callable[[], np.ndarray] | Iterable[np.ndarray],
    n_samples: int,
    config: Any,
    adapter: Any,
    cache_path: str | None = None,
) -> pd.DataFrame:
    """Generate a supervised dataset using cheap features and expensive labels."""

    rows: list[dict[str, float]] = []
    for sample_id in tqdm(range(n_samples), desc="Generating surrogate dataset"):
        try:
            theta = _draw_theta(param_sampler)
            features = compute_cheap_observables(theta, config, adapter)
            benchmark = adapter.expensive_lifetime_benchmark(theta, config)
            row = _row_from_theta(sample_id, theta)
            row.update(features)
            row.update({key: float(value) for key, value in benchmark.items()})
            if "eta" not in row and "T_X" in row and "T_Z" in row:
                row["eta"] = float(row["T_Z"] / max(row["T_X"], 1e-300))
            if "log_eta" not in row and "eta" in row:
                row["log_eta"] = float(math.log(max(row["eta"], 1e-300)))
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - simulations fail in many backend-specific ways.
            LOGGER.exception("Skipping sample %s after simulation failure: %s", sample_id, exc)

    df = pd.DataFrame(rows)
    if cache_path is not None:
        save_dataset(df, cache_path)
    return df


def save_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a dataset as parquet when possible, otherwise CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return
        except Exception as exc:  # noqa: BLE001
            fallback = path.with_suffix(".csv")
            LOGGER.warning("Could not write parquet %s (%s); writing %s", path, exc, fallback)
            df.to_csv(fallback, index=False)
            return
    df.to_csv(path, index=False)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a parquet or CSV surrogate dataset."""

    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _candidate_feature_columns(
    df: pd.DataFrame,
    target_columns: tuple[str, ...],
    feature_columns: list[str] | None,
) -> list[str]:
    if feature_columns is not None:
        return list(feature_columns)
    excluded = set(target_columns) | METADATA_COLUMNS
    columns: list[str] = []
    for column in df.columns:
        if column in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            finite_count = np.isfinite(pd.to_numeric(df[column], errors="coerce")).sum()
            if finite_count > 0:
                columns.append(column)
    return columns


def make_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str] | None = None,
    target_columns: tuple[str, ...] = TARGET_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, StandardScaler, list[str]]:
    """Select finite numeric features, impute medians, and standardize them."""

    missing_targets = [column for column in target_columns if column not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    selected_features = _candidate_feature_columns(df, target_columns, feature_columns)
    if not selected_features:
        raise ValueError("No numeric feature columns were found")

    x_df = df[selected_features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_df.median(axis=0).fillna(0.0)
    x_df = x_df.fillna(medians)

    y_df = df[list(target_columns)].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_targets = y_df.notna().all(axis=1)
    if not valid_targets.all():
        LOGGER.warning("Dropping %d rows with invalid targets", int((~valid_targets).sum()))
        x_df = x_df.loc[valid_targets]
        y_df = y_df.loc[valid_targets]

    scaler = StandardScaler()
    x = scaler.fit_transform(x_df.to_numpy(dtype=np.float32))
    y = y_df.to_numpy(dtype=np.float32)
    return x.astype(np.float32), y.astype(np.float32), scaler, selected_features
