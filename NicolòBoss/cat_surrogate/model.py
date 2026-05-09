"""PyTorch lifetime surrogate models and persistence helpers."""

from __future__ import annotations

import copy
import logging
import math
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

LOGGER = logging.getLogger(__name__)


class LifetimeSurrogate(nn.Module):
    """Small MLP that predicts ``log(T_X), log(T_Z)`` from cheap observables."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_reproducible_seed(seed: int) -> None:
    """Seed NumPy, Python, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(config: Any) -> torch.device:
    requested = getattr(config, "device", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable; using CPU")
        requested = "cpu"
    return torch.device(requested)


def _loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_surrogate_model(
    x: np.ndarray,
    y: np.ndarray,
    config: Any,
    validation_split: float = 0.2,
    seed: int | None = None,
    verbose: bool = True,
) -> tuple[LifetimeSurrogate, dict[str, list[float]]]:
    """Train one surrogate model with early stopping on validation MSE."""

    seed = getattr(config, "random_seed", 1234) if seed is None else seed
    set_reproducible_seed(seed)
    device = _device(config)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=validation_split,
        random_state=seed,
    )
    model = LifetimeSurrogate(input_dim=x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=getattr(config, "learning_rate", 1e-3),
        weight_decay=getattr(config, "weight_decay", 1e-4),
    )
    loss_fn = nn.MSELoss()
    train_loader = _loader(x_train, y_train, getattr(config, "batch_size", 64), shuffle=True)
    val_loader = _loader(x_val, y_val, getattr(config, "batch_size", 64), shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_val = math.inf
    bad_epochs = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    epochs = range(getattr(config, "max_epochs", 300))
    iterator = tqdm(epochs, desc="Training surrogate", disable=not verbose)

    for _epoch in iterator:
        model.train()
        train_losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                val_losses.append(float(loss_fn(model(xb), yb).detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        iterator.set_postfix(train=f"{train_loss:.4g}", val=f"{val_loss:.4g}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= getattr(config, "early_stopping_patience", 30):
                break

    model.load_state_dict(best_state)
    model.eval()
    return model.cpu(), history


def train_surrogate_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    config: Any,
    n_models: int = 5,
    validation_split: float = 0.2,
    verbose: bool = True,
) -> tuple[list[LifetimeSurrogate], list[dict[str, list[float]]]]:
    """Train an ensemble of independently seeded surrogate models."""

    models: list[LifetimeSurrogate] = []
    histories: list[dict[str, list[float]]] = []
    base_seed = getattr(config, "random_seed", 1234)
    for i in range(n_models):
        model, history = train_surrogate_model(
            x,
            y,
            config,
            validation_split=validation_split,
            seed=base_seed + i,
            verbose=verbose,
        )
        models.append(model)
        histories.append(history)
    return models, histories


def _config_metadata(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}


def save_surrogate_bundle(
    path: str | Path,
    models: list[LifetimeSurrogate],
    scaler: Any,
    feature_columns: list[str],
    config: Any | None = None,
    histories: list[dict[str, list[float]]] | None = None,
) -> None:
    """Save trained models, scaler, features, and metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_state_dicts": [model.cpu().state_dict() for model in models],
        "input_dim": len(feature_columns),
        "scaler": scaler,
        "feature_columns": feature_columns,
        "config": _config_metadata(config),
        "histories": histories or [],
    }
    joblib.dump(bundle, path)


def load_surrogate_bundle(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load a surrogate bundle saved by :func:`save_surrogate_bundle`."""

    raw = joblib.load(path)
    input_dim = int(raw["input_dim"])
    models: list[LifetimeSurrogate] = []
    for state in raw["model_state_dicts"]:
        model = LifetimeSurrogate(input_dim=input_dim)
        model.load_state_dict(state)
        model.to(map_location)
        model.eval()
        models.append(model)
    raw["models"] = models
    return raw


def _feature_vector(features_dict: dict[str, float], surrogate_bundle: dict[str, Any]) -> np.ndarray:
    feature_columns = surrogate_bundle["feature_columns"]
    values = np.asarray([features_dict.get(column, 0.0) for column in feature_columns], dtype=np.float32).reshape(1, -1)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return surrogate_bundle["scaler"].transform(values).astype(np.float32)


def predict_log_lifetimes(features_dict: dict[str, float], surrogate_bundle: dict[str, Any]) -> dict[str, float]:
    """Predict log lifetimes and ensemble uncertainty from a feature dict."""

    x = torch.as_tensor(_feature_vector(features_dict, surrogate_bundle), dtype=torch.float32)
    preds: list[np.ndarray] = []
    for model in surrogate_bundle["models"]:
        model.eval()
        with torch.no_grad():
            preds.append(model(x).cpu().numpy()[0])
    pred_arr = np.asarray(preds, dtype=float)
    mean = pred_arr.mean(axis=0)
    std = pred_arr.std(axis=0) if pred_arr.shape[0] > 1 else np.zeros(2, dtype=float)
    mean_log_eta = float(mean[1] - mean[0])
    return {
        "mean_log_T_X": float(mean[0]),
        "mean_log_T_Z": float(mean[1]),
        "std_log_T_X": float(std[0]),
        "std_log_T_Z": float(std[1]),
        "mean_log_eta": mean_log_eta,
        "mean_eta": float(math.exp(np.clip(mean_log_eta, -700.0, 700.0))),
    }
