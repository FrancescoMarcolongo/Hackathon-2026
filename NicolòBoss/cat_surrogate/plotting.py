"""Matplotlib plots for surrogate diagnostics and online search."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_surrogate_parity(y_true: np.ndarray, y_pred: np.ndarray):
    """Parity plots for ``log(T_X)`` and ``log(T_Z)``."""

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ["log(T_X)", "log(T_Z)"]
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], s=18, alpha=0.75)
        lo = float(min(np.min(y_true[:, i]), np.min(y_pred[:, i])))
        hi = float(max(np.max(y_true[:, i]), np.max(y_pred[:, i])))
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax.set_xlabel(f"True {labels[i]}")
        ax.set_ylabel(f"Predicted {labels[i]}")
        ax.grid(alpha=0.25)
    return fig, axes


def plot_error_histograms(y_true: np.ndarray, y_pred: np.ndarray):
    """Histogram residuals for both log-lifetime targets."""

    errors = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    labels = ["log(T_X)", "log(T_Z)"]
    for i, ax in enumerate(axes):
        ax.hist(errors[:, i], bins=30, alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(f"{labels[i]} residual")
        ax.set_xlabel("Prediction error")
        ax.set_ylabel("Count")
    return fig, axes


def plot_bias_prediction(y_true: np.ndarray, y_pred: np.ndarray):
    """Compare true and predicted log-bias ``log(T_Z/T_X)``."""

    true_bias = y_true[:, 1] - y_true[:, 0]
    pred_bias = y_pred[:, 1] - y_pred[:, 0]
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    ax.scatter(true_bias, pred_bias, s=18, alpha=0.75)
    lo = float(min(np.min(true_bias), np.min(pred_bias)))
    hi = float(max(np.max(true_bias), np.max(pred_bias)))
    ax.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    ax.set_xlabel("True log eta")
    ax.set_ylabel("Predicted log eta")
    ax.grid(alpha=0.25)
    return fig, ax


def plot_online_tracking(log_df: pd.DataFrame):
    """Plot surrogate reward and available true validation metrics over epochs."""

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True, constrained_layout=True)
    axes[0].plot(log_df["epoch"], log_df["surrogate_reward"], label="Surrogate reward")
    axes[0].set_ylabel("Reward")
    axes[0].legend()

    axes[1].plot(log_df["epoch"], log_df["predicted_eta"], label="Predicted eta")
    if "true_eta" in log_df:
        valid = log_df["true_eta"].notna()
        axes[1].scatter(log_df.loc[valid, "epoch"], log_df.loc[valid, "true_eta"], label="True eta", color="black")
    axes[1].set_ylabel("Eta")
    axes[1].legend()

    axes[2].plot(log_df["epoch"], log_df["uncertainty"], label="Ensemble uncertainty")
    axes[2].plot(log_df["epoch"], log_df["physics_penalty"], label="Physics penalty")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()
    return fig, axes
