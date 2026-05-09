"""Small shared helpers for scripts and notebooks."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a compact logging format for CLI scripts."""

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
