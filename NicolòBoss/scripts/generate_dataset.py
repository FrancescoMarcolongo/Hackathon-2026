#!/usr/bin/env python
"""Generate a surrogate dataset from cheap observables and benchmark labels."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from cat_surrogate.adapters import DummyAdapter
from cat_surrogate.config import SurrogateConfig
from cat_surrogate.dataset import generate_surrogate_dataset
from cat_surrogate.utils import configure_logging, seed_everything


def build_sampler(config: SurrogateConfig, seed: int):
    rng = np.random.default_rng(seed)
    g_radius = config.g2_over_kappa_b_max * config.kappa_b
    eps_radius = max(g_radius, 1e-3)

    def sample() -> np.ndarray:
        return np.asarray(
            [
                rng.uniform(-g_radius, g_radius),
                rng.uniform(-g_radius, g_radius),
                rng.uniform(-eps_radius, eps_radius),
                rng.uniform(-eps_radius, eps_radius),
            ],
            dtype=float,
        )

    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=256)
    parser.add_argument("--output", default="data/surrogate_dataset.parquet")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    configure_logging()
    config = SurrogateConfig(random_seed=args.seed)
    seed_everything(config.random_seed)
    adapter = DummyAdapter()
    warnings.warn("Using DummyAdapter: generated data are deterministic toy values, not physical simulations.", stacklevel=2)

    df = generate_surrogate_dataset(
        build_sampler(config, args.seed),
        n_samples=args.n_samples,
        config=config,
        adapter=adapter,
        cache_path=args.output,
    )
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
