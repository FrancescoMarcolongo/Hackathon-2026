#!/usr/bin/env python
"""Run online optimization with a learned surrogate reward."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_surrogate.adapters import DummyAdapter
from cat_surrogate.config import OptimizationConfig, SurrogateConfig
from cat_surrogate.model import load_surrogate_bundle
from cat_surrogate.optimize import RandomSearchOptimizer, hybrid_online_optimization_loop
from cat_surrogate.utils import configure_logging, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", default="artifacts/surrogate_bundle.pt")
    parser.add_argument("--output", default="logs/online_optimization.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--population-size", type=int, default=16)
    parser.add_argument("--sigma0", type=float, default=0.05)
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    configure_logging()
    surrogate_config = SurrogateConfig(random_seed=args.seed)
    opt_config = OptimizationConfig(
        n_epochs=args.epochs,
        population_size=args.population_size,
        sigma0=args.sigma0,
        validation_every=args.validation_every,
    )
    seed_everything(surrogate_config.random_seed)

    adapter = DummyAdapter()
    warnings.warn("Using DummyAdapter: online optimization results are non-physical.", stacklevel=2)
    bundle = load_surrogate_bundle(args.bundle)
    optimizer = RandomSearchOptimizer(
        sigma=opt_config.sigma0,
        population_size=opt_config.population_size,
        seed=args.seed,
    )
    log_df = hybrid_online_optimization_loop(
        optimizer,
        surrogate_config,
        bundle,
        adapter,
        n_epochs=opt_config.n_epochs,
        validation_every=opt_config.validation_every,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_df.to_csv(output, index=False)
    print(f"Wrote online optimization log with {len(log_df)} rows to {output}")


if __name__ == "__main__":
    main()
