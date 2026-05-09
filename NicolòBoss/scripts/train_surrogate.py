#!/usr/bin/env python
"""Train an ensemble surrogate from a generated dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cat_surrogate.config import SurrogateConfig
from cat_surrogate.dataset import load_dataset, make_feature_matrix
from cat_surrogate.model import save_surrogate_bundle, train_surrogate_ensemble
from cat_surrogate.utils import configure_logging, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/surrogate_dataset.parquet")
    parser.add_argument("--output", default="artifacts/surrogate_bundle.pt")
    parser.add_argument("--n-models", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    configure_logging()
    config = SurrogateConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, random_seed=args.seed)
    seed_everything(config.random_seed)

    df = load_dataset(args.dataset)
    x, y, scaler, feature_columns = make_feature_matrix(df)
    models, histories = train_surrogate_ensemble(x, y, config, n_models=args.n_models)
    save_surrogate_bundle(args.output, models, scaler, feature_columns, config=config, histories=histories)
    print(f"Saved {len(models)}-model surrogate bundle to {args.output}")


if __name__ == "__main__":
    main()
