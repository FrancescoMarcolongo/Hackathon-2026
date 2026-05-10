"""Load the already-computed no-noise baseline outputs.

This module intentionally performs no simulation, no optimization, and no
numeric post-processing beyond CSV/JSON parsing.  It is the data access layer
for the restyled step 1 figures.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


STEP_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = STEP_DIR.parent
SOURCE_ROOT = WORKSPACE_ROOT / "team-core-bias-optimizer"
RESULTS_DIR = SOURCE_ROOT / "results"

TRACE_CSV = RESULTS_DIR / "final_validated_epoch_trace.csv"
BASELINE_SUMMARY_CSV = RESULTS_DIR / "baseline_vs_optimized.csv"
BEST_CANDIDATE_JSON = RESULTS_DIR / "best_candidate.json"


NUMERIC_KEYS = {
    "epoch",
    "sigma0",
    "seed",
    "epoch_best_reward",
    "epoch_best_loss",
    "epoch_best_T_X",
    "epoch_best_T_Z",
    "epoch_best_bias",
    "epoch_best_geo_lifetime",
    "epoch_best_fit_penalty",
    "epoch_best_is_feasible",
    "epoch_best_g2_real",
    "epoch_best_g2_imag",
    "epoch_best_eps_d_real",
    "epoch_best_eps_d_imag",
    "mean_reward",
    "mean_loss",
    "mean_T_X",
    "mean_T_Z",
    "mean_bias",
    "mean_geo_lifetime",
    "mean_fit_penalty",
    "mean_is_feasible",
    "mean_g2_real",
    "mean_g2_imag",
    "mean_eps_d_real",
    "mean_eps_d_imag",
    "incumbent_reward",
    "incumbent_loss",
    "incumbent_T_X",
    "incumbent_T_Z",
    "incumbent_bias",
    "incumbent_geo_lifetime",
    "incumbent_fit_penalty",
    "incumbent_is_feasible",
    "incumbent_g2_real",
    "incumbent_g2_imag",
    "incumbent_eps_d_real",
    "incumbent_eps_d_imag",
    "optimizer_mean_x0",
    "optimizer_mean_x1",
    "optimizer_mean_x2",
    "optimizer_mean_x3",
    "T_X_us",
    "T_Z_us",
    "bias",
    "geo_lifetime_us",
    "reward",
    "loss_to_minimize",
    "fit_penalty",
    "fit_x_r2",
    "fit_z_r2",
}


def _convert_value(key: str, value: str) -> Any:
    if value == "":
        return value
    if key == "epoch" or key.endswith("_is_feasible") or key == "seed":
        return int(float(value))
    if key in NUMERIC_KEYS:
        return float(value)
    if value == "True":
        return True
    if value == "False":
        return False
    return value


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required baseline data file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _convert_value(key, value) for key, value in row.items()}
            for row in reader
        ]


def load_epoch_trace() -> list[dict[str, Any]]:
    """Return the validated selected-run epoch trace used by the old plots."""

    rows = _read_csv(TRACE_CSV)
    rows.sort(key=lambda row: int(row["epoch"]))
    return rows


def load_baseline_summary() -> list[dict[str, Any]]:
    """Return baseline/start/optimized summary rows from the existing run."""

    return _read_csv(BASELINE_SUMMARY_CSV)


def load_best_candidate_payload() -> dict[str, Any]:
    """Return metadata and fitted curves saved by the existing baseline run."""

    if not BEST_CANDIDATE_JSON.exists():
        raise FileNotFoundError(f"Required baseline metadata file not found: {BEST_CANDIDATE_JSON}")
    with BEST_CANDIDATE_JSON.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_step1_data() -> dict[str, Any]:
    """Load all persisted data needed to regenerate the step 1 figures."""

    payload = load_best_candidate_payload()
    reward_config = payload.get("reward_config", {})
    return {
        "trace": load_epoch_trace(),
        "summary": load_baseline_summary(),
        "metadata": payload,
        "target_bias": float(reward_config.get("target_bias", 100.0)),
        "target_tolerance_rel": float(reward_config.get("bias_tol_rel", 0.03)),
        "source_files": {
            "trace": TRACE_CSV,
            "summary": BASELINE_SUMMARY_CSV,
            "metadata": BEST_CANDIDATE_JSON,
        },
    }


if __name__ == "__main__":
    data = load_step1_data()
    print(f"Loaded {len(data['trace'])} validated epochs from {TRACE_CSV}")
    print(f"Loaded {len(data['summary'])} summary rows from {BASELINE_SUMMARY_CSV}")
