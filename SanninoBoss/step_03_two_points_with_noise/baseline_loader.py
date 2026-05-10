"""Baseline no-noise epoch trace used for the step 3 comparison figures.

The numbers below are copied from the already-produced validated baseline
output:
team-core-bias-optimizer/results/final_validated_epoch_trace.csv

No baseline simulation or optimization is run from this step-3 folder.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True


TARGET_BIAS = 100.0
TARGET_TOLERANCE_REL = 0.03
BASELINE_N_POINTS = 60
BASELINE_SOURCE = "team-core-bias-optimizer/results/final_validated_epoch_trace.csv"
BASELINE_METADATA_SOURCE = "team-core-bias-optimizer/results/best_candidate.json: simulation_final.n_points"

BASELINE_EPOCHS = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35,
]

BASELINE_BIAS = [
    40.351025111216, 57.484609351693, 57.484609351693,
    65.285336136020, 82.824418483699, 108.252792601328,
    108.252792601328, 102.364296463824, 102.364296463824,
    102.364296463824, 102.364296463824, 102.364296463824,
    102.364296463824, 102.364296463824, 102.364296463824,
    102.364296463824, 102.364296463824, 103.277804278963,
    102.224163523159, 102.224163523159, 102.224163523159,
    102.224163523159, 102.224163523159, 104.109128970683,
    104.109128970683, 104.355190900069, 102.316545989182,
    102.316545989182, 102.199340581685, 102.199340581685,
    102.199340581685, 102.199340581685, 102.955115232994,
    102.466980484228, 103.025716370912, 99.839292745474,
]


def load_baseline_bias_trace() -> list[dict[str, float]]:
    return [
        {"epoch": float(epoch), "bias": float(bias)}
        for epoch, bias in zip(BASELINE_EPOCHS, BASELINE_BIAS)
    ]
