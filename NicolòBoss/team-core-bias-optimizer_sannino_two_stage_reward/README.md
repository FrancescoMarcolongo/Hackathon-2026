# Two-stage reward comparison

This folder compares the original `team-core-bias-optimizer_sannino` reward against a two-stage reward.
The simulator and optimizer setup are matched; only the reward and feasible-candidate selection differ.

## Run

```bash
python run_two_stage_comparison.py --quick
python run_two_stage_comparison.py
```

## Current summary

| method | T_X | T_Z | eta | geo lifetime | target achieved | reward |
|---|---:|---:|---:|---:|---:|---:|
| baseline_original | 0.2004 | 20.09 | 100.2 | 2.007 | True | 0.1733 |
| two_stage | 0.2289 | 22.6 | 98.74 | 2.275 | True | 6.664 |

Artifacts are written to `results/` and `figures/`.
Last command options: `quick=True`, `epochs=24`, `population=6`, `seed=0`.
