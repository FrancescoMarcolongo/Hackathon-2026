# Core bias optimizer

Clean Dynamiqs implementation for the Alice & Bob cat-qubit core challenge:
reach a target bias `eta = T_Z / T_X`, then maximize the absolute lifetimes.

## Setup

The repository environment was missing only the challenge simulation packages.
The command used was:

```bash
python3 -m pip install "dynamiqs>=0.3.0" jax cmaes
```

No heavy extra dependencies are required beyond the notebook stack:
`dynamiqs`, `jax`, `cmaes`, `scipy`, `matplotlib`, and `numpy`.

## Run

From this folder:

```bash
python run_core_bias_optimization.py
```

For a short smoke test on a laptop:

```bash
python run_core_bias_optimization.py --quick
```

The full command writes:

- `results/optimization_history_proxy.csv`
- `results/final_validated_epoch_trace.csv`
- `results/baseline_vs_optimized.csv`
- `results/best_candidate.json`
- `validation_report.md`
- `figures/final_bias_vs_epoch.png`
- `figures/final_lifetimes_vs_epoch.png`
- `figures/final_reward_or_loss_vs_epoch.png`
- `figures/final_parameters_vs_epoch.png`

Each figure is also saved as PDF.

## Reward convention

`SepCMA` minimizes objective values.  The code therefore computes a readable
`reward`, then passes `loss_to_minimize = -reward` to `optimizer.tell`.

The primary reward is target-band constrained:

```text
reward =
    FEASIBILITY_BONUS * I[abs(bias / target - 1) <= tolerance]
    + W_LIFETIME * 0.5 * (log(T_X) + log(T_Z))
    - W_BIAS_EXACT * abs(log(bias) - log(target))^2
    - W_FIT * fit_penalty
    - floor_penalty
```

The selected result is chosen with the challenge rule: candidates inside the
target band first, then the smallest bias error, then the largest geometric
lifetime `sqrt(T_X * T_Z)`.
