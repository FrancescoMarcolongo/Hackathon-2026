# Version 1 - Physical-coordinate baseline

## Purpose

Search in a more physically meaningful coordinate system while preserving the original estimator, reward, and update.

This folder is a copied version of the current project. The original `team-core-bias-optimizer_sannino/` folder and the original challenge notebook are not edited.

## Version 0

Raw-coordinate baseline:

```text
u = [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]
```

The optimizer proposes raw controls directly.

## Version 1

Physical-coordinate baseline:

```text
v = [log(kappa_2), log(abs(alpha)), arg(alpha), phi_mismatch]
```

Approximate relations:

```text
kappa_2 = 4 |g2|^2 / kappa_b
alpha^2 = 2 eps_d / conj(g2)
```

The implemented inverse convention is:

```text
g2_phase = phi_mismatch
g2 = abs_g2 * exp(i * g2_phase)
eps_d = 0.5 * conj(g2) * alpha^2
```

## What Changed

- The optimizer proposes `v`.
- The code converts `v -> u`.
- The simulator/evaluator still receives `u`.

## What Did Not Change

- lifetime extraction
- reward/loss
- optimizer update rule
- drift model, if any
- baseline estimator

Version 1 intentionally does not include BLT, Bayesian optimization, trust regions, robust reward changes, or new diagnostic reward terms.

## Success Criterion

Version 1 is better than Version 0 if it reaches the same reward/loss with fewer evaluations or reaches a better validated reward/loss with the same budget.

## Risk

The physical mapping is approximate. If the point is far from the intended cat-qubit regime, these coordinates may not fully diagonalize the search landscape. The square-root branch in `raw_to_physical` can also produce equivalent coordinate representations with different phases.

## Run Checks

Round-trip coordinate smoke check:

```bash
../.venv/bin/python test_physical_coordinates_roundtrip.py
```

Raw-vs-physical comparison:

```bash
../.venv/bin/python run_compare_v0_v1.py
```

Noisy raw-vs-physical resilience comparison:

```bash
../.venv/bin/python run_noise_resilience_compare.py --generations 3 --population 4 --noise-std 0.02
```

This adds Gaussian noise to the simulated decay samples before the robust exponential fit that extracts `Tx` and `Tz`. The noisy fitted loss drives the optimizer; the same candidates are also logged with clean fitted metrics for validation.

For a quicker smoke run:

```bash
../.venv/bin/python run_compare_v0_v1.py --quick --generations 1 --population 3
```

All outputs are saved into timestamped folders under:

```text
results/run_YYYYMMDD_HHMMSS_compare_v0_v1/
```
