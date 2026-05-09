# Boundary Liouvillian Tracking (BLT)

This folder contains an original online optimizer for the Alice & Bob cat-qubit stabilization challenge.  It keeps the challenge notebook's storage+buffer physics convention:

```text
H = conj(g2) a^2 b^\dagger + g2 (a^\dagger)^2 b - eps_d b^\dagger - conj(eps_d) b
L_b = sqrt(kappa_b) b
L_a = sqrt(kappa_a) a
```

The simulator does not double-count the effective two-photon dissipator.  It evolves the explicit storage+buffer Lindblad model and computes logical observables after the experiment is finished.

## BLT Idea

After a short burn-in time `t0`, the fast Liouvillian modes should have mostly decayed.  The remaining logical boundary dynamics are close to a slow effective process near lambda = 0.  BLT estimates those slow logical rates from two times, `t0` and `t1`, instead of fitting full decay curves.

The optimizer only sees this lab-like primitive:

```python
run_experiment(knobs, prep_state, meas_axis, wait_time, n_shots=None)
```

where `knobs = [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]`.  It never receives the Hamiltonian, jump operators, density matrices, Liouvillian, hidden simulator caches, or spectra.

## Estimators

`BLT-lite` uses differential logical contrasts:

- `C_X(t) = 0.5 * (<X>_{+X} - <X>_{-X})`
- `C_Z(t) = 0.5 * (<Z>_{+Z} - <Z>_{-Z})`

It estimates `gamma_X` and `gamma_Z` from the two-time contrast ratio.  Cost per candidate: 8 lab-like settings.

`BLT-full` prepares `+/-x`, `+/-y`, and `+/-z`, measures all three logical axes at `t0` and `t1`, builds a 3x3 slow propagator, and computes an effective logical generator with `logm`.  It also penalizes off-diagonal logical mixing.  Cost per candidate: 36 lab-like settings.

The naive reference prepares `+x` and `+z`, measures full decay curves over `K` time points, and fits exponentials with a robust least-squares fit.  In quick mode this costs 60 settings per candidate and much larger physical wait-time cost.

## Optimizers

`BLT-SPSA` works internally in physical coordinates:

```text
q = (log_kappa2, log_nbar, theta_alpha, phi_g)
```

Those are converted back to the four real lab knobs before any experiment call.  SPSA uses antithetic perturbations, parameter bounds, and a trust-region step.  The baseline is a tiny random/evolution-strategy optimizer driven by the naive full-fit reward.

## Files

- `cat_env.py` - hidden storage+buffer Lindblad simulator and `run_experiment`.
- `observables.py` - explicit re-export of logical state and observable helpers.
- `estimators.py` - BLT-lite, BLT-full, spectral reward, and naive full-fit reference.
- `optimizers.py` - BLT-SPSA and the naive random/ES baseline.
- `benchmark.py` - quick/full benchmark CLI and plot generation.
- `validation.py` - simulator checks, estimator scans, time-window scans, optimizer comparisons, finite-shot studies, and report generation.
- `figures.py` - shared plotting style and PNG/PDF figure helpers.
- `report_validation.md` - validation report generated from the latest validation CSV summaries.
- `requirements.txt` - lightweight scientific dependencies requested by the challenge.
- `.gitignore` - ignores caches and generated benchmark output folders.

Generated files:

- `outputs/quick_benchmark_latest/summary.csv`
- `outputs/quick_benchmark_latest/history.csv`
- `outputs/quick_benchmark_latest/metadata.json`
- `outputs/quick_benchmark_latest/reward_vs_iteration.png`
- `outputs/quick_benchmark_latest/eta_vs_iteration.png`
- `outputs/quick_benchmark_latest/tx_tz_diagnostics.png`
- `outputs/quick_benchmark_latest/measurement_cost_comparison.png`
- `outputs_validation/*.csv`
- `outputs_validation/*.png`
- `outputs_validation/*.pdf`

The `outputs/` and `outputs_validation/` folders are ignored because these artifacts are reproducible.

## Run

From the repository workspace:

```bash
python team-blt/benchmark.py --quick
```

The default quick run uses `na=10`, `nb=3`, `kappa_b=8.0`, `kappa_a=0.15`, `t0=1.0`, `t1=6.0`, and 30 points for each naive full-fit curve.  It includes BLT-lite and BLT-full by default.  Use `--skip-full` for an even faster run.

Validation commands:

```bash
python team-blt/validation.py --simulator-check
python team-blt/validation.py --estimator-scan --mode quick --n-points 20
python team-blt/validation.py --all --mode quick --n-points 15 --seeds 3
python team-blt/validation.py --all --mode medium --n-points 80 --seeds 10
python team-blt/validation.py --all --mode full --n-points 150 --seeds 20
```

The `--all` command is a real validation job.  It checkpoints CSVs as it goes and can be resumed with `--resume`.

## Latest Quick Result

On this workspace, `python team-blt/benchmark.py --quick` completed successfully and wrote outputs to `team-blt/outputs/quick_benchmark_latest/`.

| method | gold reward | gold T_X | gold T_Z | gold eta | candidate evals | settings | wait-time cost | wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| initial | 4.96558 | 2.48499 | 250.363 | 100.750 | 0 | 0 | 0 | 0.000 s |
| naive_random_es | 5.79920 | 3.39287 | 97.457 | 28.724 | 5 | 300 | 6750 | 16.713 s |
| blt_lite_spsa | 5.88800 | 3.14181 | 123.498 | 39.308 | 13 | 104 | 364 | 1.335 s |
| blt_full_spsa | 5.90691 | 3.09610 | 143.121 | 46.226 | 10 | 360 | 1260 | 5.215 s |

In this quick run, BLT-lite and BLT-full find operating points whose gold-standard fitted rewards are slightly better than the naive random/ES baseline, while BLT-lite uses about one third of the settings and about 5 percent of the physical wait-time cost.  BLT-full is more expensive than BLT-lite but still much cheaper in wait-time cost than the naive full-curve baseline.

## Limitations

The two-time estimator is only reliable when `t0` is after the fast transient and `t1 - t0` is long enough to resolve slow logical decay.  If there is too little decay, poor contrast, strong non-single-exponential behavior, or no clean timescale separation, BLT can mis-estimate `T_Z` or reject the candidate.  The benchmark keeps fixed quick-mode time windows; adaptive `t1` expansion is the natural next improvement for very high-bias points.
