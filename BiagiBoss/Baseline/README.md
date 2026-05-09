# Core Bias Optimizer Baseline

This folder is a self-contained, no-drift baseline for the Alice & Bob
cat-qubit hackathon core challenge.  It is intended to be handed to another
Codex agent as the starting point for implementing drift models while keeping
the current exact-target optimization pipeline intact.

The current baseline does this:

1. Starts from a point below the target bias.
2. Optimizes the four real controls of complex `g2` and complex `epsilon_d`.
3. Measures `T_X`, `T_Z`, and `eta = T_Z / T_X` with Dynamiqs `dq.mesolve`.
4. Converges to the requested target bias `eta ~= 100`, not merely above it.
5. Produces validated plots and CSV/JSON reports.

No drift is currently implemented.  This is the stable no-drift reference.

## Quick Start

From this folder:

```bash
python3 run_core_bias_optimization.py
```

Short smoke test:

```bash
python3 run_core_bias_optimization.py --quick
```

Useful overrides:

```bash
python3 run_core_bias_optimization.py --epochs 20 --population 8
python3 run_core_bias_optimization.py --target-bias 100
python3 run_core_bias_optimization.py --max-configs 2
```

Dependencies used by this folder:

```bash
python3 -m pip install "dynamiqs>=0.3.0" jax cmaes
```

The code also uses `numpy`, `scipy`, and `matplotlib`.

## Current Scientific Target

The exact target is:

```text
TARGET_BIAS = 100.0
BIAS_TOL_REL = 0.03
```

A candidate is considered successful only if:

```text
97 <= eta <= 103
eta = T_Z / T_X
```

The current run starts below target:

```text
g2 = 1 + 0j
epsilon_d = 2.5 + 0j
eta ~= 40.35
```

The latest validated optimized point is:

```text
g2 = 1.3285188 - 0.7782763j
epsilon_d = 3.0005451 + 1.7545150j
T_X = 0.2691 us
T_Z = 26.8676 us
eta = 99.8393
```

This is intentionally different from the original challenge baseline:

```text
g2 = 1 + 0j
epsilon_d = 4 + 0j
eta ~= 322
```

The original baseline has much higher bias than the requested exact target, so
it is not treated as target-achieved under the exact-target objective.

## Folder Map

```text
team-core-bias-optimizer/
├── README.md
├── run_core_bias_optimization.py
├── cat_model.py
├── rewards.py
├── plotting.py
├── validation.py
├── validation_report.md
├── results/
└── figures/
```

## File-by-File Guide

### `run_core_bias_optimization.py`

This is the main executable and the best entry point for another Codex agent.

Important constants:

```python
TARGET_BIAS = 100.0
BIAS_TOL_REL = 0.03
CHALLENGE_BASELINE_X = np.array([1.0, 0.0, 4.0, 0.0])
OPTIMIZATION_START_X = np.array([1.0, 0.0, 2.5, 0.0])
BOUNDS = np.array([
    [0.25, 3.0],   # Re(g2)
    [-1.0, 1.0],   # Im(g2)
    [0.50, 8.0],   # Re(epsilon_d)
    [-3.0, 3.0],   # Im(epsilon_d)
])
```

Main responsibilities:

- Defines the optimization target and control bounds.
- Creates and runs `SepCMA`.
- Logs candidate metrics each epoch.
- Selects the best candidate according to exact target-band logic.
- Runs final notebook-quality validation.
- Runs a small final target-band refinement.
- Writes CSV/JSON/Markdown outputs.
- Calls plotting functions.

Key functions:

- `parse_args()`: command-line options.
- `optimizer_ask(opt)`: asks `SepCMA` for one population.
- `select_better(candidate, incumbent, reward_cfg)`: incumbent-selection rule.
- `evaluate_x(x, sim_cfg, reward_cfg)`: maps one 4-vector to metrics and reward.
- `run_one_optimization(...)`: online CMA loop.
- `build_sweep(args)`: defines proxy/full dimensions, epochs, population, and sweep grid.
- `validate_trajectory(...)`: re-evaluates the selected trajectory at final dimensions.
- `refine_final_target_band(...)`: final no-drift calibration of `epsilon_d` amplitude.
- `choose_selected_candidate(...)`: chooses the winning run from the sweep.
- `main()`: full orchestration.

Current optimizer:

```text
Optimizer: SepCMA
Default full population: 8 candidates per epoch
Default full epochs: 34
Default optimized variables: [Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d)]
```

Current M1-friendly proxy sweep:

```text
target_band_strict,  sigma0=0.45, seed=0
target_band_strict,  sigma0=0.75, seed=3
target_band_balanced, sigma0=0.45, seed=0
exact_target_strict, sigma0=0.45, seed=0
```

The selected run in the current report is:

```text
exact_target_strict
sigma0 = 0.45
seed = 0
population = 8
epochs = 34
```

Important note about plots: the reward plot currently shows
`incumbent_reward`, meaning the validated best-so-far/selected incumbent
reward.  It is not the noisy per-candidate reward distribution.

### `cat_model.py`

This is the physical model and lifetime-measurement module.  This is the most
important file for implementing drift in the Hamiltonian or controls.

Main objects and functions:

- `SimulationConfig`: dimensions, loss rates, fit windows, number of time points.
- `params_to_complex(x)`: converts optimizer vector to `(g2, epsilon_d)`.
- `complex_to_params(g2, eps_d)`: inverse helper.
- `estimate_alpha(g2, epsilon_d, ...)`: cat-size estimate from adiabatic elimination.
- `robust_exp_fit(times, values, ...)`: robust exponential fit using `least_squares`.
- `measure_lifetimes(g2, epsilon_d, cfg, ...)`: builds the Dynamiqs model, runs `dq.mesolve`, fits lifetimes.

The Hamiltonian currently follows the challenge notebook:

```python
H = (
    jnp.conj(g2) * a @ a @ b.dag()
    + g2 * a.dag() @ a.dag() @ b
    - eps_d * b.dag()
    - jnp.conj(eps_d) * b
)
```

The losses currently follow the challenge notebook:

```python
loss_b = jnp.sqrt(cfg.kappa_b) * b
loss_a = jnp.sqrt(cfg.kappa_a) * a
```

The logical observables:

- `X`: parity, built as `(1j * pi * a.dag() @ a).expm()`.
- `Z`: coherent-state projector difference
  `|alpha><alpha| - |-alpha><-alpha|`, tensor buffer identity.

Lifetime extraction:

- Simulate `+x` preparation to fit `T_X`.
- Simulate `+z` preparation to fit `T_Z`.
- Fit model:

```text
y = A * exp(-t / tau) + C
```

The fit uses `scipy.optimize.least_squares` with `loss="soft_l1"`.

Cache warning for drift implementation:

`measure_lifetimes` uses `_MEASURE_CACHE`, and `_cache_key(...)` currently
depends on:

```text
g2, epsilon_d, na, nb, kappa_b, kappa_a, t_final_x, t_final_z, n_points
```

If drift depends on epoch, time, random seed, drift amplitude, detuning, Kerr,
measurement noise, or any extra state, the cache key must include those new
drift parameters.  Otherwise the code may silently reuse no-drift results.

### `rewards.py`

This file defines all reward/loss logic.

Main objects:

- `RewardConfig`
- `compute_reward(metrics, cfg)`
- `default_reward_sweep(target_bias, bias_tol_rel)`

Current selected exact-target reward:

```text
bias_error = abs(log(eta) - log(target))
lifetime_score = 0.5 * (log(T_X) + log(T_Z))

reward =
    W_LIFETIME * lifetime_score
    - W_BIAS_EXACT * bias_error^2
    - W_FIT * fit_penalty
    - floor_penalty
```

Current selected hyperparameters:

```text
variant = exact_target
target_bias = 100
bias_tol_rel = 0.03
W_LIFETIME = 0.25
W_BIAS_EXACT = 180
W_FIT = 2
floor_weight = 12
```

Important optimizer convention:

```text
SepCMA minimizes.
The code computes reward.
The optimizer receives loss_to_minimize = -reward.
```

The target-band variants add a feasibility bonus when:

```text
abs(eta / target - 1) <= tolerance
```

The lower-bound variant is still present for comparison, but it is not the
current default objective because it can converge to bias values much larger
than the requested exact target.

### `plotting.py`

This file creates all figures.  It uses Matplotlib with `Agg` backend, so it
does not need a GUI.

Generated final figures:

- `figures/final_bias_vs_epoch.png`
- `figures/final_lifetimes_vs_epoch.png`
- `figures/final_reward_or_loss_vs_epoch.png`
- `figures/final_parameters_vs_epoch.png`
- `figures/baseline_decay_fits.png`
- `figures/optimized_decay_fits.png`
- `figures/sweep_summary_proxy.png`

Matching PDFs are also written.

Important functions:

- `plot_bias_vs_epoch(rows, target_bias, path)`
- `plot_lifetimes_vs_epoch(rows, path)`
- `plot_reward_vs_epoch(rows, path)`
- `plot_parameters_vs_epoch(rows, path)`
- `plot_decay_fit(result, path, title)`
- `plot_sweep_summary(rows, target_bias, path)`

For drift work, add drift-specific plots here, for example:

- drift amplitude vs epoch
- detuning vs epoch
- true drift vs compensated knob
- target tracking error `eta - target`
- measurement SNR or noise strength vs epoch

### `validation.py`

Small I/O and report helpers.

Important functions:

- `write_csv(path, rows)`
- `write_json(path, payload)`
- `result_row(label, result, reward)`
- `write_markdown_report(...)`

If drift is added, extend `result_row` and `write_markdown_report` to include:

- drift type
- drift schedule
- drift seed
- whether optimizer knew the drift model
- whether drift was compensated
- final target tracking error

### `validation_report.md`

Generated report from the most recent full run.  It records:

- reproduction command
- package versions
- reward formula
- selected reward config
- baseline/start/optimized table
- selected sweep run
- paths to final figures
- notes and caveats

Do not manually treat this as source of truth.  It is an output generated by
`run_core_bias_optimization.py`.

### `results/`

Generated machine-readable outputs.

Important files:

- `baseline_vs_optimized.csv`
  Final comparison table: challenge baseline, optimization start, optimized result.

- `optimization_history_proxy.csv`
  All logged proxy evaluations for all sweep runs.

- `sweep_final_candidates_proxy.csv`
  One final candidate per sweep configuration.

- `final_validated_epoch_trace.csv`
  Selected trajectory re-evaluated at final notebook dimensions.

- `final_refinement_candidates.csv`
  Final `epsilon_d` amplitude refinement candidates at notebook dimensions.

- `best_candidate.json`
  Full structured record of selected candidate, final validation, reward config,
  simulation configs, and refinement metadata.

For drift work, the agent should add drift columns to these CSVs instead of
creating disconnected side files.  Useful drift columns include:

```text
drift_kind
drift_epoch
drift_amp_real
drift_amp_imag
storage_detuning
kerr
measurement_sigma
true_g2_real
true_g2_imag
true_eps_d_real
true_eps_d_imag
```

### `figures/`

Generated visual outputs.  Both PNG and PDF are saved for most plots.

Current final plots show no-drift behavior only.  If drift is implemented, keep
the no-drift plots and add clearly named drift plots, for example:

```text
figures/drift_bias_vs_epoch.png
figures/drift_lifetimes_vs_epoch.png
figures/drift_reward_vs_epoch.png
figures/drift_parameters_vs_epoch.png
figures/drift_signal_vs_epoch.png
```

## Current Algorithm Flow

The no-drift pipeline is:

1. `main()` parses arguments.
2. `build_sweep()` defines proxy simulation budget and hyperparameter sweep.
3. For each reward/sigma/seed configuration:
   - create `SepCMA`;
   - start from `OPTIMIZATION_START_X`;
   - sample a population of 8 candidates each epoch;
   - evaluate each candidate with `evaluate_x`;
   - pass `(x, loss_to_minimize)` to `optimizer.tell`;
   - keep an incumbent using `select_better`.
4. `choose_selected_candidate()` chooses the best sweep run.
5. `validate_trajectory()` re-evaluates selected trajectory at final dimensions.
6. `refine_final_target_band()` scans a scalar factor on `epsilon_d` amplitude
   at final dimensions to remove proxy-to-full bias offset.
7. Results, report, and plots are written.

## Where To Implement Drift

The best places to implement drift are listed below.  Prefer small explicit
changes over rewriting the architecture.

### Option A: Control Drift

Use this for challenge drift like amplitude shifts in the buffer drive or
complex prefactors affecting `g2`/`epsilon_d`.

Recommended implementation:

1. Add a `DriftConfig` dataclass in `cat_model.py`, for example:

```python
@dataclass(frozen=True)
class DriftConfig:
    kind: str = "none"
    amplitude: float = 0.0
    period_epochs: float = 50.0
    phase: float = 0.0
    seed: int = 0
```

2. Add a helper:

```python
def apply_control_drift(g2, eps_d, epoch, drift_cfg):
    ...
    return g2_effective, eps_d_effective
```

3. Modify `measure_lifetimes(...)` to accept `epoch` and `drift_cfg`.
4. Apply drift before building `H`.
5. Add drift fields to `_cache_key(...)`.
6. Modify `evaluate_x(...)` in `run_core_bias_optimization.py` to pass epoch
   and drift config.
7. Log both optimizer knobs and effective drifted knobs.

Important: keep the optimizer knobs separate from the true drifted physical
controls.  Otherwise it becomes impossible to tell whether the optimizer is
compensating drift or simply being evaluated under changed parameters.

### Option B: Hamiltonian Drift

Use this for storage detuning, Kerr, or extra Hamiltonian terms.

Examples:

Storage detuning:

```python
H = H + delta_a * a.dag() @ a
```

Kerr:

```python
H = H + K * a.dag() @ a.dag() @ a @ a
```

Where to add:

In `cat_model.py`, inside `measure_lifetimes`, immediately after the base
Hamiltonian `H` is constructed and before `dq.mesolve`.

Also add all drift parameters to the cache key and logs.

### Option C: Measurement Drift or Noise

Use this for measurement SNR degradation.

Possible location:

- After `x_curve` and `z_curve` are extracted from `res_x.expects` and
  `res_z.expects`, apply deterministic or random measurement noise.

If random noise is used:

- store the random seed;
- include seed and noise strength in cache key;
- consider disabling cache for noisy repeated measurements;
- log noise parameters per epoch.

## Runner Changes Needed For Drift

For an online drift challenge, the optimizer should see a changing environment
across epochs.  The current runner does not pass `epoch` into `evaluate_x`.
Add it like this:

```python
def evaluate_x(x, sim_cfg, reward_cfg, *, epoch=0, drift_cfg=None):
    ...
    metrics = measure_lifetimes(g2, eps_d, sim_cfg, epoch=epoch, drift_cfg=drift_cfg)
```

Then in `run_one_optimization`:

```python
evaluated = [
    evaluate_x(x, sim_cfg, reward_cfg, epoch=epoch, drift_cfg=drift_cfg)
    for x in xs
]
```

Also pass the same drift context to:

- `start_eval`
- `mean_eval`
- `validate_trajectory`
- final validation
- final refinement, if refinement is still desired under drift

Decide explicitly whether final refinement is allowed in the drift setting.  If
the challenge requires only online adaptation, a post-hoc refinement may need to
be disabled or relabeled as a calibration step.

## Selection Logic To Preserve

The current exact-target selection rule is:

1. Prefer candidates inside the target band.
2. If both are inside the band, prefer smaller log-bias error.
3. If the bias errors are almost the same, prefer larger geometric lifetime:

```text
sqrt(T_X * T_Z)
```

This is implemented in `select_better(...)` and `choose_selected_candidate(...)`.

If implementing drift, keep this logic unless the task explicitly changes the
objective.

## Output Expectations After Adding Drift

A drift implementation should still produce:

- bias vs epoch, with target line
- lifetimes vs epoch
- reward/loss vs epoch
- optimizer parameters vs epoch
- drift signal vs epoch
- drift compensation vs epoch, if applicable
- baseline no-drift comparison
- drifted unoptimized comparison
- drifted optimized comparison

At minimum, the final report should answer:

1. What drift was applied?
2. Was the drift deterministic or random?
3. Did the optimizer know the drift model?
4. Did bias stay within `97 <= eta <= 103`?
5. How did `T_X` and `T_Z` change compared with no drift?
6. How much online adaptation was needed?

## Common Pitfalls

- Do not forget that `SepCMA` minimizes.  Pass `loss_to_minimize`, not reward.
- Do not let cache reuse no-drift results under drift.
- Do not plot only incumbent reward if you want to show population noise.
- Do not call a candidate successful just because `eta > 100`; exact-target
  success means `eta` is near `100`.
- Do not mix optimizer knobs and drifted effective physical parameters without
  logging both.
- Do not silently keep final refinement if the drift challenge disallows
  post-hoc calibration.

## Reproducibility Checklist For The Next Agent

Before modifying drift, run:

```bash
python3 run_core_bias_optimization.py --quick
python3 run_core_bias_optimization.py
```

Confirm the no-drift optimized row is close to:

```text
eta ~= 99.84
g2 ~= 1.3285 - 0.7783j
epsilon_d ~= 3.0005 + 1.7545j
```

After implementing drift, rerun both quick and full modes and update:

- `validation_report.md`
- `results/*.csv`
- `figures/*.png`
- this README if new files or drift modes are added

