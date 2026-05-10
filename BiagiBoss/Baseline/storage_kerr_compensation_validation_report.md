# Storage-Kerr five-knob compensation validation

## Reproduction command

```bash
python run_storage_kerr_compensation_tracking.py --epochs 60 --population 8 --warmup-epoch 45 --sim-preset medium
```

## Package versions

- python: 3.13.2
- platform: macOS-14.6-arm64-arm-64bit-Mach-O
- dynamiqs: 0.3.4
- jax: 0.6.2
- cmaes: 0.13.0
- scipy: 1.17.1
- matplotlib: 3.10.9
- numpy: 2.4.4

## Drift and compensation model

The fifth knob is an explicit Kerr compensation command:

```text
x = [Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d), K_cmd]
K_eff(e) = K_drift(e) + K_cmd(e)
H_K(e) = -0.5 K_eff(e) n_a (n_a - I)
```

The deterministic storage Kerr drift is:

```text
K_drift(e) = A [
    0.68 sin(2*pi*e/P + phi1)
  + 0.32 sin(4*pi*e/P + phi2)
]
```

- P: `100` epochs
- bandwidth: `2` harmonics
- highest-harmonic period: `50` epochs
- A: `0.3`
- weights: `[0.68, 0.32]`
- phases: `[0, 0]`
- stationary four-control x_ref: `[1.6192076, -0.38684915, 3.4954393, -0.66028127]`
- Kerr reference: `0`

The true five-dimensional optimum is analytically known:

```text
x_opt_true(e) = [x_ref, -K_drift(e)]
```

The optimizer is not given `K_drift(e)` or `x_opt_true(e)` when choosing candidates. It only sees measured rewards/lifetimes/bias under the current residual Kerr. For the five-knob comparison I also apply a mild stationary-calibration prior on the four original cat controls, with no epoch-dependent drift information, so the physically informative Kerr knob is preferred over equivalent four-control retunings.

- cat-control stationary prior weight: `2`

Bounds check:

```json
{
  "inside_bounds": true,
  "path_min": [
    1.6192075681,
    -0.3868491524,
    3.4954392642,
    -0.6602812715,
    -0.25982204357714167
  ],
  "path_max": [
    1.6192075681,
    -0.3868491524,
    3.4954392642,
    -0.6602812715,
    0.028606765903329726
  ],
  "bounds": [
    [
      0.25,
      3.0
    ],
    [
      -1.0,
      1.0
    ],
    [
      0.5,
      8.0
    ],
    [
      -3.0,
      3.0
    ],
    [
      -0.45,
      0.45
    ]
  ]
}
```

Stationary reference measurement:

```json
{
  "T_X": 0.2697814927247565,
  "T_Z": 26.97715961197772,
  "bias": 99.99633162198069,
  "fit_penalty": 0.0,
  "fit_x_r2": 0.9997496779803163,
  "fit_z_r2": 0.9999925879055767
}
```

## Plots produced

- storage_kerr5_bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_bias_vs_epoch.png`
- storage_kerr5_lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_lifetimes_vs_epoch.png`
- storage_kerr5_reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_reward_or_loss_vs_epoch.png`
- storage_kerr5_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_parameters_vs_epoch.png`
- storage_kerr5_signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_signal_vs_epoch.png`
- storage_kerr5_tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_tracking_error_vs_epoch.png`
- storage_kerr5_effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_effective_parameters_vs_epoch.png`
- storage_kerr5_optimized_decay_fits_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_optimized_decay_fits_final.png`

The parameter plot keeps the original four cat controls in the top panel and puts the physically informative Kerr compensation knob in the bottom panel. The bottom panel is the clean oscillatory target that the four-knob manifold lacks.

## Is the result good?

Yes. The explicit Kerr compensation command tracks the oscillating target, the residual physical Kerr remains small after warm-up, and eta stays in the target band.

- warm-up epoch: `45`
- final bias: `99.849`
- median bias after warm-up: `99.8847`
- fraction of post-warm-up epochs with 97 <= eta <= 103: `1`
- median tracking error L2 after warm-up: `0.0859547`
- final tracking error L2: `0.0998187`
- median absolute residual storage Kerr after warm-up: `0.080862`
- final absolute residual storage Kerr: `0.0992099`
- median T_X after warm-up: `0.269262` us
- median T_Z after warm-up: `26.9643` us
- median fit penalty after warm-up: `0`
- median fit R2 after warm-up: X=`0.99975`, Z=`0.999999`

## Limitations and next steps

This is still a deterministic, noise-free Kerr benchmark. The natural next step is a combined detuning-plus-Kerr drift and an uncertainty-aware reward under stochastic measurement noise.

## Optimizer details

```json
{
  "initial_mean": [
    1.3946033433295317,
    -0.19567274712248806,
    3.2646868049642372,
    -0.9198153365523273,
    -0.009797754464323842
  ],
  "reheat_count": 49,
  "tracking_sigma_floor": 0.055
}
```

## Run configuration

```json
{
  "simulation": {
    "na": 12,
    "nb": 4,
    "kappa_b": 10.0,
    "kappa_a": 1.0,
    "t_final_x": 1.2,
    "t_final_z": 220.0,
    "n_points": 36,
    "max_tau_factor": 80.0,
    "alpha_margin": 2.5
  },
  "reward": {
    "name": "storage_kerr_compensation_reference_signature",
    "variant": "target_band",
    "target_bias": 100.0,
    "bias_tol_rel": 0.03,
    "w_lifetime": 0.35,
    "w_bias_under": 60.0,
    "w_bias_exact": 170.0,
    "w_fit": 2.0,
    "feasibility_bonus": 18.0,
    "min_tx": 0.05,
    "min_tz": 5.0,
    "floor_weight": 12.0
  },
  "storage_kerr_compensation": {
    "kind": "storage_kerr_compensation",
    "period_epochs": 100.0,
    "bandwidth": 2,
    "amplitude": 0.3,
    "weights": [
      0.68,
      0.32
    ],
    "phases": [
      0.0,
      0.0
    ],
    "x_reference": [
      1.6192075681,
      -0.3868491524,
      3.4954392642,
      -0.6602812715
    ],
    "kerr_reference": 0.0
  },
  "cat_reference_weight": 2.0
}
```
