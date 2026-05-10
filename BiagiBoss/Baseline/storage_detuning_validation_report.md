# Storage-detuning drift tracking validation

## Reproduction command

```bash
python run_storage_detuning_tracking.py --epochs 150 --population 8 --x-reference 1.6192076 -0.38684915 3.4954393 -0.66028127
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

I added one explicit optimizer knob:

```text
Delta_cmd = commanded storage-detuning compensation
Delta_eff(e) = Delta_cmd(e) - Delta_drift(e)
H_det(e) = Delta_eff(e) a^dag a
```

The optimizer now commands five real controls:

```text
[Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d), Delta_cmd]
```

The deterministic storage drift is faster than the previous benchmark but still slower than the roughly 15-epoch convergence scale:

```text
Delta_drift(e) = A [
    0.55 sin(2*pi*e/P + phi1)
  + 0.30 sin(4*pi*e/P + phi2)
  + 0.15 sin(6*pi*e/P + phi3)
]
```

- P: `96` epochs
- bandwidth: `3` harmonics
- highest-harmonic period: `32` epochs
- A: `0.08`
- weights: `[0.55, 0.3, 0.15]`
- phases: `[0.2, 1.35, 2.8]`
- stationary four-control x_ref: `[1.6192076, -0.38684915, 3.4954393, -0.66028127]`
- detuning reference: `0`

The true five-dimensional optimum is analytically known:

```text
x_opt_true(e) = [x_ref, Delta_drift(e)]
```

If the optimizer commands this curve, the physical Hamiltonian sees `Delta_eff = 0` and the measured cat-qubit behavior matches the stationary reference signature.

The optimizer is not given Delta_drift(e) or x_opt_true(e). It only receives measured rewards/lifetimes/bias from the drifted Hamiltonian. The scalar reward keeps the same target-bias objective and adds a measured T_X/T_Z reference signature penalty, so the benchmark has a local stationary target without revealing the hidden drift.

Bounds check:

```json
{
  "inside_bounds": true,
  "path_min": [
    1.6192075681484381,
    -0.38684915244951523,
    3.495439264194342,
    -0.6602812715117858,
    -0.07823561912303674
  ],
  "path_max": [
    1.6192075681484381,
    -0.38684915244951523,
    3.495439264194342,
    -0.6602812715117858,
    0.03769335929520456
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
      -0.18,
      0.18
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

- storage_detuning_bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_bias_vs_epoch.png`
- storage_detuning_lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_lifetimes_vs_epoch.png`
- storage_detuning_reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_reward_or_loss_vs_epoch.png`
- storage_detuning_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_parameters_vs_epoch.png`
- storage_detuning_signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_signal_vs_epoch.png`
- storage_detuning_tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_tracking_error_vs_epoch.png`
- storage_detuning_effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_effective_parameters_vs_epoch.png`
- storage_detuning_optimized_decay_fits_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_optimized_decay_fits_final.png`
- storage_detuning_optimized_decay_fits_epoch_000: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_optimized_decay_fits_epoch_000.png`
- storage_detuning_optimized_decay_fits_epoch_mid: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_optimized_decay_fits_epoch_mid.png`
- storage_detuning_optimized_decay_fits_epoch_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_optimized_decay_fits_epoch_final.png`

The parameter plot has the original four cat controls on the top panel and the new detuning-compensation knob on the bottom panel. The bottom panel should show an explicitly oscillating dashed target and a solid command curve converging to it.

## Is the result good?

Yes. The compensation command tracks the oscillating detuning target closely, the residual physical detuning remains small after warm-up, and eta stays in the 97-103 success band.

- warm-up epoch: `15`
- final bias: `100.09`
- median bias after warm-up: `100.319`
- fraction of post-warm-up epochs with 97 <= eta <= 103: `1`
- median tracking error L2 after warm-up: `0.042322`
- final tracking error L2: `0.0505978`
- median absolute residual storage detuning after warm-up: `0.022941`
- final absolute residual storage detuning: `0.0201873`
- median T_X after warm-up: `0.269695` us
- median T_Z after warm-up: `27.0599` us
- median fit penalty after warm-up: `0`
- median fit R2 after warm-up: X=`0.999745`, Z=`0.999993`

## Limitations and next steps

This is a deterministic storage-detuning benchmark with one explicit compensation knob. It is still noise-free. Natural next steps are stochastic measurement noise, SNR degradation, Kerr drift, and comparing this online SepCMA tracker against PPO or other online optimizers.

## Optimizer details

```json
{
  "initial_mean": [
    1.0,
    0.0,
    2.5,
    0.0,
    0.036178668944693376
  ],
  "reheat_count": 142,
  "tracking_sigma_floor": 0.06,
  "source": "first 150 epochs of deterministic 160-epoch run"
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
    "name": "storage_detuning_reference_signature",
    "variant": "target_band",
    "target_bias": 100.0,
    "bias_tol_rel": 0.03,
    "w_lifetime": 0.35,
    "w_bias_under": 60.0,
    "w_bias_exact": 160.0,
    "w_fit": 2.0,
    "feasibility_bonus": 18.0,
    "min_tx": 0.05,
    "min_tz": 5.0,
    "floor_weight": 12.0
  },
  "storage_detuning": {
    "kind": "storage_detuning_compensation",
    "period_epochs": 96.0,
    "bandwidth": 3,
    "amplitude": 0.08,
    "weights": [
      0.55,
      0.3,
      0.15
    ],
    "phases": [
      0.2,
      1.35,
      2.8
    ],
    "x_reference": [
      1.6192075681484381,
      -0.38684915244951523,
      3.495439264194342,
      -0.6602812715117858
    ],
    "detuning_reference": 0.0
  }
}
```
