# Slow Fourier control-drift tracking validation

## Reproduction command

```bash
python run_control_drift_tracking.py --epochs 160 --population 8 --reward-preset reference_signature_tracking --tracking-sigma-floor 0.08 --x-reference 1.5779816 -0.22211626 3.2540442 1.3158266
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

## 1. Drift chosen

The experiment uses deterministic, epoch-level, four-dimensional control drift:

```text
d_i(e) = A_i [0.70 sin(2*pi*e/P + phi1_i) + 0.30 sin(4*pi*e/P + phi2_i)]
x_eff(e) = x_cmd(e) - d(e)
```

- P: `240` epochs
- bandwidth: `2` harmonics
- amplitude scale: `1`
- effective amplitudes A: `[0.12, 0.06, 0.25, 0.18]`
- phi1: `[0, 0.8, 1.6, 2.4]`
- phi2: `[1.3, 2.1, 2.9, 3.7]`
- x_ref: `[1.5779816, -0.22211626, 3.2540442, 1.3158266]`

The components are ordered as Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d).

Reference note: the `DriftConfig` default preserves the supplied full-truncation baseline point `[1.3285188, -0.7782763, 3.0005451, 1.754515]`. This reproducible medium-preset run passes an explicit calibrated stationary reference with `--x-reference`, because the medium truncation has a different control representative with the same target-band lifetime signature.

Bounds check over the planned epoch range:

```json
{
  "inside_bounds": true,
  "original_amplitude_scale": 1.0,
  "amplitude_scale": 1.0,
  "was_scaled": false,
  "allowed_scale": 13.552235559650855,
  "limiting_component": 2,
  "limiting_epoch": 141,
  "path_min": [
    1.4962312168631706,
    -0.27816075392709316,
    3.050827233479758,
    1.2021070876414963
  ],
  "path_max": [
    1.6474761034103726,
    -0.17576926186112185,
    3.446913280223312,
    1.3723238091403864
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
    ]
  ]
}
```

## 2. Why this drift is slow

The no-drift convergence scale is about 15 optimizer epochs. The fundamental drift period is `240` epochs and the highest-harmonic period is `120` epochs. Both are much larger than 15 epochs, so the drift is adiabatic relative to the optimizer convergence scale.

## 3. Why this model

This is a generic low-bandwidth control drift on the four physical calibration knobs, directly analogous to the pi-pulse example where the measured knob is the commanded knob minus drift. In this cat-qubit setting it represents slow amplitude and phase offsets in the complex g2 and epsilon_d controls.

More physical drifts such as Kerr drift, storage detuning, TLS coupling, or SNR degradation are useful next benchmarks, but they do not provide an analytically obvious four-dimensional optimal command path. This control-drift benchmark is chosen first because the correct moving optimum is known exactly.

## 4. True optimal functional form

The true optimal command is:

```text
x_opt_true(e) = x_ref + d(e)
```

If the optimizer commands this path, the physical Hamiltonian receives `x_eff = x_ref`, so the lifetimes and bias should match the stationary no-drift optimized behavior.

## 5. What the optimizer sees

The optimizer sees only measured rewards, T_X, T_Z, and bias under the current epoch's drift. The drift vector and true optimum are used by the simulator and for plotting/reporting only; they are not used to choose candidates.

For `reference_signature_tracking`, the scalar reward also penalizes measured T_X/T_Z deviations from the stationary no-drift reference signature. This uses only measured lifetime outputs and a fixed no-drift calibration target; it does not reveal d(e) or x_ref + d(e) to the optimizer.

## 6. Plots produced

- drift_bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_bias_vs_epoch.png`
- drift_lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_lifetimes_vs_epoch.png`
- drift_reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_reward_or_loss_vs_epoch.png`
- drift_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_parameters_vs_epoch.png`
- drift_signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_signal_vs_epoch.png`
- drift_tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_tracking_error_vs_epoch.png`
- drift_effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_effective_parameters_vs_epoch.png`
- drift_optimized_decay_fits_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_optimized_decay_fits_final.png`
- drift_optimized_decay_fits_epoch_000: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_optimized_decay_fits_epoch_000.png`
- drift_optimized_decay_fits_epoch_mid: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_optimized_decay_fits_epoch_mid.png`
- drift_optimized_decay_fits_epoch_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/drift_optimized_decay_fits_epoch_final.png`

The parameter plot is the key diagnostic: solid lines are optimizer commands and dashed lines are the analytically known true command optimum x_ref + d(e).

## 7. Is the result good?

Yes. The post-warm-up command path tracks the dashed true optimum closely enough to keep the effective controls near the stationary reference and maintain the bias in the target band for most post-warm-up epochs.

Quantitative post-warm-up metrics:

- warm-up epoch: `15`
- final bias: `98.7569`
- median bias after warm-up: `100.209`
- fraction of post-warm-up epochs with 97 <= eta <= 103: `1`
- median tracking error L2 after warm-up: `0.0892681`
- final tracking error L2: `0.076035`
- median T_X after warm-up: `0.269895` us
- median T_Z after warm-up: `27.034` us
- median fit penalty after warm-up: `0`
- median fit R2 after warm-up: X=`0.999787`, Z=`0.999998`

The decay fits are considered well conditioned when fit penalty remains near zero and both X/Z R2 values are close to one.

Optimizer tracking details:

```json
{
  "initial_mean": [
    1.433242907528213,
    0.06763363403118226,
    3.2364215540093064,
    1.2543947360890555
  ],
  "reheat_count": 149,
  "tracking_sigma_floor": 0.08
}
```

## 8. What to see in the graphs

Expected good behavior is that the command curves converge to and follow the dashed true optimum curves, the drift signal remains smooth and slow, the effective physical parameters stay close to x_ref after warm-up, the bias stays near eta = 100 after warm-up, T_X and T_Z stay near the no-drift optimized values with small variations, and reward stabilizes after initial convergence.

## 9. Limitations

This is a deterministic slow-drift control benchmark. It is not yet a stochastic measurement-noise benchmark, Hamiltonian-detuning benchmark, Kerr benchmark, or SNR-degradation benchmark.

## 10. Next steps

- Add storage detuning drift with an explicit detuning compensation knob.
- Add Kerr drift and compare whether the four-command manifold is sufficient.
- Add measurement SNR degradation and uncertainty-aware reward/proxy logic.
- Compare this online SepCMA tracker against PPO or other online optimizers.

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
    "name": "reference_signature_tracking",
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
  "drift": {
    "kind": "slow_fourier_control",
    "period_epochs": 240.0,
    "bandwidth": 2,
    "amplitude_scale": 1.0,
    "amplitudes": [
      0.12,
      0.06,
      0.25,
      0.18
    ],
    "phi1": [
      0.0,
      0.8,
      1.6,
      2.4
    ],
    "phi2": [
      1.3,
      2.1,
      2.9,
      3.7
    ],
    "x_reference": [
      1.57798161,
      -0.22211626,
      3.2540442,
      1.3158266
    ]
  }
}
```
