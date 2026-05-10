# Storage-Kerr four-knob drift tracking validation

## Reproduction command

```bash
python run_storage_kerr_tracking.py --epochs 80 --population 8 --period-epochs 100.0 --kerr-amplitude 0.3 --sim-preset medium
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

## Kerr drift model

I returned to the original four optimizer knobs:

```text
x = [Re(g2), Im(g2), Re(epsilon_d), Im(epsilon_d)]
```

The storage Hamiltonian now includes a drifted Kerr nonlinearity:

```text
n_a = a^dag a
H_K(e) = -0.5 K(e) n_a (n_a - I)
```

The deterministic storage Kerr drift is:

```text
K(e) = A [
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

The highest harmonic period is still longer than the roughly 15-epoch no-drift convergence scale, but it is faster than the original 240-epoch control-drift benchmark. The imposed Kerr signal is visibly oscillatory; the local four-control target moves only weakly because Kerr is not a direct command offset in this manifold.

## Calibrated moving optimum

Unlike pure command drift, Kerr drift does not give an analytic command path such as `x_ref + d(e)`. I therefore estimate a local physical compensation direction from finite-difference simulations at the stationary reference. The calibration solves:

```text
J_x s + J_K ~= 0
x_opt_calibrated(e) = x_ref + s [K(e) - K_ref]
```

where the measured signature vector is `[log(T_X), log(T_Z), log(eta)]`. The vector `s = dx/dK` is used only to draw the dashed target curves and quantify tracking; the optimizer is not given `K(e)`, `s`, or `x_opt_calibrated(e)` when choosing candidates.

Calibration details:

```json
{
  "reference_metrics": {
    "g2_real": 1.6192075681,
    "g2_imag": -0.3868491524,
    "eps_d_real": 3.4954392642,
    "eps_d_imag": -0.6602812715,
    "storage_detuning": 0.0,
    "storage_kerr": 0.0,
    "T_X": 0.2697814927247565,
    "T_Z": 26.97715961197772,
    "bias": 99.99633162198069,
    "geo_lifetime": 2.697765443768854,
    "alpha_abs": 1.3171953717325784,
    "nbar": 1.7350036473137254,
    "fit_ok": true,
    "fit_penalty": 0.0,
    "fit_x_r2": 0.9997496779803163,
    "fit_z_r2": 0.9999925879055767,
    "fit_x_rmse": 0.003396849717292561,
    "fit_z_rmse": 0.0006460927029952727,
    "fit_x_hit_tau_bound": false,
    "fit_z_hit_tau_bound": false,
    "valid": true,
    "reason": ""
  },
  "metric_names": [
    "log_T_X",
    "log_T_Z",
    "log_bias"
  ],
  "control_steps": [
    0.035,
    0.025,
    0.06,
    0.05
  ],
  "kerr_step": 0.1,
  "ridge": 0.002,
  "jac_controls": [
    [
      0.555404430505319,
      -0.024039283060690565,
      -0.24355434078689803,
      0.09631032445798651
    ],
    [
      -0.44253245913105665,
      0.11638529793004605,
      0.7820285555180025,
      -0.14277982140761467
    ],
    [
      -0.9979368896363788,
      0.14042458099073218,
      1.0255828963048987,
      -0.23909014586560118
    ]
  ],
  "jac_kerr": [
    0.0048406715498150454,
    -0.050437460147492175,
    -0.05527813169730944
  ],
  "compensation_sensitivity": [
    0.02531794021627227,
    0.013223810456303194,
    0.07549028195211813,
    -0.005335999210745647
  ],
  "linear_residual": [
    -0.00031542087745820187,
    -0.00030498420680180605,
    1.0436670653903024e-05
  ],
  "linear_residual_norm": 0.00043887882203333395
}
```

Bounds check:

```json
{
  "inside_bounds": true,
  "path_min": [
    1.6128668693872967,
    -0.390160962018834,
    3.4765332580436317,
    -0.661667681719462
  ],
  "path_max": [
    1.6257857270661558,
    -0.38341331494336656,
    3.5150533035270137,
    -0.6589449083109908
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

- storage_kerr_bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_bias_vs_epoch.png`
- storage_kerr_lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_lifetimes_vs_epoch.png`
- storage_kerr_reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_reward_or_loss_vs_epoch.png`
- storage_kerr_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_parameters_vs_epoch.png`
- storage_kerr_signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_signal_vs_epoch.png`
- storage_kerr_tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_tracking_error_vs_epoch.png`
- storage_kerr_effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_effective_parameters_vs_epoch.png`
- storage_kerr_optimized_decay_fits_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr_optimized_decay_fits_final.png`

The parameter plot shows that the four-control optimizer finds a valid bias-restoring branch, but not the small finite-difference target branch. The signal plot shows the imposed oscillatory storage Kerr drift.

## Is the result good?

Not fully. The run completed, but the four-control tracker did not meet the requested tracking or bias-band threshold; this is the regime where a fifth explicit Kerr-compensation knob should be tested next.

- warm-up epoch: `15`
- final bias: `100.1`
- median bias after warm-up: `99.9039`
- fraction of post-warm-up epochs with 97 <= eta <= 103: `1`
- median tracking error L2 after warm-up: `1.65783`
- final tracking error L2: `1.62429`
- median distance from stationary reference after warm-up: `1.65817`
- median T_X after warm-up: `0.269159` us
- median T_Z after warm-up: `26.8558` us
- median fit penalty after warm-up: `0`
- median fit R2 after warm-up: X=`0.999808`, Z=`0.999998`

## Four knobs versus a possible fifth knob

This run tests the requested four-knob manifold first. A fifth physically informative knob would be an explicit Kerr compensation command `K_cmd`, with `K_eff(e) = K_drift(e) - K_cmd(e)`. Because the four-knob result is judged by the metrics above, the fifth-knob comparison is only necessary if the four-control tracker misses the bias band or fails to follow the dashed target.

## Limitations and next steps

This is a deterministic Kerr-drift benchmark without stochastic measurement noise. The dashed optimum is a local finite-difference compensation curve, not a global proof of uniqueness. Natural extensions are a five-knob explicit Kerr compensator, Kerr plus detuning drift, measurement SNR degradation, and PPO or other online optimizer comparisons.

## Optimizer details

```json
{
  "initial_mean": [
    1.1263207707628304,
    -0.012747961527778613,
    3.05036283000986,
    0.9954067669639889
  ],
  "reheat_count": 67,
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
    "name": "storage_kerr_reference_signature",
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
  "storage_kerr": {
    "kind": "storage_kerr_four_knob",
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
    "compensation_sensitivity": [
      0.02531794021627227,
      0.013223810456303194,
      0.07549028195211813,
      -0.005335999210745647
    ],
    "kerr_reference": 0.0
  }
}
```
