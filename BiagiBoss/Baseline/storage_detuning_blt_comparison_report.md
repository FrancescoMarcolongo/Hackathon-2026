# Storage-detuning BLT tracker comparison

## Reproduction command

```bash
python run_storage_detuning_blt_tracking.py --quick --epochs 6 --population 8 --no-decay-snapshots
```

## Analytic model

```text
Delta_eff(e) = Delta_cmd(e) - Delta_drift(e)
H_det(e) = Delta_eff(e) a^dag a
x_opt_true(e) = [x_ref, Delta_drift(e)]
```

Unlike the Kerr four-control case, this benchmark has an exact analytic optimum for the fifth knob. If the optimizer commands `Delta_cmd = Delta_drift`, the Hamiltonian sees zero residual detuning and returns to the stationary reference.

- candidate mode: `fair_start`
- P: `96` epochs
- bandwidth: `3`
- amplitude: `0.08`

The optimizer does not receive `Delta_drift(e)` or `x_opt_true(e)`. In fair-start mode it also does not inject the stationary `x_ref` command; in calibrated-tracker mode it uses `x_ref` as a fixed no-drift calibration prior.

## Results

- final bias: `38.7157`
- median bias after warm-up: `38.7155`
- post-warm-up success-band fraction: `0`
- median tracking L2 after warm-up: `1.6381`
- median residual detuning after warm-up: `0.00100432`
- median T_X after warm-up: `0.304386` us
- median T_Z after warm-up: `11.7845` us
- tracking good: `False`

SepCMA detuning baseline reference from the current CSV:

- baseline final bias: `100.09`
- baseline median tracking L2 after warm-up: `0.042322`
- baseline median residual detuning after warm-up: `0.022941`

## Figures

- storage_detuning_blt_bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_bias_vs_epoch.png`
- storage_detuning_blt_lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_lifetimes_vs_epoch.png`
- storage_detuning_blt_reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_reward_or_loss_vs_epoch.png`
- storage_detuning_blt_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_parameters_vs_epoch.png`
- storage_detuning_blt_signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_signal_vs_epoch.png`
- storage_detuning_blt_tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_tracking_error_vs_epoch.png`
- storage_detuning_blt_effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_effective_parameters_vs_epoch.png`
- bias_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_vs_sepcma_bias_comparison.png`
- lifetimes_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_vs_sepcma_lifetimes_comparison.png`
- detuning_command_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_vs_sepcma_detuning_command_comparison.png`
- tracking_error_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_detuning_blt_vs_sepcma_tracking_error_comparison.png`

## Run configuration

```json
{
  "simulation": {
    "na": 8,
    "nb": 3,
    "kappa_b": 10.0,
    "kappa_a": 1.0,
    "t_final_x": 1.0,
    "t_final_z": 90.0,
    "n_points": 24,
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
      1.57798161,
      -0.22211626,
      3.2540442,
      1.3158266
    ],
    "detuning_reference": 0.0
  },
  "bounds_check": {
    "inside_bounds": true,
    "path_min": [
      1.57798161,
      -0.22211626,
      3.2540442,
      1.3158266,
      0.03594605284682649
    ],
    "path_max": [
      1.57798161,
      -0.22211626,
      3.2540442,
      1.3158266,
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
  },
  "optimizer_metadata": {
    "optimizer": "BLT_DETUNING_COORDINATE_SCAN",
    "candidate_mode": "fair_start",
    "initial_mean": [
      1.0,
      0.0,
      2.5,
      0.0,
      0.036178668944693376
    ],
    "population": 8,
    "seed": 13,
    "local_detuning_step": 0.03,
    "notes": "fair_start does not inject x_ref or Delta_drift. calibrated_tracker injects stationary x_ref candidates and is a post-calibration tracker."
  },
  "package_versions": {
    "python": "3.13.2",
    "platform": "macOS-14.6-arm64-arm-64bit-Mach-O",
    "dynamiqs": "0.3.4",
    "jax": "0.6.2",
    "cmaes": "0.13.0",
    "scipy": "1.17.1",
    "matplotlib": "3.10.9",
    "numpy": "2.4.4"
  }
}
```
