# Complex g2-gain drift: SepCMA vs BLT

## Reproduction command

```bash
python run_g2_gain_drift_comparison.py --epochs 72 --blt-population 24 --sim-preset medium
```

## Drift model

This model is the challenge notebook's suggested amplitude drift in the buffer/two-photon control chain, implemented as a complex transfer function on the driven controls:

```text
G(e) = (1 + r(e)) exp(i theta(e))
g2_eff(e) = G(e) g2_cmd(e)
epsilon_d_eff(e) = G(e) epsilon_d_cmd(e)
r(e) = A_r [0.65 sin(2 pi e/P + 0.15) + 0.35 sin(4 pi e/P + 1.40)]
theta(e) = A_phi [0.60 sin(2 pi e/P + 1.10) + 0.40 sin(4 pi e/P + 2.65)]
x_opt_true(e) = [Re(g2_ref/G(e)), Im(g2_ref/G(e)), Re(epsilon_ref/G(e)), Im(epsilon_ref/G(e))]
```

- P: `72` epochs
- highest harmonic period: `36` epochs
- amplitude modulation A_r: `0.1`
- phase modulation A_phi: `0.18` rad
- x_reference source: `results/storage_detuning_summary_metrics.json`

Both optimizers start from the same seeded random valid command. The shared reward is the target-band lifetime reward plus a stationary reference-signature penalty on `T_X`, `T_Z`, and `alpha_abs`. The BLT candidate generator does not receive `G(e)`, `x_ref`, or `x_opt_true(e)` as candidate inputs; those are used by the simulator and by the offline error plots.

## Random start

```json
{
  "start_seed": 17,
  "attempt": 1,
  "x0": [
    2.1246570648755934,
    -0.5763457450125181,
    3.6771738273553223,
    -0.5276802333429149
  ],
  "bias_at_epoch0": 71.97016822104298,
  "T_X_at_epoch0": 0.3527381551235616,
  "T_Z_at_epoch0": 25.38662436222308,
  "tracking_error_l2_at_epoch0": 0.9394288045551682
}
```

## Results after warm-up

| metric | SepCMA baseline | BLT optimized |
|---|---:|---:|
| final bias | 100.124 | 98.842 |
| median bias | 99.9072 | 99.9018 |
| success-band fraction | 0.9273 | 1 |
| median tracking L2 | 0.622841 | 0.485981 |
| median effective L2 | 0.621701 | 0.485606 |
| median T_X (us) | 0.277402 | 0.269009 |
| median T_Z (us) | 27.8249 | 26.914 |

Interpretation: lower tracking/effective error means the commanded controls better compensate the drifting complex gain and keep the physical Hamiltonian near the stationary cat optimum.

## Figures

- bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_bias_vs_epoch.png`
- lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_lifetimes_vs_epoch.png`
- reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_reward_or_loss_vs_epoch.png`
- parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_parameters_vs_epoch.png`
- signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_signal_vs_epoch.png`
- tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_tracking_error_vs_epoch.png`
- effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_effective_parameters_vs_epoch.png`
- g2_gain_drift_comparison_baseline_decay_fits_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_baseline_decay_fits_final.png`
- g2_gain_drift_comparison_blt_decay_fits_final: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/g2_gain_drift_comparison_blt_decay_fits_final.png`

## Configuration

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
    "name": "g2_gain_reference_signature",
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
  "g2_gain_drift": {
    "kind": "complex_g2_gain",
    "affect_epsilon_d": true,
    "period_epochs": 72.0,
    "bandwidth": 2,
    "amplitude_scale": 1.0,
    "amplitude_modulation": 0.1,
    "phase_modulation": 0.18,
    "amplitude_weights": [
      0.65,
      0.35
    ],
    "phase_weights": [
      0.6,
      0.4
    ],
    "amplitude_phases": [
      0.15,
      1.4
    ],
    "phase_phases": [
      1.1,
      2.65
    ],
    "x_reference": [
      1.6192075681484381,
      -0.38684915244951523,
      3.495439264194342,
      -0.6602812715117858
    ]
  },
  "bounds_check": {
    "inside_bounds": true,
    "original_amplitude_scale": 1.0,
    "amplitude_scale": 1.0,
    "was_scaled": false,
    "path_min": [
      1.4831364838943049,
      -0.6048115141979589,
      3.219188661791165,
      -1.1323244941454773
    ],
    "path_max": [
      1.8289989630085375,
      -0.11750859708264307,
      3.928653475450775,
      -0.07115273340987026
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
  },
  "baseline_metadata": {
    "optimizer": "SepCMA",
    "initial_command": [
      2.1246570648755934,
      -0.5763457450125181,
      3.6771738273553223,
      -0.5276802333429149
    ],
    "population": 8,
    "sigma0": 0.42,
    "sigma_floor": 0.045,
    "seed": 23,
    "reheat_count": 62
  },
  "blt_metadata": {
    "optimizer": "BLT_TRUST_REGION_COORDINATE_SCAN",
    "initial_command": [
      2.1246570648755934,
      -0.5763457450125181,
      3.6771738273553223,
      -0.5276802333429149
    ],
    "population": 24,
    "seed": 23,
    "initial_trust": 0.64,
    "min_trust": 0.08,
    "max_trust": 0.68,
    "continuity_weight": 2.0,
    "notes": "No x_reference, gain, drift vector, or true optimum is injected as a candidate."
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
