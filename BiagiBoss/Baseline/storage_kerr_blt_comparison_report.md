# Storage-Kerr BLT tracker comparison

## Reproduction command

```bash
python run_storage_kerr_blt_tracking.py --epochs 20 --population 33 --sim-preset medium --warmup-epoch 10 --cat-reference-weight 0.0 --reference-signature-weight 0.0 --no-decay-snapshots
```

## Model validity checks

The explicit Kerr knob is additive in the Hamiltonian:

```text
K_eff(e) = K_drift(e) + K_cmd(e)
H_K(e) = -0.5 K_eff(e) n_a (n_a - I)
x_opt_true(e) = [x_ref, -K_drift(e)]
```

This is an analytic compensation optimum for the five-knob model because setting `K_cmd = -K_drift` makes `K_eff = 0` and returns the Hamiltonian to the stationary reference. It is not a proof that the four original cat knobs have a unique global optimum under Kerr drift.

- maximum bias deviation for the sign-correct cancellation check: `0`
- P: `100` epochs
- bandwidth: `2`
- highest harmonic period: `50` epochs
- Kerr amplitude: `0.3`

The BLT-style tracker does not receive `K_drift(e)` or `x_opt_true(e)` when choosing candidates. In `fair_start` mode it also does not inject the stationary `x_ref` command; it probes measured rewards with current/past commands, local coordinate moves, local `K_cmd` grids, and random exploration. In `prior_assisted` mode it additionally injects stationary `x_ref` candidates and should be interpreted as a post-calibration tracker.

## Results

- final bias: `100.059`
- median bias after warm-up: `100.026`
- post-warm-up success-band fraction: `1`
- median tracking L2 after warm-up: `2.11683`
- median residual Kerr after warm-up: `0.234318`
- median T_X after warm-up: `0.353275` us
- median T_Z after warm-up: `35.3357` us
- tracking good: `False`

SepCMA baseline reference from the current Kerr-compensation CSV:

- baseline final bias: `99.849`
- baseline median tracking L2 after warm-up: `0.133378`
- baseline median residual Kerr after warm-up: `0.0950147`

## Figures

- storage_kerr5_blt_bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_bias_vs_epoch.png`
- storage_kerr5_blt_lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_lifetimes_vs_epoch.png`
- storage_kerr5_blt_reward_or_loss_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_reward_or_loss_vs_epoch.png`
- storage_kerr5_blt_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_parameters_vs_epoch.png`
- storage_kerr5_blt_signal_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_signal_vs_epoch.png`
- storage_kerr5_blt_tracking_error_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_tracking_error_vs_epoch.png`
- storage_kerr5_blt_effective_parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_effective_parameters_vs_epoch.png`
- bias_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_vs_sepcma_bias_comparison.png`
- lifetimes_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_vs_sepcma_lifetimes_comparison.png`
- kerr_command_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_vs_sepcma_kerr_command_comparison.png`
- tracking_error_comparison: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/storage_kerr5_blt_vs_sepcma_tracking_error_comparison.png`

## Sign-consistency table

```json
{
  "reference": {
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
  "rows": [
    {
      "epoch": 0,
      "case": "no_command",
      "k_drift": 0.0,
      "k_cmd": 0.0,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 0,
      "case": "correct_cancel_command",
      "k_drift": 0.0,
      "k_cmd": -0.0,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 0,
      "case": "wrong_flipped_command",
      "k_drift": 0.0,
      "k_cmd": 0.0,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 10,
      "case": "no_command",
      "k_drift": 0.21120961703199928,
      "k_cmd": 0.0,
      "k_eff": 0.21120961703199928,
      "T_X": 0.26855995511183417,
      "T_Z": 26.017559614602824,
      "bias": 96.87803084331225,
      "bias_error_vs_reference": -3.1183007786684414
    },
    {
      "epoch": 10,
      "case": "correct_cancel_command",
      "k_drift": 0.21120961703199928,
      "k_cmd": -0.21120961703199928,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 10,
      "case": "wrong_flipped_command",
      "k_drift": 0.21120961703199928,
      "k_cmd": 0.21120961703199928,
      "k_eff": 0.42241923406399856,
      "T_X": 0.2640205955545484,
      "T_Z": 23.972918679405645,
      "bias": 90.79942657144974,
      "bias_error_vs_reference": -9.196905050530944
    },
    {
      "epoch": 20,
      "case": "no_command",
      "k_drift": 0.25044291354428877,
      "k_cmd": 0.0,
      "k_eff": 0.25044291354428877,
      "T_X": 0.26798693675024016,
      "T_Z": 25.704562031476996,
      "bias": 95.91722023164608,
      "bias_error_vs_reference": -4.079111390334603
    },
    {
      "epoch": 20,
      "case": "correct_cancel_command",
      "k_drift": 0.25044291354428877,
      "k_cmd": -0.25044291354428877,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 20,
      "case": "wrong_flipped_command",
      "k_drift": 0.25044291354428877,
      "k_cmd": 0.25044291354428877,
      "k_eff": 0.5008858270885775,
      "T_X": 0.2613579203459815,
      "T_Z": 23.04901788706204,
      "bias": 88.1894754004398,
      "bias_error_vs_reference": -11.806856221540883
    },
    {
      "epoch": 30,
      "case": "no_command",
      "k_drift": 0.13758814510413395,
      "k_cmd": 0.0,
      "k_eff": 0.13758814510413395,
      "T_X": 0.2693323407475114,
      "T_Z": 26.49866162275508,
      "bias": 98.38648247444056,
      "bias_error_vs_reference": -1.6098491475401318
    },
    {
      "epoch": 30,
      "case": "correct_cancel_command",
      "k_drift": 0.13758814510413395,
      "k_cmd": -0.13758814510413395,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 30,
      "case": "wrong_flipped_command",
      "k_drift": 0.13758814510413395,
      "k_cmd": 0.13758814510413395,
      "k_eff": 0.2751762902082679,
      "T_X": 0.26756551599133277,
      "T_Z": 25.489048362308342,
      "bias": 95.2628303683723,
      "bias_error_vs_reference": -4.733501253608381
    },
    {
      "epoch": 45,
      "case": "no_command",
      "k_drift": 0.006612082632411848,
      "k_cmd": 0.0,
      "k_eff": 0.006612082632411848,
      "T_X": 0.269788811490404,
      "T_Z": 26.96744313492402,
      "bias": 99.95760382332688,
      "bias_error_vs_reference": -0.038727798653809486
    },
    {
      "epoch": 45,
      "case": "correct_cancel_command",
      "k_drift": 0.006612082632411848,
      "k_cmd": -0.006612082632411848,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 45,
      "case": "wrong_flipped_command",
      "k_drift": 0.006612082632411848,
      "k_cmd": 0.006612082632411848,
      "k_eff": 0.013224165264823695,
      "T_X": 0.26979305629254685,
      "T_Z": 26.956394695393065,
      "bias": 99.91507960146767,
      "bias_error_vs_reference": -0.08125202051301983
    },
    {
      "epoch": 60,
      "case": "no_command",
      "k_drift": -0.028606765903329726,
      "k_cmd": 0.0,
      "k_eff": -0.028606765903329726,
      "T_X": 0.2697174463372169,
      "T_Z": 27.003292952913895,
      "bias": 100.11696803310514,
      "bias_error_vs_reference": 0.12063641112445112
    },
    {
      "epoch": 60,
      "case": "correct_cancel_command",
      "k_drift": -0.028606765903329726,
      "k_cmd": 0.028606765903329726,
      "k_eff": 0.0,
      "T_X": 0.2697814927247565,
      "T_Z": 26.97715961197772,
      "bias": 99.99633162198069,
      "bias_error_vs_reference": 0.0
    },
    {
      "epoch": 60,
      "case": "wrong_flipped_command",
      "k_drift": -0.028606765903329726,
      "k_cmd": -0.028606765903329726,
      "k_eff": -0.05721353180665945,
      "T_X": 0.26959920074369526,
      "T_Z": 27.00370727788956,
      "bias": 100.16241592482191,
      "bias_error_vs_reference": 0.16608430284122733
    }
  ]
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
  "bounds_check": {
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
      0.0
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
  },
  "optimizer_metadata": {
    "optimizer": "BLT_DIRECT_K_SCAN",
    "initial_mean": [
      1.0,
      0.0,
      2.5,
      0.0,
      0.0
    ],
    "population": 33,
    "seed": 7,
    "local_k_step": 0.055,
    "candidate_mode": "fair_start",
    "notes": "Uses measured reward probes plus local momentum. In fair_start mode it does not inject x_ref candidates and does not use K_drift(epoch) or x_opt_true(epoch) for candidate generation."
  },
  "candidate_mode": "fair_start",
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
