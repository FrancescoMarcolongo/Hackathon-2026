# Core bias optimizer validation report

## Reproduction command

```bash
python run_core_bias_optimization.py --quick
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

## Reward formula

`reward = FEASIBILITY_BONUS * I[abs(eta/target - 1) <= tol] + W_LIFETIME * 0.5*(log(T_X)+log(T_Z)) - W_BIAS_EXACT*abs(log(eta)-log(target))^2 - W_FIT*fit_penalty - floor_penalty`; `loss_to_minimize = -reward`.

Selected reward config:

```json
{
  "name": "target_band_balanced",
  "variant": "target_band",
  "target_bias": 100.0,
  "bias_tol_rel": 0.03,
  "w_lifetime": 0.65,
  "w_bias_under": 60.0,
  "w_bias_exact": 120.0,
  "w_fit": 2.0,
  "feasibility_bonus": 16.0,
  "min_tx": 0.05,
  "min_tz": 5.0,
  "floor_weight": 12.0
}
```

Target bias: `100`
Target achieved by optimized candidate: `no`

## Baseline vs optimized

| label | g2 | epsilon_d | T_X (us) | T_Z (us) | bias | reward | target achieved |
|---|---:|---:|---:|---:|---:|---:|---:|
| challenge_baseline | `1+0j` | `4+0j` | 0.1816 | 13.93 | 76.68 | -8.16 | False |
| optimization_start | `1+0j` | `2.5+0j` | 0.3044 | 11.78 | 38.72 | -107.6 | False |
| optimized | `0.94569097+0.55446354j` | `2.7405815+1.1669386j` | 0.1943 | 15.11 | 77.79 | -7.317 | False |

## Selected sweep run

```json
{
  "run_id": "run02_target_band_balanced_s0.45_seed0",
  "proxy_simulation": {
    "na": 8,
    "nb": 3,
    "kappa_b": 10.0,
    "kappa_a": 1.0,
    "t_final_x": 1.0,
    "t_final_z": 90.0,
    "n_points": 22,
    "max_tau_factor": 80.0,
    "alpha_margin": 2.5
  },
  "final_simulation": {
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
  "sigma0": 0.45,
  "seed": 0,
  "epochs": 8,
  "population": 4
}
```

## Figures

- bias_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/final_bias_vs_epoch.png`
- lifetimes_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/final_lifetimes_vs_epoch.png`
- reward_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/final_reward_or_loss_vs_epoch.png`
- parameters_vs_epoch: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/final_parameters_vs_epoch.png`
- baseline_decay_fits: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/baseline_decay_fits.png`
- optimized_decay_fits: `/Users/lorenzobiagi/Desktop/Hackathon-2026/BiagiBoss/Baseline/figures/optimized_decay_fits.png`

## Notes

- Validated incumbent reward improved from -107.6 to -7.317 and stabilized in the final epochs.
- The sweep used exact-target / target-band rewards; candidates outside the target band are not treated as feasible even when eta is above the target.
- Final exponential fits are well conditioned: optimized R2 values are 0.999671 for X and 0.997561 for Z.
- The optimization start was deliberately set below target: eta_start=38.72 with g2=1 and epsilon_d=2.5.
- A final target-band refinement at the notebook truncation scanned epsilon_d amplitude only; selected scale=1.024.
- The final optimized candidate did not satisfy the target-bias feasibility criterion.
