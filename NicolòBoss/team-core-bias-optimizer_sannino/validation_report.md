# Core bias optimizer validation report

## Reproduction command

```bash
python run_core_bias_optimization.py
```

## Package versions

- python: 3.13.1
- platform: macOS-26.3.1-arm64-arm-64bit-Mach-O
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
  "name": "exact_target_strict",
  "variant": "exact_target",
  "target_bias": 100.0,
  "bias_tol_rel": 0.03,
  "w_lifetime": 0.25,
  "w_bias_under": 60.0,
  "w_bias_exact": 180.0,
  "w_fit": 2.0,
  "feasibility_bonus": 0.0,
  "min_tx": 0.05,
  "min_tz": 5.0,
  "floor_weight": 12.0
}
```

Target bias: `100`
Target achieved by optimized candidate: `yes`

## Baseline vs optimized

| label | g2 | epsilon_d | T_X (us) | T_Z (us) | bias | reward | target achieved |
|---|---:|---:|---:|---:|---:|---:|---:|
| challenge_baseline | `1+0j` | `4+0j` | 0.1771 | 57.05 | 322.1 | -246 | False |
| optimization_start | `1+0j` | `2.5+0j` | 0.3062 | 12.36 | 40.35 | -148.1 | False |
| optimized | `1.3285188-0.7782763j` | `3.0005451+1.754515j` | 0.2691 | 26.87 | 99.84 | 0.2468 | True |

## Selected sweep run

```json
{
  "run_id": "run04_exact_target_strict_s0.45_seed0",
  "proxy_simulation": {
    "na": 10,
    "nb": 3,
    "kappa_b": 10.0,
    "kappa_a": 1.0,
    "t_final_x": 1.0,
    "t_final_z": 130.0,
    "n_points": 28,
    "max_tau_factor": 80.0,
    "alpha_margin": 2.5
  },
  "final_simulation": {
    "na": 15,
    "nb": 5,
    "kappa_b": 10.0,
    "kappa_a": 1.0,
    "t_final_x": 1.2,
    "t_final_z": 260.0,
    "n_points": 60,
    "max_tau_factor": 80.0,
    "alpha_margin": 2.5
  },
  "sigma0": 0.45,
  "seed": 0,
  "epochs": 34,
  "population": 8
}
```

## Figures

- bias_vs_epoch: `/Users/nicolobattocletti/Desktop/Hackathon-2026/NicolòBoss/team-core-bias-optimizer_sannino/figures/final_bias_vs_epoch.png`
- lifetimes_vs_epoch: `/Users/nicolobattocletti/Desktop/Hackathon-2026/NicolòBoss/team-core-bias-optimizer_sannino/figures/final_lifetimes_vs_epoch.png`
- reward_vs_epoch: `/Users/nicolobattocletti/Desktop/Hackathon-2026/NicolòBoss/team-core-bias-optimizer_sannino/figures/final_reward_or_loss_vs_epoch.png`
- parameters_vs_epoch: `/Users/nicolobattocletti/Desktop/Hackathon-2026/NicolòBoss/team-core-bias-optimizer_sannino/figures/final_parameters_vs_epoch.png`
- baseline_decay_fits: `/Users/nicolobattocletti/Desktop/Hackathon-2026/NicolòBoss/team-core-bias-optimizer_sannino/figures/baseline_decay_fits.png`
- optimized_decay_fits: `/Users/nicolobattocletti/Desktop/Hackathon-2026/NicolòBoss/team-core-bias-optimizer_sannino/figures/optimized_decay_fits.png`

## Notes

- Validated incumbent reward improved from -148.1 to 0.2468 and stabilized in the final epochs.
- The sweep used exact-target / target-band rewards; candidates outside the target band are not treated as feasible even when eta is above the target.
- Final exponential fits are well conditioned: optimized R2 values are 0.999830 for X and 1.000000 for Z.
- The optimization start was deliberately set below target: eta_start=40.35 with g2=1 and epsilon_d=2.5.
- A final target-band refinement at the notebook truncation scanned epsilon_d amplitude only; selected scale=0.992.
- Optimized T_Z is below baseline; inspect the lifetime tradeoff before using this candidate.
