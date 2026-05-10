# Reward function baseline

This folder is a separate sandbox for reward-function experiments. It does not edit or depend on modifying `1-challenge.ipynb`.

`simple_lifetime_reward.py` contains the core lifetime simulation helpers and the editable reward function.

`simple_lifetime_reward_baseline.ipynb` contains a small smoke test for the first simple lifetime-based baseline.

The function to edit later is:

```python
cat_reward_from_lifetimes
```

To run a configurable CMA-ES training loop and save a loss plot, edit the
`USER_CONFIG` block at the top of `train_lifetime_reward.py`, then run:

```bash
python reward_function_baseline/train_lifetime_reward.py
```

You can also override values from the command line:

```bash
python reward_function_baseline/train_lifetime_reward.py \
  --generations 5 \
  --population-size 4 \
  --sigma 0.25 \
  --x0 1.0 0.0 4.0 0.0
```

Each run writes `config.json`, `history.csv`, `summary.json`, and `loss_curve.png` under `reward_function_baseline/runs/`.
