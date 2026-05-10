# Flexible Reward Experiment

This folder contains a separate, notebook-friendly experiment for testing more flexible cat-qubit reward designs. The original challenge notebook and the existing baseline code are not edited.

Outputs are written to timestamped run folders:

```text
reward_function_experiments/results/run_YYYYMMDD_HHMMSS/
```

Each run folder contains the candidate CSV, config JSON, summary JSON, and plots, so previous results are not overwritten.

## Main Files

- `flexible_cat_reward.py`: reusable simulation, diagnostic, reward, save, and plot helpers.
- `flexible_reward_experiment.ipynb`: smoke test plus a small NumPy-only evolutionary optimizer.
- `run_reward_comparison.py`: compares the flexible reward with the old simple fitted-lifetime reward.
- `run_no_fit_proxy_search.py`: compares the old fitted-lifetime reward with faster no-fit proxy rewards.

## Editing Rewards

The main function to edit is:

```python
cat_reward_from_metrics(metrics, config=None)
```

It receives already-computed metrics and only combines them into a scalar reward. By default it extends the simple lifetime reward:

```python
reward = log(Tx) + log(Tz) - lambda_bias * log(bias / eta_target)**2
```

Optional reward terms can be enabled in the config:

- alpha target penalty with `use_alpha_penalty`
- photon number penalty with `use_nbar_penalty`
- parity contrast bonus with `use_parity_bonus`

The optional terms are disabled by default.

## No-Fit Proxy Reward

For faster optimization, `cat_proxy_loss` avoids exponential fitting. It simulates short endpoint logical contrasts and converts them into lifetime proxies:

```python
T_proxy = -tfinal / log(contrast)
```

Available proxy modes:

- `single_endpoint`: uses `+x` and `+z` endpoint observables.
- `two_sided_flip_syndrome`: uses `+/-x` and `+/-z` contrasts to reduce offset sensitivity.

The proxy is meant for searching. The comparison runner validates each epoch's best candidate with the slower fitted-lifetime metric.

## Running

From the project root, open and run:

```text
reward_function_experiments/flexible_reward_experiment.ipynb
```

The notebook creates a timestamped run directory, evaluates `x0 = jnp.array([1.0, 0.0, 4.0, 0.0])`, runs a small conservative optimizer, and saves all artifacts under that run directory.

To run the no-fit proxy comparison from the project root:

```bash
../.venv/bin/python reward_function_experiments/run_no_fit_proxy_search.py
```
