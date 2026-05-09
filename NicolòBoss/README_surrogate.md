# Cat-Qubit Surrogate Reward System

This package adds a modular learned reward path for online cat-qubit stabilization. It predicts `log(T_X)` and `log(T_Z)` from cheap observables so an optimizer can evaluate many candidates without running a full lifetime benchmark every time.

The notebook should stay thin: import this package, provide a concrete `SimulationAdapter`, and call the dataset, training, reward, and optimization functions from normal Python modules.

## Architecture

- `cat_surrogate.config`: dataclass configuration for surrogate training and optimization.
- `cat_surrogate.params`: four-real knob packing plus analytic derived features.
- `cat_surrogate.adapters`: the physics boundary. `SimulationAdapter` contains TODO hooks for notebook/Dynamiqs functions. `DummyAdapter` is deterministic and non-physical.
- `cat_surrogate.features`: combines parameter, static, and short-time observables into flat scalar features.
- `cat_surrogate.dataset`: builds supervised datasets with expensive labels.
- `cat_surrogate.model`: PyTorch MLP ensemble trained on `log(T_X), log(T_Z)`.
- `cat_surrogate.reward`: true, short-time proxy, and learned surrogate rewards.
- `cat_surrogate.optimize`: ask/tell optimizer interface and online loop with periodic true validation.
- `cat_surrogate.plotting`: matplotlib diagnostics.

## Install

```bash
python -m pip install -r requirements.txt
```

## Connect Notebook Physics

Subclass `SimulationAdapter` in a notebook or support file:

```python
from cat_surrogate import SimulationAdapter

class NotebookAdapter(SimulationAdapter):
    def compute_static_observables(self, theta, config):
        # TODO: call existing notebook/Dynamiqs steady-state code.
        return {...}

    def compute_short_time_observables(self, theta, config):
        # TODO: call cheap short-time evolution code.
        return {...}

    def expensive_lifetime_benchmark(self, theta, config):
        # TODO: call full T_X/T_Z simulation and exponential fitting.
        return {"T_X": ..., "T_Z": ..., "eta": ..., "log_T_X": ..., "log_T_Z": ..., "log_eta": ...}
```

Do not put the surrogate implementation in the notebook. Keep only imports, adapter wiring, and small experiment calls there.

## Generate A Dataset

```bash
python scripts/generate_dataset.py --n-samples 512 --output data/surrogate_dataset.parquet
```

The script uses `DummyAdapter` by default and prints a warning. Replace it with a concrete adapter before producing physical results.

## Train The Surrogate

```bash
python scripts/train_surrogate.py \
  --dataset data/surrogate_dataset.parquet \
  --output artifacts/surrogate_bundle.pt \
  --n-models 5
```

The saved bundle contains the ensemble, feature scaler, feature column order, config metadata, and training histories.

## Run Online Optimization

```bash
python scripts/run_online_optimization.py \
  --bundle artifacts/surrogate_bundle.pt \
  --output logs/online_optimization.csv
```

The loop logs surrogate reward, predicted lifetimes, predicted eta, uncertainty, physics penalty, and periodic true validation when the adapter provides it.

## Notebook Usage

```python
from cat_surrogate import SurrogateConfig, compute_cheap_observables, load_surrogate_bundle, surrogate_reward

config = SurrogateConfig()
adapter = NotebookAdapter()
bundle = load_surrogate_bundle("artifacts/surrogate_bundle.pt")
reward, info = surrogate_reward(theta, config, bundle, adapter, return_info=True)
```

## Important Warning

The learned reward can be exploited by the optimizer, especially away from the training distribution. Keep periodic calls to `expensive_lifetime_benchmark`, watch ensemble uncertainty, add new true-labeled data near promising regions, and retrain when validation and surrogate predictions drift apart.

## Suggested `.gitignore`

```gitignore
# Generated data, models, and logs
data/
artifacts/
logs/
*.pt
*.pth
*.ckpt
*.joblib
*.parquet

# Python caches and test/build artifacts
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/
build/
dist/
*.egg-info/

# Virtual environments
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# Dependency manager files, if generated locally
Pipfile.lock
poetry.lock
uv.lock
```
