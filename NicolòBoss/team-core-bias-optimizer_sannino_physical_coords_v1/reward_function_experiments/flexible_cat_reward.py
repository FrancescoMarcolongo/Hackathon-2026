"""Flexible cat-qubit reward utilities.

This module is intentionally notebook-friendly: the physical simulation and
diagnostics are computed once, then ``cat_reward_from_metrics`` combines those
already-computed numbers into a scalar reward that is easy to edit by hand.
"""

from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

EPS = 1.0e-12


def _dynamiqs():
    import dynamiqs as dq

    return dq


def default_reward_config() -> dict[str, Any]:
    return {
        "eta_target": 100.0,
        "lambda_bias": 2.0,
        "use_alpha_penalty": False,
        "alpha_target": 2.0,
        "lambda_alpha": 0.1,
        "use_nbar_penalty": False,
        "nbar_max": 8.0,
        "lambda_nbar": 0.1,
        "use_parity_bonus": False,
        "lambda_parity": 0.1,
        "tfinal_z": 200.0,
        "tfinal_x": 2.0,
        "proxy_tfinal_z": 30.0,
        "proxy_tfinal_x": 0.6,
        "proxy_n_points": 2,
        "proxy_mode": "two_sided_flip_syndrome",
        "lambda_syndrome_floor": 0.25,
        "na": 15,
        "nb": 5,
        "kappa_a": 1.0,
        "kappa_b": 10.0,
        "n_points": 60,
    }


def merged_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_reward_config()
    if config:
        cfg.update(config)
    return cfg


def create_run_dir(
    base_dir: str | Path = "reward_function_experiments/results",
    label: str | None = None,
) -> Path:
    """Create and return a unique timestamped run directory."""

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [f"run_{timestamp}"]
    if label:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_.-")
        if safe_label:
            parts.append(safe_label[:60])
    run_dir = base / "_".join(parts)

    # If two runs start in the same second, keep both.
    if run_dir.exists():
        suffix = 1
        candidate = Path(f"{run_dir}_{suffix:02d}")
        while candidate.exists():
            suffix += 1
            candidate = Path(f"{run_dir}_{suffix:02d}")
        run_dir = candidate

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def unpack_knobs(x: Any) -> tuple[Any, Any]:
    """Map ``[Re(g2), Im(g2), Re(eps_d), Im(eps_d)]`` to complex knobs."""

    x_arr = jnp.asarray(x)
    if x_arr.shape != (4,):
        raise ValueError("x must have shape (4,): [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]")
    g2 = x_arr[0] + 1j * x_arr[1]
    eps_d = x_arr[2] + 1j * x_arr[3]
    return g2, eps_d


def build_cat_system(
    x: Any,
    drift: dict[str, Any] | None = None,
    na: int = 15,
    nb: int = 5,
    kappa_a: float = 1.0,
    kappa_b: float = 10.0,
) -> dict[str, Any]:
    """Construct the Dynamiqs cat system for one optimizer point."""

    dq = _dynamiqs()
    g2, eps_d = unpack_knobs(x)
    if drift is not None:
        g2 = g2 * drift.get("g2_prefactor", 1.0)
        eps_d = eps_d * drift.get("epsd_prefactor", 1.0)

    a = dq.tensor(dq.destroy(na), dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    eps_2 = 2 * g2 * eps_d / kappa_b
    kappa_2 = 4 * jnp.abs(g2) ** 2 / kappa_b
    alpha_estimate = jnp.sqrt(
        jnp.maximum(
            1e-9,
            2 / kappa_2 * (jnp.abs(eps_2) - kappa_a / 4),
        )
    )

    H = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )

    loss_ops = [jnp.sqrt(kappa_b) * b, jnp.sqrt(kappa_a) * a]

    plus_z = dq.coherent(na, alpha_estimate)
    minus_z = dq.coherent(na, -alpha_estimate)
    plus_x = (plus_z + minus_z) / jnp.sqrt(2)
    minus_x = (plus_z - minus_z) / jnp.sqrt(2)

    states = {
        "+z": dq.tensor(plus_z, dq.fock(nb, 0)),
        "-z": dq.tensor(minus_z, dq.fock(nb, 0)),
        "+x": dq.tensor(plus_x, dq.fock(nb, 0)),
        "-x": dq.tensor(minus_x, dq.fock(nb, 0)),
    }

    n_storage = dq.destroy(na).dag() @ dq.destroy(na)
    parity_storage = (1j * jnp.pi * n_storage).expm()
    sx = dq.tensor(parity_storage, dq.eye(nb))
    sz_storage = plus_z @ plus_z.dag() - minus_z @ minus_z.dag()
    sz = dq.tensor(sz_storage, dq.eye(nb))
    n_total = dq.tensor(n_storage, dq.eye(nb)) + dq.tensor(
        dq.eye(na),
        dq.destroy(nb).dag() @ dq.destroy(nb),
    )
    parity = dq.tensor(parity_storage, dq.eye(nb))

    return {
        "g2": g2,
        "eps_d": eps_d,
        "alpha": alpha_estimate,
        "H": H,
        "loss_ops": loss_ops,
        "states": states,
        "observables": {
            "sx": sx,
            "sz": sz,
            "n_total": n_total,
            "parity": parity,
        },
        "a": a,
        "b": b,
        "n_total": n_total,
        "parity": parity,
        "sx": sx,
        "sz": sz,
    }


def _exp_model(params: np.ndarray, t: np.ndarray) -> np.ndarray:
    amp, tau, offset = params
    return amp * np.exp(-t / tau) + offset


def robust_exp_fit(
    times: Any,
    values: Any,
    max_tau: float | None = None,
    f_scale: float = 0.05,
) -> dict[str, Any]:
    """Robustly fit ``y = A exp(-t / tau) + C`` using SciPy."""

    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if max_tau is None:
        max_tau = max(100.0, 80.0 * float(t[-1] - t[0])) if len(t) else 100.0
    if len(t) < 4 or len(t) != len(y) or not np.all(np.isfinite(y)):
        return {
            "success": False,
            "reason": "bad input",
            "params": np.array([np.nan, np.nan, np.nan], dtype=float),
            "fit": np.full_like(t, np.nan, dtype=float),
            "r2": np.nan,
            "rmse": np.nan,
            "hit_tau_bound": True,
        }

    amp0 = float(max(1e-4, y[0] - y[-1], np.max(y) - np.min(y)))
    tau0 = float(max(1e-3, 0.5 * (t[-1] - t[0])))
    offset0 = float(y[-1])

    def residual(params: np.ndarray) -> np.ndarray:
        return _exp_model(params, t) - y

    try:
        res = least_squares(
            residual,
            np.array([amp0, tau0, offset0], dtype=float),
            bounds=([0.0, 1e-6, -2.0], [3.0, float(max_tau), 2.0]),
            loss="soft_l1",
            f_scale=float(f_scale),
            max_nfev=600,
        )
    except Exception as exc:
        return {
            "success": False,
            "reason": f"least_squares failed: {exc}",
            "params": np.array([np.nan, np.nan, np.nan], dtype=float),
            "fit": np.full_like(t, np.nan, dtype=float),
            "r2": np.nan,
            "rmse": np.nan,
            "hit_tau_bound": True,
        }

    fit = _exp_model(res.x, t)
    residuals = fit - y
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - sse / sst) if sst > EPS else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2)))
    hit_tau_bound = bool(res.x[1] > 0.98 * float(max_tau))
    success = bool(res.success and np.isfinite(res.x[1]) and res.x[1] > 0 and not hit_tau_bound)
    return {
        "success": success,
        "reason": "" if success else str(res.message),
        "params": np.asarray(res.x, dtype=float),
        "fit": np.asarray(fit, dtype=float),
        "r2": r2,
        "rmse": rmse,
        "hit_tau_bound": hit_tau_bound,
        "cost": float(res.cost),
    }


def measure_lifetime_cat(
    initial_state: str | Any,
    tfinal: float,
    x: Any,
    drift: dict[str, Any] | None = None,
    na: int = 15,
    nb: int = 5,
    kappa_a: float = 1.0,
    kappa_b: float = 10.0,
    n_points: int = 60,
) -> dict[str, Any]:
    """Simulate one logical decay curve and return its fitted lifetime."""

    dq = _dynamiqs()
    system = build_cat_system(x, drift=drift, na=na, nb=nb, kappa_a=kappa_a, kappa_b=kappa_b)
    label = initial_state if isinstance(initial_state, str) else ""
    psi0 = system["states"][label] if label else initial_state
    observable = system["sx"] if "x" in label else system["sz"]
    times = jnp.linspace(0.0, float(tfinal), int(n_points))

    result = dq.mesolve(
        system["H"],
        system["loss_ops"],
        psi0,
        times,
        options=dq.Options(progress_meter=False),
        exp_ops=[observable],
    )

    values = np.asarray(result.expects[0].real, dtype=float)
    times_np = np.asarray(times, dtype=float)
    fit = robust_exp_fit(
        times_np,
        values,
        max_tau=max(100.0, 80.0 * float(tfinal)),
    )
    return {
        "T": float(fit["params"][1]),
        "times": times_np,
        "values": values,
        "fit": fit,
    }


def _contrast_to_lifetime_proxy(contrast: float, tfinal: float) -> float:
    """Convert one endpoint contrast into an exponential lifetime proxy."""

    contrast = float(abs(contrast))
    contrast = float(np.clip(contrast, 1.0e-9, 1.0 - 1.0e-9))
    return float(-float(tfinal) / math.log(contrast))


def _simulate_expectation(
    system: dict[str, Any],
    state_label: str,
    observable: Any,
    tfinal: float,
    n_points: int,
) -> np.ndarray:
    dq = _dynamiqs()
    times = jnp.linspace(0.0, float(tfinal), int(n_points))
    result = dq.mesolve(
        system["H"],
        system["loss_ops"],
        system["states"][state_label],
        times,
        options=dq.Options(progress_meter=False),
        exp_ops=[observable],
    )
    return np.asarray(result.expects[0].real, dtype=float)


def compute_proxy_lifetime_metrics(
    x: Any,
    config: dict[str, Any] | None = None,
    drift: dict[str, Any] | None = None,
    na: int = 15,
    nb: int = 5,
    kappa_a: float = 1.0,
    kappa_b: float = 10.0,
) -> dict[str, float]:
    """Estimate logical lifetimes from short endpoint flip syndromes, no fit.

    ``single_endpoint`` uses only ``+x`` and ``+z``. ``two_sided_flip_syndrome``
    uses ``+/-`` logical pairs and measures the final contrast between them,
    which is usually a steadier optimization signal when offsets drift.
    """

    cfg = merged_config(config)
    system = build_cat_system(x, drift=drift, na=na, nb=nb, kappa_a=kappa_a, kappa_b=kappa_b)
    n_points = int(cfg.get("proxy_n_points", 2))
    tfinal_x = float(cfg.get("proxy_tfinal_x", cfg["tfinal_x"]))
    tfinal_z = float(cfg.get("proxy_tfinal_z", cfg["tfinal_z"]))
    mode = str(cfg.get("proxy_mode", "two_sided_flip_syndrome"))

    plus_x = _simulate_expectation(system, "+x", system["sx"], tfinal_x, n_points)
    plus_z = _simulate_expectation(system, "+z", system["sz"], tfinal_z, n_points)
    x_plus_final = float(plus_x[-1])
    z_plus_final = float(plus_z[-1])

    metrics: dict[str, float] = {
        "x_plus_final": x_plus_final,
        "z_plus_final": z_plus_final,
    }

    if mode == "single_endpoint":
        x_contrast = abs(x_plus_final)
        z_contrast = abs(z_plus_final)
    elif mode == "two_sided_flip_syndrome":
        minus_x = _simulate_expectation(system, "-x", system["sx"], tfinal_x, n_points)
        minus_z = _simulate_expectation(system, "-z", system["sz"], tfinal_z, n_points)
        x_minus_final = float(minus_x[-1])
        z_minus_final = float(minus_z[-1])
        x_contrast = 0.5 * abs(x_plus_final - x_minus_final)
        z_contrast = 0.5 * abs(z_plus_final - z_minus_final)
        metrics.update(
            {
                "x_minus_final": x_minus_final,
                "z_minus_final": z_minus_final,
            }
        )
    else:
        raise ValueError("proxy_mode must be 'single_endpoint' or 'two_sided_flip_syndrome'")

    Tx = _contrast_to_lifetime_proxy(x_contrast, tfinal_x)
    Tz = _contrast_to_lifetime_proxy(z_contrast, tfinal_z)
    bias = float(Tz / Tx) if Tx > EPS else np.nan
    metrics.update(
        {
            "Tx": Tx,
            "Tz": Tz,
            "bias": bias,
            "eta": bias,
            "x_contrast": float(x_contrast),
            "z_contrast": float(z_contrast),
            "x_flip_syndrome": float(0.5 * (1.0 - np.clip(x_contrast, -1.0, 1.0))),
            "z_flip_syndrome": float(0.5 * (1.0 - np.clip(z_contrast, -1.0, 1.0))),
            "proxy_tfinal_x": tfinal_x,
            "proxy_tfinal_z": tfinal_z,
        }
    )
    metrics.update(_diagnostics_from_system(system))
    return metrics


def _to_float(value: Any) -> float:
    if hasattr(value, "to_jax"):
        value = value.to_jax()
    elif hasattr(value, "data"):
        value = value.data
    arr = np.asarray(value)
    return float(np.real(np.squeeze(arr)))


def _expectation(state: Any, operator: Any) -> float:
    return _to_float(state.dag() @ operator @ state)


def _diagnostics_from_system(system: dict[str, Any]) -> dict[str, float]:
    plus_z = system["states"]["+z"]
    minus_z = system["states"]["-z"]
    parity_plus_z = _expectation(plus_z, system["parity"])
    parity_minus_z = _expectation(minus_z, system["parity"])
    return {
        "alpha": _to_float(system["alpha"]),
        "nbar": _expectation(plus_z, system["n_total"]),
        "parity_plus_z": parity_plus_z,
        "parity_minus_z": parity_minus_z,
        "parity_contrast": float(abs(parity_plus_z - parity_minus_z)),
    }


def compute_cat_diagnostics(
    x: Any,
    drift: dict[str, Any] | None = None,
    na: int = 15,
    nb: int = 5,
    kappa_a: float = 1.0,
    kappa_b: float = 10.0,
) -> dict[str, float]:
    """Compute cheap diagnostic quantities from the initial logical states."""

    system = build_cat_system(x, drift=drift, na=na, nb=nb, kappa_a=kappa_a, kappa_b=kappa_b)
    return _diagnostics_from_system(system)


# EDIT THIS FUNCTION TO TRY NEW REWARD DESIGNS
def cat_reward_from_metrics(metrics: dict[str, Any], config: dict[str, Any] | None = None) -> float:
    """Combine precomputed metrics into one scalar reward."""

    cfg = merged_config(config)
    Tx = max(EPS, float(metrics.get("Tx", metrics.get("T_X", EPS))))
    Tz = max(EPS, float(metrics.get("Tz", metrics.get("T_Z", EPS))))
    bias = max(EPS, float(metrics.get("bias", Tz / Tx)))
    eta_target = max(EPS, float(cfg["eta_target"]))

    reward = (
        math.log(Tx)
        + math.log(Tz)
        - float(cfg["lambda_bias"]) * math.log(bias / eta_target) ** 2
    )

    alpha = float(metrics.get("alpha", 0.0))
    nbar = float(metrics.get("nbar", 0.0))
    parity_contrast = float(metrics.get("parity_contrast", 0.0))

    if cfg["use_alpha_penalty"]:
        reward -= float(cfg["lambda_alpha"]) * (alpha - float(cfg["alpha_target"])) ** 2

    if cfg["use_nbar_penalty"]:
        reward -= float(cfg["lambda_nbar"]) * max(0.0, nbar - float(cfg["nbar_max"])) ** 2

    if cfg["use_parity_bonus"]:
        reward += float(cfg["lambda_parity"]) * parity_contrast

    return float(reward)


def cat_lifetime_loss(
    x: Any,
    config: dict[str, Any] | None = None,
    return_metrics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Return negative flexible reward for optimizer minimization."""

    cfg = merged_config(config)
    try:
        sim_kwargs = {
            "drift": cfg.get("drift"),
            "na": int(cfg["na"]),
            "nb": int(cfg["nb"]),
            "kappa_a": float(cfg["kappa_a"]),
            "kappa_b": float(cfg["kappa_b"]),
            "n_points": int(cfg.get("n_points", 60)),
        }
        z_result = measure_lifetime_cat("+z", float(cfg["tfinal_z"]), x, **sim_kwargs)
        x_result = measure_lifetime_cat("+x", float(cfg["tfinal_x"]), x, **sim_kwargs)
        Tx = float(x_result["T"])
        Tz = float(z_result["T"])
        bias = float(Tz / Tx) if Tx > EPS else np.nan

        diagnostics = compute_cat_diagnostics(
            x,
            drift=cfg.get("drift"),
            na=int(cfg["na"]),
            nb=int(cfg["nb"]),
            kappa_a=float(cfg["kappa_a"]),
            kappa_b=float(cfg["kappa_b"]),
        )
        metrics: dict[str, Any] = {
            "Tx": Tx,
            "Tz": Tz,
            "bias": bias,
            **diagnostics,
        }
        reward = cat_reward_from_metrics(metrics, cfg)
        loss = -float(reward)
        metrics["reward"] = float(reward)
        metrics["loss"] = loss
        if return_metrics:
            return loss, metrics
        return loss
    except Exception as exc:
        print(f"cat_lifetime_loss failed: {exc}")
        if return_metrics:
            return 1.0e6, {"error": repr(exc)}
        return 1.0e6


def cat_proxy_loss(
    x: Any,
    config: dict[str, Any] | None = None,
    return_metrics: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    """Fast no-fit loss using endpoint flip-syndrome lifetime proxies."""

    cfg = merged_config(config)
    try:
        metrics = compute_proxy_lifetime_metrics(
            x,
            config=cfg,
            drift=cfg.get("drift"),
            na=int(cfg["na"]),
            nb=int(cfg["nb"]),
            kappa_a=float(cfg["kappa_a"]),
            kappa_b=float(cfg["kappa_b"]),
        )
        reward = cat_reward_from_metrics(metrics, cfg)
        reward += float(cfg.get("lambda_syndrome_floor", 0.0)) * (
            math.log(max(EPS, metrics["x_contrast"]))
            + math.log(max(EPS, metrics["z_contrast"]))
        )
        loss = -float(reward)
        metrics["reward"] = float(reward)
        metrics["loss"] = loss
        metrics["reward_type"] = "proxy_no_fit"
        if return_metrics:
            return loss, metrics
        return loss
    except Exception as exc:
        print(f"cat_proxy_loss failed: {exc}")
        if return_metrics:
            return 1.0e6, {"error": repr(exc)}
        return 1.0e6


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return value


def save_run_config(config: dict[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "config.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(config), f, indent=2, sort_keys=True)
    return path


HISTORY_COLUMNS = [
    "generation",
    "candidate_index",
    "loss",
    "reward",
    "Tx",
    "Tz",
    "bias",
    "alpha",
    "nbar",
    "parity_plus_z",
    "parity_minus_z",
    "parity_contrast",
    "x0",
    "x1",
    "x2",
    "x3",
]


def flatten_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history:
        metrics = item.get("metrics", {})
        x = np.asarray(item.get("x", [np.nan, np.nan, np.nan, np.nan]), dtype=float)
        row = {
            "generation": item.get("generation"),
            "candidate_index": item.get("candidate_index"),
            "loss": item.get("loss", metrics.get("loss")),
            "reward": metrics.get("reward"),
            "Tx": metrics.get("Tx"),
            "Tz": metrics.get("Tz"),
            "bias": metrics.get("bias"),
            "alpha": metrics.get("alpha"),
            "nbar": metrics.get("nbar"),
            "parity_plus_z": metrics.get("parity_plus_z"),
            "parity_minus_z": metrics.get("parity_minus_z"),
            "parity_contrast": metrics.get("parity_contrast"),
            "x0": x[0],
            "x1": x[1],
            "x2": x[2],
            "x3": x[3],
        }
        rows.append(row)
    return rows


def save_metrics_history(history: list[dict[str, Any]], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "candidate_history.csv"
    rows = flatten_history(history)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_safe(row.get(key, "")) for key in HISTORY_COLUMNS})
    return path


def save_summary(summary: dict[str, Any], run_dir: str | Path) -> Path:
    path = Path(run_dir) / "summary.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, sort_keys=True)
    return path


def plot_metrics_history(history: list[dict[str, Any]], run_dir: str | Path) -> list[Path]:
    """Save standard diagnostic plots into ``run_dir``."""

    import matplotlib.pyplot as plt

    rows = flatten_history(history)
    run_path = Path(run_dir)
    eval_idx = np.arange(len(rows), dtype=int)
    paths: list[Path] = []

    def series(name: str) -> np.ndarray:
        return np.asarray([float(row.get(name, np.nan)) for row in rows], dtype=float)

    def save_line(filename: str, y_names: list[str], ylabel: str, logy: bool = False) -> None:
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in y_names:
            ax.plot(eval_idx, series(name), marker="o", lw=1.5, label=name)
        ax.set_xlabel("Evaluation")
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        if len(y_names) > 1:
            ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = run_path / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    save_line("loss_vs_evaluation.png", ["loss"], "Loss")
    save_line("reward_vs_evaluation.png", ["reward"], "Reward")
    save_line("Tx_Tz_vs_evaluation.png", ["Tx", "Tz"], "Lifetime", logy=True)
    save_line("bias_vs_evaluation.png", ["bias"], "Bias", logy=True)
    save_line("alpha_nbar_vs_evaluation.png", ["alpha", "nbar"], "Value")
    save_line(
        "parity_vs_evaluation.png",
        ["parity_plus_z", "parity_minus_z", "parity_contrast"],
        "Parity",
    )
    return paths
