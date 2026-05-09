"""Adapter boundary between reusable surrogate code and notebook simulations."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .params import derived_physics_features, unpack_params


class SimulationAdapter:
    """Interface to connect the package to notebook or Dynamiqs physics code."""

    def compute_static_observables(self, theta: np.ndarray, config: Any) -> dict[str, float]:
        """
        TODO: connect to existing Dynamiqs/notebook simulation.

        Should return static cheap observables such as:
            n_mean, n2_mean, n_var, parity, abs_a, re_a, im_a,
            abs_a2, re_a2, im_a2, cat_residual, purity, logical_leakage.
        """

        raise NotImplementedError

    def compute_short_time_observables(self, theta: np.ndarray, config: Any) -> dict[str, float]:
        """
        TODO: connect to existing short-time evolution.

        Should return:
            S_X_t0, S_X_t1, ...
            S_Z_t0, S_Z_t1, ...
            gamma_X_short, gamma_Z_short
            T_X_short, T_Z_short
        """

        raise NotImplementedError

    def expensive_lifetime_benchmark(self, theta: np.ndarray, config: Any) -> dict[str, float]:
        """
        TODO: connect to full T_X, T_Z lifetime simulation and exponential fitting.

        Should return:
            T_X, T_Z, eta, log_T_X, log_T_Z, log_eta.
        """

        raise NotImplementedError


class DummyAdapter(SimulationAdapter):
    """Deterministic non-physical adapter for tests, demos, and plumbing checks."""

    def compute_static_observables(self, theta: np.ndarray, config: Any) -> dict[str, float]:
        g2, eps_d = unpack_params(theta)
        derived = derived_physics_features(g2, eps_d, config.kappa_b, config.na)
        alpha = derived["alpha_est"]
        n_mean = alpha**2
        n_var = max(alpha, 0.0) + 0.05 * abs(eps_d)
        a = eps_d / (abs(g2) + 1e-6)
        a2 = a * a
        return {
            "n_mean": float(n_mean),
            "n2_mean": float(n_mean**2 + n_var),
            "n_var": float(n_var),
            "parity": float(math.cos(math.pi * n_mean)),
            "abs_a": float(abs(a)),
            "re_a": float(a.real),
            "im_a": float(a.imag),
            "abs_a2": float(abs(a2)),
            "re_a2": float(a2.real),
            "im_a2": float(a2.imag),
            "cat_residual": float(1.0 / (1.0 + alpha**2)),
            "purity": float(1.0 - 0.02 * min(alpha, 10.0)),
            "logical_leakage": float(max(0.0, -derived.get("truncation_margin", 1.0)) / max(config.na, 1)),
        }

    def compute_short_time_observables(self, theta: np.ndarray, config: Any) -> dict[str, float]:
        result: dict[str, float] = {}
        bench = self.expensive_lifetime_benchmark(theta, config)
        t_x = max(bench["T_X"], 1e-9)
        t_z = max(bench["T_Z"], 1e-9)
        for i, t in enumerate(config.probe_times_X):
            result[f"S_X_t{i}"] = float(math.exp(-float(t) / t_x))
        for i, t in enumerate(config.probe_times_Z):
            result[f"S_Z_t{i}"] = float(math.exp(-float(t) / t_z))
        result["gamma_X_short"] = float(1.0 / t_x)
        result["gamma_Z_short"] = float(1.0 / t_z)
        result["T_X_short"] = float(t_x)
        result["T_Z_short"] = float(t_z)
        return result

    def expensive_lifetime_benchmark(self, theta: np.ndarray, config: Any) -> dict[str, float]:
        g2, eps_d = unpack_params(theta)
        derived = derived_physics_features(g2, eps_d, config.kappa_b, config.na)
        alpha = derived["alpha_est"]
        g_ratio = abs(g2) / max(abs(config.kappa_b), 1e-12)
        drive = abs(eps_d)
        phase_match = math.cos(np.angle(g2 + 1e-12) - np.angle(eps_d + 1e-12))

        log_t_x = 1.0 + 0.55 * alpha - 1.5 * g_ratio + 0.08 * phase_match
        log_t_z = log_t_x + math.log(max(config.eta_target, 1e-9)) - 0.15 * (alpha - 2.0) ** 2 + 0.1 * drive
        t_x = float(math.exp(log_t_x))
        t_z = float(math.exp(log_t_z))
        eta = t_z / max(t_x, 1e-12)
        return {
            "T_X": t_x,
            "T_Z": t_z,
            "eta": float(eta),
            "log_T_X": float(math.log(t_x)),
            "log_T_Z": float(math.log(t_z)),
            "log_eta": float(math.log(max(eta, 1e-300))),
        }
