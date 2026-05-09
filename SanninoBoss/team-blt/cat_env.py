"""Black-box cat-qubit lab simulator for Boundary Liouvillian Tracking.

Only :meth:`CatLab.run_experiment` is meant to be visible to optimizers.  The
Hamiltonian, jump operators, density matrices, and Liouvillian live behind that
lab primitive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import expm_multiply


PREP_STATES = ("+x", "-x", "+y", "-y", "+z", "-z")
MEAS_AXES = ("x", "y", "z")


@dataclass(frozen=True)
class CatSystemConfig:
    """Physical constants for the storage+buffer model.

    Units follow the challenge notebook: rates are in MHz-like inverse-us units
    and evolution times are in us-like units.
    """

    na: int = 10
    nb: int = 3
    kappa_b: float = 8.0
    kappa_a: float = 0.15
    z_observable: str = "ideal"  # "ideal" or "quadrature_proxy"
    z_proxy_smoothing: float = 0.35
    measurement_sigma: float = 0.0
    cache_round_decimals: int = 12
    seed: int = 1234


@dataclass(frozen=True)
class PhysicalSummary:
    g2: complex
    eps_d: complex
    kappa_2: float
    eps_2: complex
    alpha: complex
    nbar: float


@dataclass
class _Model:
    liouvillian: sparse.csc_matrix
    alpha: complex
    observables: Dict[str, np.ndarray]


def destroy(n: int) -> np.ndarray:
    """Annihilation operator in a truncated Fock basis."""

    op = np.zeros((n, n), dtype=np.complex128)
    for k in range(1, n):
        op[k - 1, k] = np.sqrt(k)
    return op


def coherent_state(n: int, alpha: complex) -> np.ndarray:
    """Truncated coherent state, normalized after truncation."""

    vec = np.empty(n, dtype=np.complex128)
    vec[0] = np.exp(-0.5 * abs(alpha) ** 2)
    for k in range(1, n):
        vec[k] = vec[k - 1] * alpha / np.sqrt(k)
    norm = np.linalg.norm(vec)
    if norm == 0 or not np.isfinite(norm):
        out = np.zeros(n, dtype=np.complex128)
        out[0] = 1.0
        return out
    return vec / norm


def fock_state(n: int, level: int) -> np.ndarray:
    vec = np.zeros(n, dtype=np.complex128)
    vec[level] = 1.0
    return vec


def normalized_superposition(a: np.ndarray, b: np.ndarray, coeff: complex) -> np.ndarray:
    """Return normalized ``a + coeff*b``.

    This avoids the notebook's convenient but approximate division by sqrt(2),
    which is inaccurate when the coherent states have visible overlap.
    """

    vec = a + coeff * b
    norm = np.linalg.norm(vec)
    if norm < 1e-14 or not np.isfinite(norm):
        raise ValueError("logical superposition has near-zero norm")
    return vec / norm


def logical_storage_state(na: int, alpha: complex, prep_state: str) -> np.ndarray:
    """Storage-mode logical state using the notebook's coherent-state labels."""

    if prep_state not in PREP_STATES:
        raise ValueError(f"unknown prep_state {prep_state!r}")

    plus_z = coherent_state(na, alpha)
    minus_z = coherent_state(na, -alpha)
    if prep_state == "+z":
        return plus_z
    if prep_state == "-z":
        return minus_z
    if prep_state == "+x":
        return normalized_superposition(plus_z, minus_z, 1.0)
    if prep_state == "-x":
        return normalized_superposition(plus_z, minus_z, -1.0)
    if prep_state == "+y":
        return normalized_superposition(plus_z, minus_z, 1.0j)
    if prep_state == "-y":
        return normalized_superposition(plus_z, minus_z, -1.0j)
    raise AssertionError("unreachable")


def knobs_to_complex(knobs: np.ndarray) -> Tuple[complex, complex]:
    knobs = np.asarray(knobs, dtype=float)
    if knobs.shape != (4,):
        raise ValueError("knobs must be [Re(g2), Im(g2), Re(eps_d), Im(eps_d)]")
    g2 = knobs[0] + 1.0j * knobs[1]
    eps_d = knobs[2] + 1.0j * knobs[3]
    return g2, eps_d


def physical_summary(
    knobs: np.ndarray, *, kappa_b: float, kappa_a: float
) -> PhysicalSummary:
    """Map lab knobs to the effective notebook parameters."""

    g2, eps_d = knobs_to_complex(knobs)
    kappa_2 = float(4.0 * abs(g2) ** 2 / kappa_b)
    if kappa_2 <= 0:
        alpha = 0.0j
        eps_2 = 0.0j
    else:
        eps_2 = 2.0 * g2 * eps_d / kappa_b
        alpha_sq = 2.0 * (eps_2 - kappa_a / 4.0) / kappa_2
        alpha = np.sqrt(alpha_sq + 0.0j)
    return PhysicalSummary(
        g2=g2,
        eps_d=eps_d,
        kappa_2=kappa_2,
        eps_2=eps_2,
        alpha=alpha,
        nbar=float(abs(alpha) ** 2),
    )


def q_to_knobs(q: np.ndarray, *, kappa_b: float, kappa_a: float) -> np.ndarray:
    """Internal BLT physical coordinates -> four real lab knobs.

    q = (log_kappa2, log_nbar, theta_alpha, phi_g).  The returned knobs are
    still exactly the lab controls: complex g2 and complex eps_d.
    """

    log_kappa2, log_nbar, theta_alpha, phi_g = np.asarray(q, dtype=float)
    kappa_2 = float(np.exp(log_kappa2))
    nbar = float(np.exp(log_nbar))
    alpha = np.sqrt(nbar) * np.exp(1.0j * theta_alpha)
    g2 = 0.5 * np.sqrt(kappa_2 * kappa_b) * np.exp(1.0j * phi_g)
    eps_2 = 0.5 * kappa_2 * alpha**2 + kappa_a / 4.0
    eps_d = eps_2 * kappa_b / (2.0 * g2)
    return np.array([g2.real, g2.imag, eps_d.real, eps_d.imag], dtype=float)


def default_q_bounds() -> np.ndarray:
    """Conservative quick-mode trust region in physical coordinates."""

    return np.array(
        [
            [np.log(0.12), np.log(2.5)],  # kappa_2
            [np.log(0.65), np.log(4.0)],  # nbar
            [-0.75, 0.75],  # alpha phase
            [-0.9, 0.9],  # g2 phase
        ],
        dtype=float,
    )


def _projector(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def storage_observables(
    na: int, alpha: complex, *, z_observable: str, z_proxy_smoothing: float
) -> Dict[str, np.ndarray]:
    """Logical observables on the storage mode.

    X follows the challenge notebook: photon-number parity.  Z defaults to the
    notebook's ideal coherent-projector observable.  Y is a Hermitian Pauli on
    the Gram-Schmidt logical subspace defined by |alpha> and |-alpha>.
    """

    plus_z = coherent_state(na, alpha)
    minus_z = coherent_state(na, -alpha)
    parity = np.diag([(-1.0) ** n for n in range(na)]).astype(np.complex128)

    if z_observable == "ideal":
        z_op = _projector(plus_z) - _projector(minus_z)
    elif z_observable == "quadrature_proxy":
        a = destroy(na)
        theta = np.angle(alpha) if abs(alpha) > 1e-12 else 0.0
        quad = (np.exp(-1.0j * theta) * a + np.exp(1.0j * theta) * a.conj().T) / np.sqrt(2.0)
        evals, evecs = np.linalg.eigh(quad)
        smooth_sign = np.tanh(evals / max(z_proxy_smoothing, 1e-6))
        z_op = (evecs * smooth_sign) @ evecs.conj().T
    else:
        raise ValueError("z_observable must be 'ideal' or 'quadrature_proxy'")

    overlap = np.vdot(plus_z, minus_z)
    y1 = minus_z - overlap * plus_z
    y1_norm = np.linalg.norm(y1)
    if y1_norm < 1e-10:
        y1 = normalized_superposition(plus_z, minus_z, -1.0)
    else:
        y1 = y1 / y1_norm
    y_op = -1.0j * np.outer(plus_z, y1.conj()) + 1.0j * np.outer(y1, plus_z.conj())
    y_op = 0.5 * (y_op + y_op.conj().T)

    return {
        "x": parity,
        "y": y_op,
        "z": 0.5 * (z_op + z_op.conj().T),
    }


class CatLab:
    """Lab-like black-box wrapper around the storage+buffer Lindblad model."""

    def __init__(self, config: CatSystemConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._model_cache: Dict[Tuple[float, ...], _Model] = {}
        self._settings = 0
        self._wait_time_cost = 0.0

        na, nb = config.na, config.nb
        self._a_storage = destroy(na)
        self._b_buffer = destroy(nb)
        self._eye_a = np.eye(na, dtype=np.complex128)
        self._eye_b = np.eye(nb, dtype=np.complex128)
        self._a = np.kron(self._a_storage, self._eye_b)
        self._b = np.kron(self._eye_a, self._b_buffer)
        self._vac_b = fock_state(nb, 0)
        self._dim = na * nb

    def reset_counters(self) -> None:
        self._settings = 0
        self._wait_time_cost = 0.0

    def counters(self) -> Dict[str, float]:
        return {
            "settings": int(self._settings),
            "wait_time_cost": float(self._wait_time_cost),
        }

    def clear_cache(self) -> None:
        self._model_cache.clear()

    def _cache_key(self, knobs: np.ndarray) -> Tuple[float, ...]:
        return tuple(np.round(np.asarray(knobs, dtype=float), self.config.cache_round_decimals))

    def _hamiltonian_jumps_summary(
        self, knobs: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], PhysicalSummary]:
        summary = physical_summary(
            knobs, kappa_b=self.config.kappa_b, kappa_a=self.config.kappa_a
        )
        a = self._a
        b = self._b
        adag = a.conj().T
        bdag = b.conj().T

        # Challenge notebook convention:
        # H = conj(g2) a^2 b^dag + g2 (a^dag)^2 b - eps_d b^dag - conj(eps_d) b.
        hamiltonian = (
            np.conj(summary.g2) * (a @ a @ bdag)
            + summary.g2 * (adag @ adag @ b)
            - summary.eps_d * bdag
            - np.conj(summary.eps_d) * b
        )
        jumps = [np.sqrt(self.config.kappa_b) * b]
        if self.config.kappa_a > 0:
            jumps.append(np.sqrt(self.config.kappa_a) * a)
        return hamiltonian, jumps, summary

    def _build_model(self, knobs: np.ndarray) -> _Model:
        hamiltonian, jumps, summary = self._hamiltonian_jumps_summary(knobs)
        dim = self._dim
        ident = sparse.identity(dim, dtype=np.complex128, format="csc")
        h_sp = sparse.csc_matrix(hamiltonian)
        liouvillian = -1.0j * (sparse.kron(ident, h_sp) - sparse.kron(h_sp.T, ident))
        for jump in jumps:
            j_sp = sparse.csc_matrix(jump)
            jdj = j_sp.conj().T @ j_sp
            liouvillian = liouvillian + sparse.kron(j_sp.conj(), j_sp)
            liouvillian = liouvillian - 0.5 * sparse.kron(ident, jdj)
            liouvillian = liouvillian - 0.5 * sparse.kron(jdj.T, ident)
        liouvillian = liouvillian.tocsc()

        storage_ops = storage_observables(
            self.config.na,
            summary.alpha,
            z_observable=self.config.z_observable,
            z_proxy_smoothing=self.config.z_proxy_smoothing,
        )
        observables = {axis: np.kron(op, self._eye_b) for axis, op in storage_ops.items()}
        return _Model(liouvillian=liouvillian, alpha=summary.alpha, observables=observables)

    def _model(self, knobs: np.ndarray) -> _Model:
        key = self._cache_key(knobs)
        model = self._model_cache.get(key)
        if model is None:
            model = self._build_model(knobs)
            self._model_cache[key] = model
        return model

    def run_experiment(
        self,
        knobs: np.ndarray,
        prep_state: str,
        meas_axis: str,
        wait_time: float,
        n_shots: int | None = None,
    ) -> float:
        """Return one logical expectation value in [-1, 1].

        This is the sole lab primitive used by estimators and optimizers.
        """

        if prep_state not in PREP_STATES:
            raise ValueError(f"unknown prep_state {prep_state!r}")
        if meas_axis not in MEAS_AXES:
            raise ValueError(f"unknown meas_axis {meas_axis!r}")
        wait_time = float(wait_time)
        if wait_time < 0:
            raise ValueError("wait_time must be non-negative")

        self._settings += 1
        self._wait_time_cost += wait_time

        model = self._model(knobs)
        psi_a = logical_storage_state(self.config.na, model.alpha, prep_state)
        psi = np.kron(psi_a, self._vac_b)
        rho0 = np.outer(psi, psi.conj())
        vec0 = rho0.reshape(-1, order="F")

        if wait_time == 0:
            vec_t = vec0
        else:
            vec_t = expm_multiply(model.liouvillian * wait_time, vec0)
        rho_t = vec_t.reshape((self._dim, self._dim), order="F")
        obs = model.observables[meas_axis]
        value = float(np.real(np.trace(obs @ rho_t)))
        value = float(np.clip(value, -1.0, 1.0))

        if n_shots is not None:
            if n_shots <= 0:
                raise ValueError("n_shots must be positive when provided")
            p_plus = np.clip(0.5 * (1.0 + value), 0.0, 1.0)
            clicks = self.rng.binomial(int(n_shots), p_plus)
            value = 2.0 * clicks / int(n_shots) - 1.0
        if self.config.measurement_sigma > 0:
            value = float(value + self.rng.normal(0.0, self.config.measurement_sigma))
            value = float(np.clip(value, -1.0, 1.0))
        return value

    # The following oracle methods are for validation scripts only.  Optimizers
    # and estimators in this submission never call them.
    def oracle_hamiltonian(self, knobs: np.ndarray) -> np.ndarray:
        hamiltonian, _, _ = self._hamiltonian_jumps_summary(knobs)
        return hamiltonian

    def oracle_liouvillian_matrix(self, knobs: np.ndarray) -> sparse.csc_matrix:
        return self._model(knobs).liouvillian

    def oracle_apply_liouvillian(self, knobs: np.ndarray, rho: np.ndarray) -> np.ndarray:
        vec = np.asarray(rho, dtype=np.complex128).reshape(-1, order="F")
        out = self._model(knobs).liouvillian @ vec
        return out.reshape((self._dim, self._dim), order="F")

    def oracle_evolve_density(self, knobs: np.ndarray, rho: np.ndarray, wait_time: float) -> np.ndarray:
        vec = np.asarray(rho, dtype=np.complex128).reshape(-1, order="F")
        if wait_time == 0:
            out = vec
        else:
            out = expm_multiply(self._model(knobs).liouvillian * float(wait_time), vec)
        return out.reshape((self._dim, self._dim), order="F")
