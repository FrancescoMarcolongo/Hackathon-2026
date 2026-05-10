"""Simple lifetime-based reward baseline for cat-qubit stabilization."""

import dynamiqs as dq
import jax.numpy as jnp
from scipy.optimize import least_squares


def _exp_model(p, t):
    A, tau, C = p
    return A * jnp.exp(-t / tau) + C


def _exp_residuals(p, x, y):
    return _exp_model(p, x) - y


def robust_exp_fit(x, y):
    """Robustly fit y = A * exp(-t / tau) + C."""
    A0 = y.max() - y.min()
    C0 = y.min()
    tau0 = x.max() - x.min()
    p0 = [A0, tau0, C0]

    res = least_squares(
        _exp_residuals,
        p0,
        args=(x, y),
        bounds=([0, 0, -jnp.inf], [jnp.inf, jnp.inf, jnp.inf]),
        loss="soft_l1",
        f_scale=0.1,
    )

    y_fit = _exp_model(res.x, x)

    return {
        "popt": res.x,
        "y_fit": y_fit,
    }


def unpack_knobs(x):
    """Convert real optimization knobs into complex control parameters."""
    g2 = x[0] + 1j * x[1]
    eps_d = x[2] + 1j * x[3]
    return g2, eps_d


def measure_lifetime_cat(
    initial_state,
    tfinal,
    x,
    drift=None,
    na=15,
    nb=5,
    kappa_b=10.0,
    kappa_a=1.0,
    nsave=100,
):
    """Simulate the cat-qubit lifetime experiment for one logical input state."""
    g2, eps_d = unpack_knobs(x)

    if drift is not None:
        g2 = g2 * drift.get("g2_prefactor", 1.0)
        eps_d = eps_d * drift.get("epsd_prefactor", 1.0)

    a_storage = dq.destroy(na)
    a = dq.tensor(a_storage, dq.eye(nb))
    b = dq.tensor(dq.eye(na), dq.destroy(nb))

    eps_2 = 2 * g2 * eps_d / kappa_b
    kappa_2 = 4 * jnp.abs(g2) ** 2 / kappa_b

    alpha_estimate = jnp.sqrt(
        jnp.maximum(
            1e-9,
            2 / kappa_2 * (jnp.abs(eps_2) - kappa_a / 4),
        )
    )

    plus_z = dq.coherent(na, alpha_estimate)
    minus_z = dq.coherent(na, -alpha_estimate)
    plus_x = (plus_z + minus_z) / jnp.sqrt(2)
    minus_x = (plus_z - minus_z) / jnp.sqrt(2)

    buffer_vacuum = dq.fock(nb, 0)
    basis = {
        "+z": dq.tensor(plus_z, buffer_vacuum),
        "-z": dq.tensor(minus_z, buffer_vacuum),
        "+x": dq.tensor(plus_x, buffer_vacuum),
        "-x": dq.tensor(minus_x, buffer_vacuum),
    }

    sx_storage = (1j * jnp.pi * a_storage.dag() @ a_storage).expm()
    sx = dq.tensor(sx_storage, dq.eye(nb))

    sz_storage = plus_z @ plus_z.dag() - minus_z @ minus_z.dag()
    sz = dq.tensor(sz_storage, dq.eye(nb))

    H = (
        jnp.conj(g2) * a @ a @ b.dag()
        + g2 * a.dag() @ a.dag() @ b
        - eps_d * b.dag()
        - jnp.conj(eps_d) * b
    )

    psi0 = basis[initial_state]
    tsave = jnp.linspace(0, tfinal, nsave)

    res = dq.mesolve(
        H,
        [jnp.sqrt(kappa_b) * b, jnp.sqrt(kappa_a) * a],
        psi0,
        tsave,
        options=dq.Options(progress_meter=False),
        exp_ops=[sx, sz],
    )

    return res


def cat_reward_from_lifetimes(Tx, Tz, eta_target=100.0, lambda_bias=2.0):
    eps = 1e-9
    Tx = jnp.maximum(Tx, eps)
    Tz = jnp.maximum(Tz, eps)
    bias = Tz / Tx
    lifetime_reward = jnp.log(Tx) + jnp.log(Tz)
    bias_penalty = lambda_bias * (
        jnp.log((bias + eps) / eta_target)
    ) ** 2
    reward = lifetime_reward - bias_penalty
    return reward


def cat_lifetime_metrics(
    x,
    eta_target=100.0,
    lambda_bias=2.0,
    tfinal_z=200.0,
    tfinal_x=2.0,
    na=15,
    nb=5,
    kappa_b=10.0,
    kappa_a=1.0,
    nsave=100,
    drift=None,
):
    res_z = measure_lifetime_cat(
        "+z",
        tfinal=tfinal_z,
        x=x,
        drift=drift,
        na=na,
        nb=nb,
        kappa_b=kappa_b,
        kappa_a=kappa_a,
        nsave=nsave,
    )
    z_signal = res_z.expects[1, :].real
    fit_z = robust_exp_fit(res_z.tsave, z_signal)
    Tz = fit_z["popt"][1]

    res_x = measure_lifetime_cat(
        "+x",
        tfinal=tfinal_x,
        x=x,
        drift=drift,
        na=na,
        nb=nb,
        kappa_b=kappa_b,
        kappa_a=kappa_a,
        nsave=nsave,
    )
    x_signal = res_x.expects[0, :].real
    fit_x = robust_exp_fit(res_x.tsave, x_signal)
    Tx = fit_x["popt"][1]

    eta = Tz / Tx
    reward = cat_reward_from_lifetimes(
        Tx,
        Tz,
        eta_target=eta_target,
        lambda_bias=lambda_bias,
    )
    loss = float(-reward)

    return {
        "loss": loss,
        "reward": float(reward),
        "Tx": float(Tx),
        "Tz": float(Tz),
        "eta": float(eta),
        "res_x": res_x,
        "res_z": res_z,
        "fit_x": fit_x,
        "fit_z": fit_z,
    }


def cat_lifetime_loss(
    x,
    eta_target=100.0,
    lambda_bias=2.0,
    tfinal_z=200.0,
    tfinal_x=2.0,
    na=15,
    nb=5,
    kappa_b=10.0,
    kappa_a=1.0,
    nsave=100,
    drift=None,
):
    try:
        return cat_lifetime_metrics(
            x,
            eta_target=eta_target,
            lambda_bias=lambda_bias,
            tfinal_z=tfinal_z,
            tfinal_x=tfinal_x,
            na=na,
            nb=nb,
            kappa_b=kappa_b,
            kappa_a=kappa_a,
            nsave=nsave,
            drift=drift,
        )["loss"]
    except Exception as exc:
        print("cat_lifetime_loss failed:", repr(exc))
        return 1e6
