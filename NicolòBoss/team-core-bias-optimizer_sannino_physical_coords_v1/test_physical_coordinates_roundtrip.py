"""Lightweight physical-coordinate round-trip smoke check."""

from __future__ import annotations

import numpy as np

from physical_coordinates import (
    physical_diagnostics_from_raw,
    physical_to_raw,
    raw_to_physical,
)


def main() -> None:
    u0 = np.array([1.0, 0.0, 4.0, 0.0], dtype=float)
    v0 = raw_to_physical(u0, kappa_b=10.0)
    u0_round = physical_to_raw(v0, kappa_b=10.0)
    d0 = physical_diagnostics_from_raw(u0, kappa_b=10.0)
    d_round = physical_diagnostics_from_raw(u0_round, kappa_b=10.0)

    print("u0 =", u0)
    print("v0 =", v0)
    print("u0_round =", u0_round)
    print("diagnostics u0 =", d0)
    print("diagnostics u0_round =", d_round)

    assert np.isclose(d0["kappa_2"], d_round["kappa_2"], rtol=1e-8, atol=1e-10)
    assert np.isclose(d0["alpha_abs"], d_round["alpha_abs"], rtol=1e-8, atol=1e-10)
    print("round-trip physical diagnostics passed")


if __name__ == "__main__":
    main()

