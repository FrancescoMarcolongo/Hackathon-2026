import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import eval_genlaguerre, factorial


# ============================================================
# Parameters
# ============================================================
N = 35                 # Fock truncation
alpha = 2.0            # cat amplitude
theta = 0.20           # small but visible vortex deformation
xmax = 5.0             # phase-space window
n_grid = 181           # grid resolution

output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)
output_file = output_dir / "vortex_even_wigner_2d.png"


# ============================================================
# Coherent-state coefficients in Fock basis
# ============================================================
def coherent_coeffs(beta, N):
    n = np.arange(N)
    coeffs = np.exp(-0.5 * abs(beta)**2) * beta**n / np.sqrt(factorial(n))
    return coeffs.astype(complex)


# ============================================================
# Expectation value <a> for a pure state in Fock basis
# If psi = sum_n psi_n |n>, then
# <a> = sum_{n>=1} sqrt(n) conj(psi_{n-1}) psi_n
# ============================================================
def exp_a(psi):
    n = np.arange(1, len(psi))
    return np.sum(np.sqrt(n) * np.conjugate(psi[:-1]) * psi[1:])


# ============================================================
# Standard coherent branches |+alpha> and |-alpha>
# ============================================================
psi_plus = coherent_coeffs(alpha, N)
psi_minus = coherent_coeffs(-alpha, N)

psi_plus = psi_plus / np.linalg.norm(psi_plus)
psi_minus = psi_minus / np.linalg.norm(psi_minus)


# ============================================================
# Apply the vortex/Kerr-like twist
# U_theta = exp(i theta/2 * n(n-1))
# ============================================================
n = np.arange(N)
vortex_phase = np.exp(1j * theta * n * (n - 1) / 2.0)

# Logical vortex zero and one
psi_vortex_0 = vortex_phase * psi_plus
psi_vortex_1 = vortex_phase * psi_minus

psi_vortex_0 = psi_vortex_0 / np.linalg.norm(psi_vortex_0)
psi_vortex_1 = psi_vortex_1 / np.linalg.norm(psi_vortex_1)


# ============================================================
# Even vortex cat state
# |C_+^(theta)> ~ U_theta(|alpha> + |-alpha>)
# ============================================================
psi_even_cat = psi_plus + psi_minus
psi_even_cat = psi_even_cat / np.linalg.norm(psi_even_cat)

psi_vortex_even = vortex_phase * psi_even_cat
psi_vortex_even = psi_vortex_even / np.linalg.norm(psi_vortex_even)

# Density matrix
rho = np.outer(psi_vortex_even, np.conjugate(psi_vortex_even))


# ============================================================
# Compute approximate physical phase-space markers
# using <a> for the vortex logical states
# ============================================================
a0 = exp_a(psi_vortex_0)
a1 = exp_a(psi_vortex_1)

x0 = np.sqrt(2.0) * np.real(a0)
p0 = np.sqrt(2.0) * np.imag(a0)

x1 = np.sqrt(2.0) * np.real(a1)
p1 = np.sqrt(2.0) * np.imag(a1)

print("Vortex logical 0 marker:", (x0, p0))
print("Vortex logical 1 marker:", (x1, p1))


# ============================================================
# Wigner function from density matrix in Fock basis
#
# Convention:
# x = sqrt(2) Re(beta), p = sqrt(2) Im(beta)
# ============================================================
def wigner_from_rho(rho, xvec, pvec):
    X, P = np.meshgrid(xvec, pvec, indexing="xy")
    Z = np.sqrt(2.0) * (X + 1j * P)
    R2 = X**2 + P**2

    W = np.zeros_like(X, dtype=complex)

    N = rho.shape[0]
    fact = factorial(np.arange(N), exact=False)

    for m in range(N):
        for n in range(m + 1):
            rho_mn = rho[m, n]
            if abs(rho_mn) < 1e-14:
                continue

            pref = ((-1)**n / np.pi) * np.sqrt(fact[n] / fact[m])
            lag = eval_genlaguerre(n, m - n, 2.0 * R2)
            term = pref * (Z ** (m - n)) * lag * np.exp(-R2)

            if m == n:
                W += rho_mn * term
            else:
                W += 2.0 * np.real(rho_mn * term)

    return np.real(W)


# ============================================================
# Phase-space grid
# ============================================================
xvec = np.linspace(-xmax, xmax, n_grid)
pvec = np.linspace(-xmax, xmax, n_grid)

W = wigner_from_rho(rho, xvec, pvec)


# ============================================================
# Plot
# ============================================================
X, P = np.meshgrid(xvec, pvec, indexing="xy")

plt.figure(figsize=(7, 6))

contour = plt.contourf(X, P, W, levels=120)
plt.colorbar(contour, label=r"$W(x,p)$")

# ------------------------------------------------------------
# Original cat wells (for reference)
# ------------------------------------------------------------
x_alpha = np.sqrt(2.0) * np.real(alpha)
p_alpha = np.sqrt(2.0) * np.imag(alpha)

plt.scatter(
    [x_alpha, -x_alpha],
    [p_alpha, -p_alpha],
    marker="x",
    s=100,
    color="red",
    label=r"original cat wells $|\pm\alpha\rangle$"
)

# ------------------------------------------------------------
# Vortex logical zero and one markers
# ------------------------------------------------------------
plt.scatter(
    [x0],
    [p0],
    marker="o",
    s=120,
    edgecolors="black",
    linewidths=1.5,
    label=r"vortex logical $|0_L\rangle_\theta$"
)

plt.scatter(
    [x1],
    [p1],
    marker="s",
    s=120,
    edgecolors="black",
    linewidths=1.5,
    label=r"vortex logical $|1_L\rangle_\theta$"
)

plt.text(x0 + 0.15, p0 + 0.15, r"$0_L^\theta$", fontsize=11)
plt.text(x1 + 0.15, p1 + 0.15, r"$1_L^\theta$", fontsize=11)

plt.xlabel(r"$x$")
plt.ylabel(r"$p$")
plt.title(
    rf"2D Wigner function of the vortex even cat state"
    "\n"
    rf"$U_\theta(|\alpha\rangle + |-\alpha\rangle)$ with $\alpha={alpha}$, $\theta={theta}$"
)
plt.legend(loc="upper right", fontsize=9)
plt.axis("equal")
plt.tight_layout()

plt.savefig(output_file, dpi=300, bbox_inches="tight")
print(f"Saved plot to: {output_file.resolve()}")

plt.show()