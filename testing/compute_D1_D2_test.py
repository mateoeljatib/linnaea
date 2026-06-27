import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

from compute_D1_D2 import compute_D1_D2

# Parameters must match those used in langevin_em_cuadratic_D2.py
gamma = 1.0
sigma0 = 1.0
sigma2 = 0.3
alpha = 0.05

delta_em = 2.0
path = f"magnitudes/traj_x_DeltaEM_{str(delta_em).replace('.', '-')}_cuadratic.npy"
data = np.load(path)
dt = 5e-3
expected_delta_us = round(delta_em / dt)

n_xi_bins       = 30
xi_min_counts   = 50
n_theta_centers = 50

xi_centers, theta_centers, D1, D2 = compute_D1_D2(
    data=data,
    delta_theta_us=expected_delta_us,
    dt=dt,
    n_xi_bins=n_xi_bins,
    xi_min_counts=xi_min_counts,
    n_theta_centers=n_theta_centers,
)
xi_mesh, theta_mesh = np.meshgrid(xi_centers, theta_centers, indexing="ij")

# Finite-lag KM corrections via Lyapunov analysis of the linearized 2D system.
#
# The simulation drives x with colored OU noise η (correlation time tau_em).
# In the stationary state, x and η are CORRELATED, so conditioning on x₀ also
# shifts E[η₀|x₀]. This makes the KM estimates differ strongly from the bare
# Fokker-Planck coefficients. The correct predictions come from:
#   E[x(Δθ)|x₀] = ρ·x₀      →   D1_KM = (ρ-1)/Δθ · xi
#   Var[x(Δθ)|x₀] / (2Δθ)   →   D2_KM = D2_bare · correction
# where ρ and the correction are derived from the Lyapunov equation of the
# linearized system  A·[x;η]  with  A = [[-γ, σ₀],[0, -1/τ]].
delta_theta = expected_delta_us * dt   # physical lag = delta_em
tau_em = delta_em

# 2D system matrix and noise matrix (additive approximation: σ_eff = σ₀)
_A = np.array([[-gamma, sigma0], [0., -1. / tau_em]])
_Q = np.array([[0., 0.], [0., 2. / tau_em ** 2]])

# Stationary covariance S: _A @ S + S @ _A.T + _Q = 0  (Kronecker form)
_K = np.kron(np.eye(2), _A) + np.kron(_A, np.eye(2))
_S = np.linalg.solve(_K, -_Q.ravel()).reshape(2, 2)

# Time-lagged covariance Cov(x(Δθ), x(0)) via matrix exponential of _A
# For 2×2 upper-triangular with distinct eigenvalues λ₁=-γ, λ₂=-1/τ:
_l1, _l2 = -gamma, -1. / tau_em
_Phi00 = np.exp(_l1 * delta_theta)
_Phi01 = sigma0 * (np.exp(_l1 * delta_theta) - np.exp(_l2 * delta_theta)) / (_l1 - _l2)
_cov_lag = _Phi00 * _S[0, 0] + _Phi01 * _S[1, 0]

# ρ = E[x(Δθ)|x₀] / x₀
_rho = _cov_lag / _S[0, 0]

# D1 theoretical slope and D2 scale correction
D1_slope_KM     = (_rho - 1.0) / delta_theta
_var_cond        = _S[0, 0] * (1.0 - _rho ** 2)
D2_correction_KM = _var_cond / (2.0 * delta_theta * sigma0 ** 2)

D1_theory = D1_slope_KM * xi_mesh
D2_theory = (sigma0**2 + sigma2**2 * xi_mesh**2) * (1.0 + alpha * theta_mesh) * D2_correction_KM

# ── DIAGNÓSTICO ──────────────────────────────────────────────────────────────
_xi_edges = np.linspace(data.min(), data.max(), n_xi_bins + 1)
_bin_width = _xi_edges[1] - _xi_edges[0]

# Máscaras de valores válidos (no-NaN); forma: (n_theta, n_xi)
_ok1 = np.isfinite(D1)
_ok2 = np.isfinite(D2)

# Cobertura por slice de theta
_n_ok1_per_th = _ok1.sum(axis=1)   # cuántos bins válidos en cada theta
_n_ok2_per_th = _ok2.sum(axis=1)

# Rango de xi cubierto (bins válidos en ≥50 % de los slices de theta)
_xi_always = xi_centers[_ok2.mean(axis=0) >= 0.5]
_xi_lo = _xi_always.min() if len(_xi_always) else float("nan")
_xi_hi = _xi_always.max() if len(_xi_always) else float("nan")

# Cuentas por bin en el slice central
_th_c_idx = len(theta_centers) // 2
_col_c    = min(int(round(theta_centers[_th_c_idx] / dt)), data.shape[1] - 1)
_counts_c, _ = np.histogram(data[:, _col_c], bins=_xi_edges)
_counts_pos   = _counts_c[_counts_c > 0]
_sigma_xi     = data[:, _col_c].std()

# Aplanar D1/D2 válidos con sus xi y theta correspondientes
_xiT = xi_mesh.T     # (n_theta, n_xi): _xiT[i,j] = xi_centers[j]
_thT = theta_mesh.T  # (n_theta, n_xi): _thT[i,j] = theta_centers[i]

_xi1, _th1, _d1 = _xiT[_ok1], _thT[_ok1], D1[_ok1]
_xi2, _th2, _d2 = _xiT[_ok2], _thT[_ok2], D2[_ok2]

# Ajuste lineal D1 ~ b·xi (sin intercept, pasa por 0 por simetría)
_b_D1  = np.dot(_xi1, _d1) / np.dot(_xi1, _xi1) if len(_xi1) > 1 else float("nan")
_r_D1  = float(np.corrcoef(_xi1, _d1)[0, 1])    if len(_xi1) > 1 else float("nan")

# Correlaciones para D2
_r_D2_xi2   = float(np.corrcoef(_xi2**2, _d2)[0, 1]) if len(_d2) > 1 else float("nan")
_r_D2_theta = float(np.corrcoef(_th2,    _d2)[0, 1]) if len(_d2) > 1 else float("nan")

# Residuos respecto a la teoría corregida
_d1_th = D1_slope_KM * _xi1
_d2_th = (sigma0**2 + sigma2**2 * _xi2**2) * (1 + alpha * _th2) * D2_correction_KM
_rms_D1 = float(np.sqrt(np.mean((_d1 - _d1_th)**2)))
_rms_D2 = float(np.sqrt(np.mean((_d2 - _d2_th)**2)))

# D2 en xi ≈ 0 promediado sobre theta, vs teoría
_xi0_idx       = int(np.argmin(np.abs(xi_centers)))
_D2_at_xi0     = float(np.nanmean(D2[:, _xi0_idx]))
_D2_th_at_xi0  = float(sigma0**2 * np.mean(1 + alpha * theta_centers) * D2_correction_KM)

_SEP = "=" * 62
print(_SEP)
print("  DIAGNÓSTICO  compute_D1_D2")
print(_SEP)

print(f"\n  [Parámetros]")
print(f"    n_xi_bins       = {n_xi_bins:<6}  delta_em      = {delta_em}")
print(f"    xi_min_counts   = {xi_min_counts:<6}  delta_theta   = {delta_theta:.4f} s")
print(f"    n_theta_centers = {n_theta_centers:<6}  τ_em          = {tau_em:.4f} s")
print(f"    [Lyapunov]  ρ  = {_rho:.4f}   D1_slope_KM = {D1_slope_KM:.4f}   D2_correction = {D2_correction_KM:.4f}")

print(f"\n  [Grid xi]")
print(f"    Rango global datos : [{data.min():.2f}, {data.max():.2f}]")
print(f"    Ancho de bin       : {_bin_width:.4f}")
print(f"    σ_xi (slice θ_c)   : {_sigma_xi:.3f}")

print(f"\n  [Cobertura — slice θ central = {theta_centers[_th_c_idx]:.1f} s]")
if len(_counts_pos):
    print(f"    Cuentas/bin : mediana={int(np.median(_counts_pos))}  "
          f"min={_counts_pos.min()}  max={_counts_pos.max()}")
print(f"    Bins ≥ umbral ({xi_min_counts})   : "
      f"{(_counts_c >= xi_min_counts).sum()}/{n_xi_bins}  "
      f"→  xi ∈ [{_xi_lo:.2f}, {_xi_hi:.2f}]")
print(f"    Cobertura en σ     : |xi|_max / σ_xi = "
      f"{max(abs(_xi_lo), abs(_xi_hi)):.2f} / {_sigma_xi:.2f} = "
      f"{max(abs(_xi_lo), abs(_xi_hi)) / _sigma_xi:.1f} σ")

print(f"\n  [Cobertura — todos los θ]")
_tot = n_theta_centers * n_xi_bins
print(f"    Celdas válidas D1 : {_ok1.sum()}/{_tot}  "
      f"({100*_ok1.mean():.1f} %)")
print(f"    Celdas válidas D2 : {_ok2.sum()}/{_tot}  "
      f"({100*_ok2.mean():.1f} %)")
print(f"    Bins válidos/slice: media={_n_ok2_per_th.mean():.1f}  "
      f"std={_n_ok2_per_th.std():.1f}  "
      f"min={_n_ok2_per_th.min()}  max={_n_ok2_per_th.max()}")

print(f"\n  [Forma D1  — esperado: lineal en xi]")
print(f"    Correlación D1 vs xi : r = {_r_D1:.3f}   (→ -1 si lineal)")
print(f"    Pendiente ajustada   : b = {_b_D1:.3f}   "
      f"(KM a lag Δθ: {D1_slope_KM:.3f},  FP desnudo: {-gamma:.3f})")
print(f"    RMS residuo vs teoría: {_rms_D1:.4f}")

print(f"\n  [Forma D2  — esperado: cuadrático en xi, lineal en θ]")
print(f"    Correlación D2 vs xi²  : r = {_r_D2_xi2:.3f}   (→ +1 cuanto más curvatura)")
print(f"    Correlación D2 vs θ    : r = {_r_D2_theta:.3f}   (→ +1 si α > 0)")
print(f"    D2 en xi≈0 (media θ)   : {_D2_at_xi0:.4f}   "
      f"(teoría corregida: {_D2_th_at_xi0:.4f})")
print(f"    Rango D2 computado     : [{np.nanmin(D2):.3f}, {np.nanmax(D2):.3f}]")
print(f"    Rango D2 teoría (corr) : [{np.nanmin(_d2_th):.3f}, {np.nanmax(_d2_th):.3f}]")
print(f"    RMS residuo vs teoría  : {_rms_D2:.4f}")

print(f"\n{_SEP}\n")
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.scatter(xi_mesh, theta_mesh, D1.T, s=2, alpha=0.6, color="steelblue", label="computed")
ax1.plot_surface(xi_mesh, theta_mesh, D1_theory, alpha=0.3, color="orange")
ax1.set_xlabel(r"$\xi$")
ax1.set_ylabel(r"$\theta$")
ax1.set_zlabel("$D_1$")
ax1.set_title("Drift $D_1$")

ax2 = fig.add_subplot(1, 2, 2, projection="3d")
ax2.scatter(xi_mesh, theta_mesh, D2.T, s=2, alpha=0.6, color="steelblue", label="computed")
ax2.plot_surface(xi_mesh, theta_mesh, D2_theory, alpha=0.3, color="orange")
ax2.set_xlabel(r"$\xi$")
ax2.set_ylabel(r"$\theta$")
ax2.set_zlabel("$D_2$")
ax2.set_title("Diffusion $D_2$")

plt.tight_layout()
plt.show()
