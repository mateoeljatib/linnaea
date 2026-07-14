"""Validación de la teoría en el LÍMITE BLANCO (ruido delta-correlacionado).

Objetivo: chequear si el desajuste teoría–datos que aparece en el régimen de
ruido coloreado (compute_D1_D2_test.py) viene de la TEORÍA y no de los datos.

En el límite blanco la SDE es de Stratonovich y sus coeficientes de
Fokker-Planck son EXACTOS y conocidos:

    D1_FP(x, θ) = -γ·x + σ2²·x·(1 + α·θ)      ← incluye el DRIFT INDUCIDO
                    │                             por el ruido multiplicativo
                    └ parte lineal "ingenua"     (½·∂ₓ(2·D2) = σ2²·x·(1+αθ))

    D2_FP(x, θ) = (σ0² + σ2²·x²)·(1 + α·θ)     ← FP desnudo, sin correcciones

Diferencias clave con compute_D1_D2_test.py:
  · La simulación usa delta_em ≤ 0  → rama Stratonovich-Heun (estable; no hay
    1/τ que explote).
  · El lag de Kramers-Moyal está DESACOPLADO del ruido: se mide a `lag_us`
    pasos (chico), no a round(delta_em/dt).
  · NO se aplica la corrección de Lyapunov (es un artefacto del ruido coloreado
    que → 1 en el límite blanco a lag chico). Se compara contra el FP desnudo.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

from compute_D1_D2 import compute_D1_D2
from langevin_em_cuadratic_D2 import simulate

# ═══════════════════════════════════════════════════════════════════════════
#  PARÁMETROS  —  único lugar para editar.
# ═══════════════════════════════════════════════════════════════════════════
# Modelo físico
gamma  = 1.5
sigma0 = 2.0
sigma2 = 0.6
alpha  = 0.05

# Integración de la simulación (límite blanco: delta_em ≤ 0 → rama Stratonovich)
delta_em_sim = 0.0        # cualquier valor ≤ 0 dispara la rama de ruido blanco
dt           = 5e-3
T_sim        = 40.0
npart_sim    = 20_000
seed_sim     = 42

# Estimación Kramers-Moyal
# lag_us: lag del incremento EN PASOS, desacoplado del ruido. Trade-off:
#   · D1 quiere lag grande  (mejor SNR: el drift es débil frente a la difusión)
#   · D2 quiere lag chico   (menos washout de la curvatura)
# 20 pasos (Δθ=0.1s) es buen compromiso: << relajación 1/γ, y el drift inducido
# por el ruido se ve nítido. Debe cumplirse lag_us << (1/γ)/dt.
lag_us          = 20
n_xi_bins       = 50
xi_min_counts   = 50
n_theta_centers = 50
xi_percentile   = (2, 98)
# ═══════════════════════════════════════════════════════════════════════════


def load_or_simulate():
    """Trayectorias en el límite blanco para los params de arriba (con caché)."""
    tag = (
        f"white_g{gamma}_s0{sigma0}_s2{sigma2}_a{alpha}"
        f"_dt{dt}_T{T_sim}_N{npart_sim}_seed{seed_sim}"
    ).replace(".", "-")
    path = os.path.join("magnitudes", f"traj_{tag}.npy")

    if os.path.exists(path):
        print(f"[datos] cache encontrada → cargando  {path}")
        return np.load(path)

    print(f"[datos] sin cache → simulando (límite blanco)  {path}")
    x = simulate(
        delta_em_sim, gamma=gamma, sigma0=sigma0, sigma2=sigma2, alpha=alpha,
        dt=dt, T=T_sim, npart=npart_sim, seed=seed_sim,
    )
    os.makedirs("magnitudes", exist_ok=True)
    np.save(path, x)
    return x


data = load_or_simulate()

xi_centers, theta_centers, D1, D2 = compute_D1_D2(
    data=data,
    delta_theta_us=lag_us,
    dt=dt,
    n_xi_bins=n_xi_bins,
    xi_min_counts=xi_min_counts,
    n_theta_centers=n_theta_centers,
    xi_percentile=xi_percentile,
)
xi_mesh, theta_mesh = np.meshgrid(xi_centers, theta_centers, indexing="ij")

# ── Teoría exacta de Fokker-Planck en el límite blanco (SIN Lyapunov) ────────
D1_theory_naive = -gamma * xi_mesh                                  # solo lineal
D1_theory       = -gamma * xi_mesh + sigma2**2 * xi_mesh * (1.0 + alpha * theta_mesh)
D2_theory       = (sigma0**2 + sigma2**2 * xi_mesh**2) * (1.0 + alpha * theta_mesh)

# ── DIAGNÓSTICO: ¿la pendiente de D1 confirma el drift inducido? ─────────────
_ok1 = np.isfinite(D1)
_ok2 = np.isfinite(D2)
_xiT = xi_mesh.T      # (n_theta, n_xi)
_thT = theta_mesh.T

_xi1, _th1, _d1 = _xiT[_ok1], _thT[_ok1], D1[_ok1]
_xi2, _th2, _d2 = _xiT[_ok2], _thT[_ok2], D2[_ok2]

# RMS de D1 contra las dos teorías (menor = mejor)
_d1_naive = -gamma * _xi1
_d1_corr  = -gamma * _xi1 + sigma2**2 * _xi1 * (1.0 + alpha * _th1)
_rms_D1_naive = float(np.sqrt(np.mean((_d1 - _d1_naive) ** 2)))
_rms_D1_corr  = float(np.sqrt(np.mean((_d1 - _d1_corr) ** 2)))

# RMS de D2 contra el FP desnudo
_d2_th = (sigma0**2 + sigma2**2 * _xi2**2) * (1.0 + alpha * _th2)
_rms_D2 = float(np.sqrt(np.mean((_d2 - _d2_th) ** 2)))

_SEP = "=" * 68
print(_SEP)
print("  VALIDACIÓN EN EL LÍMITE BLANCO  (FP exacto, sin Lyapunov)")
print(_SEP)
print(f"\n  [Parámetros]")
print(f"    γ={gamma}  σ0={sigma0}  σ2={sigma2}  α={alpha}   (ruido BLANCO)")
print(f"    lag = {lag_us} pasos  →  Δθ = {lag_us*dt:g} s")
print(f"    datos: std={data.std():.2f}   rango=[{data.min():.1f}, {data.max():.1f}]")
print(f"    celdas válidas D1/D2: {_ok1.sum()}/{_ok1.size}  ({100*_ok1.mean():.0f} %)")

print(f"\n  [D1 — test del drift inducido por el ruido]")
print(f"    RMS(D1 vs  -γx)                = {_rms_D1_naive:.3f}   (teoría ingenua)")
print(f"    RMS(D1 vs  -γx + σ2²x(1+αθ))   = {_rms_D1_corr:.3f}   (FP Stratonovich)  <- menor gana")
print(f"\n    Pendiente de D1 por corte de θ (fit lineal):")
print(f"      {'θ [s]':>7} | {'medida':>8} | {'-γ':>7} | {'-γ+σ2²(1+αθ)':>13}")
for ti in np.linspace(0, n_theta_centers - 1, 5).astype(int):
    th = theta_centers[ti]
    m = _ok1[ti]
    if m.sum() < 3:
        continue
    slope = float(np.polyfit(xi_centers[m], D1[ti][m], 1)[0])
    naive = -gamma
    corr = -gamma + sigma2**2 * (1.0 + alpha * th)
    print(f"      {th:7.1f} | {slope:8.3f} | {naive:7.2f} | {corr:13.3f}")

print(f"\n  [D2 — vs FP desnudo (σ0²+σ2²x²)(1+αθ)]")
print(f"    RMS(D2 vs FP) = {_rms_D2:.3f}")
_xi0 = int(np.argmin(np.abs(xi_centers)))
print(f"    D2 en x≈0 (media θ): dato={np.nanmean(D2[:, _xi0]):.3f}  "
      f"teo={sigma0**2 * np.mean(1 + alpha*theta_centers):.3f}")
print(f"\n{_SEP}\n")


def cut_figure(theta_idxs=None):
    """Cortes a θ fijo: D1 (izq) y D2 (der), datos vs teoría FP exacta.

    En D1 se muestran AMBAS teorías: la ingenua -γx (gris, punteada) y la
    corregida de Stratonovich (naranja). El punto del test es ver que los datos
    siguen a la naranja, no a la gris.
    """
    if theta_idxs is None:
        theta_idxs = np.linspace(0, n_theta_centers - 1, 5).astype(int)[1:4]

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)

    for row, ti in enumerate(theta_idxs):
        th = theta_centers[ti]

        axL = axes[row, 0]
        axL.scatter(xi_centers, D1[ti], s=14, color="steelblue", alpha=0.7, label="datos")
        axL.plot(xi_centers, D1_theory_naive[:, ti], color="gray", lw=1.5, ls="--",
                 label=r"teoría ingenua $-\gamma x$")
        axL.plot(xi_centers, D1_theory[:, ti], color="orange", lw=2,
                 label=r"FP $-\gamma x+\sigma_2^2 x(1+\alpha\theta)$")
        axL.axhline(0.0, color="gray", lw=0.5)
        axL.set_ylabel(r"$D_1$")
        axL.set_title(rf"$D_1$  —  $\theta = {th:.1f}$ s")
        axL.grid(alpha=0.3)

        axR = axes[row, 1]
        axR.scatter(xi_centers, D2[ti], s=14, color="steelblue", alpha=0.7, label="datos")
        axR.plot(xi_centers, D2_theory[:, ti], color="orange", lw=2, label="FP desnudo")
        axR.set_ylabel(r"$D_2$")
        axR.set_title(rf"$D_2$  —  $\theta = {th:.1f}$ s")
        axR.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel(r"$\xi$")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)

    title_sim = (
        rf"Límite BLANCO:  $\gamma$={gamma}   $\sigma_0$={sigma0}   "
        rf"$\sigma_2$={sigma2}   $\alpha$={alpha}   dt={dt}   T={T_sim}   "
        rf"npart={npart_sim}   seed={seed_sim}"
    )
    title_est = (
        rf"Estim:  lag={lag_us} pasos ($\Delta\theta$={lag_us*dt:g}s)   "
        rf"n_xi_bins={n_xi_bins}   xi_min_counts={xi_min_counts}   "
        rf"n_theta={n_theta_centers}   percentil={xi_percentile}   "
        rf"(sin corrección de Lyapunov)"
    )
    fig.suptitle(title_sim + "\n" + title_est, fontsize=9, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return fig, axes


cut_figure()
