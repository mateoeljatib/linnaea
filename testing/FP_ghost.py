import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")

from compute_D1_D2 import compute_D1_D2

save_path = './magnitudes/ghost_sim'
x_matrix = np.load(os.path.join(save_path, 'x_matrix_non_periodic_HIT512_HPMR_st05.npy'))
x_matrix = x_matrix - x_matrix[:, :, [0]]
dt = 0.015

n_xi_bins       = 70
xi_min_counts   = 50
n_theta_centers = 50
xi_percentile   = (2, 98)

# lag detectado por wilcoxon_ghost.py (en muestras)
delta_em_us = int(round(float(np.load(os.path.join(save_path, 'delta_em_us.npy')))))
delta_em_us = 215
x_matrix = np.sqrt(np.sum(x_matrix**2, axis=0))

xi_centers, theta_centers, D1, D2 = compute_D1_D2(
        data=x_matrix,
        delta_theta_us=delta_em_us,
        dt=dt,
        n_xi_bins=n_xi_bins,
        xi_min_counts=xi_min_counts,
        n_theta_centers=n_theta_centers,
        xi_percentile=xi_percentile,
        )

xi_mesh, theta_mesh = np.meshgrid(xi_centers, theta_centers, indexing='ij')

title_sim = "HIT512 HPMR st05 (ghost)"
title_est = (
    rf"Est:  $\Delta\theta$={delta_em_us} us   n_xi_bins={n_xi_bins}   "
    rf"xi_min_counts={xi_min_counts}   n_theta_centers={n_theta_centers}"
)

def plot_3d_figure():
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.scatter(xi_mesh, theta_mesh, D1.T, s=3, alpha=0.6, color="steelblue", label="computed")
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel(r"$\theta$")
    ax1.set_zlabel("$D_1$")
    ax1.set_title("Drift $D_1$")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.scatter(xi_mesh, theta_mesh, D2.T, s=3, alpha=0.6, color="steelblue", label="computed")
    ax2.set_xlabel(r"$\xi$")
    ax2.set_ylabel(r"$\theta$")
    ax2.set_zlabel("$D_2$")
    ax2.set_title("Diffusion $D_2$")
    plt.tight_layout()
    plt.show()

plot_3d_figure()


def cut_figure(theta_idxs=None):

    if theta_idxs is None:
        # Tres cortes repartidos, evitando los extremos (transitorio inicial /
        # último θ con menos estadística).
        theta_idxs = np.linspace(0, n_theta_centers - 1, 5).astype(int)[1:4]

    fig, axes = plt.subplots(3, 2, figsize=(11, 11), sharex=True)

    for row, ti in enumerate(theta_idxs):
        th = theta_centers[ti]

        # —— Columna izquierda: D1 ——
        axL = axes[row, 0]
        axL.scatter(xi_centers, D1[ti], s=14, color="steelblue",
                    alpha=0.7, label="datos")
        axL.axhline(0.0, color="gray", lw=0.5)
        axL.set_ylabel(r"$D_1$")
        axL.set_title(rf"$D_1$  —  $\theta = {th:.1f}$ s")
        axL.grid(alpha=0.3)

        # —— Columna derecha: D2 ——
        axR = axes[row, 1]
        axR.scatter(xi_centers, D2[ti], s=14, color="steelblue",
                    alpha=0.7, label="datos")
        axR.set_ylabel(r"$D_2$")
        axR.set_title(rf"$D_2$  —  $\theta = {th:.1f}$ s")
        axR.grid(alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel(r"$\xi$")
    axes[0, 0].legend()
    axes[0, 1].legend()

    fig.suptitle(title_sim + "\n" + title_est, fontsize=9, family="monospace")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    return fig, axes

cut_figure()
