import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

from compute_D1_D2 import compute_D1_D2

delta_em = 2.0
path = f"magnitudes/traj_x_DeltaEM_{str(delta_em).replace('.', '-')}_cuadratic.npy"
data = np.load(path)
dt = 5e-3
expected_delta_us = round(delta_em / dt)

xi_centers, theta_centers, D1, D2 = compute_D1_D2(
    data=data,
    delta_theta_us=expected_delta_us,
    dt=dt,
    n_xi_bins=100,
    xi_min_counts=200,
    n_theta_centers=50,
)
xi_mesh, theta_mesh = np.meshgrid(
    xi_centers, theta_centers, indexing="ij"
)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection="3d")

ax.scatter(xi_mesh, theta_mesh, D1.T)

ax.set_xlabel(r"$\xi$")
ax.set_ylabel(r"$\theta$")
ax.set_zlabel("$D_2$")

plt.show()
