import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

sys.path.append("../")

from wilcoxon_test import compute_wilcoxon_test

def _autocorr_time(r: np.ndarray) -> int:
    """Lag where the autocorrelation first drops below 1/e."""
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 1
    r = r - r.mean()
    acf = np.correlate(r, r, mode="full")[len(r) - 1:]
    acf /= acf[0]
    below = int(np.argmax(acf < np.exp(-1.0)))
    return below if below > 0 else len(r)

def robust_crossing(
    x: np.ndarray,
    y: np.ndarray,
    level: float = 1.0,
    smooth: int | None = None,
    mask_below: float = 0.0,
) -> float | None:
    """First downward crossing of `level` by the robust trend of y.

    ``mask_below`` ignora los primeros lags: en estos datos T tiene un pozo
    espurio en lags muy chicos (xi0, xi1, xi2 casi identicos => distribuciones
    condicionadas triviales), que NO es el tiempo de Markov-Einstein. El Delta_EM
    fisico es el cruce de la derecha, donde T decae y se asienta en 1.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if smooth is None:
        finite = y[np.isfinite(y)]
        tail = finite[-max(len(finite) // 4, 1):]
        smooth = max(min(_autocorr_time(tail), len(y) // 3), 3)

    ys = median_filter(y, size=smooth)

    keep = x >= mask_below
    xk, yk = x[keep], ys[keep]

    d = yk - level
    cross = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
    if cross.size == 0:
        return None

    i = int(cross[0])
    return float(xk[i] - d[i] * (xk[i + 1] - xk[i]) / (d[i + 1] - d[i]))

save_path = './magnitudes/ghost_sim'
x_matrix = np.load(os.path.join(save_path, 'x_matrix_non_periodic_HIT512_HPMR_st05.npy'))
x_matrix = x_matrix - x_matrix[:, :, [0]]

dt    = 0.015
nbins = 40

# (tiempos, particulas, componentes) -> promedio sobre componentes,
# transpuesto a (particulas, tiempos) que es lo que espera compute_wilcoxon_test
#x_matrix_mean = np.mean(x_matrix, axis=2).T
x_matrix = np.sqrt(np.sum(x_matrix**2, axis=0))**2

# el el eje x son posiciones en la caja periodica [0, 2pi]: condicionamos en el
# centro de la distribucion (mediana), no en 0 donde no hay particulas
xi0_c = float(np.median(x_matrix[:, 0]))

# el lag maximo permitido es (n_tiempos - 1) / 2 muestras
n_times = x_matrix.shape[1]
end_scale = (n_times - 1) // 2 * dt

delta_theta_arr, wt_arr = compute_wilcoxon_test(
        data  = x_matrix,
        dt    = dt,
        nbins = nbins,
        end_scale = end_scale,
        xi0_c     = xi0_c,
        )

# enmascaramos el pozo espurio de lags chicos (ver docstring de robust_crossing)
delta_em_us = robust_crossing(delta_theta_arr, wt_arr, mask_below=25)
if delta_em_us is None:
    # sin cruce por debajo de 1: usamos el minimo de la curva como candidato
    delta_em_us = float(delta_theta_arr[np.nanargmin(wt_arr)])
    print(f"# Sin cruce T<1; minimo de T en {delta_em_us:.0f} muestras "
          f"= {delta_em_us * dt:.4f} s (T={np.nanmin(wt_arr):.2f})")
else:
    print(f"# Delta_EM detectado: {delta_em_us:.1f} muestras = {delta_em_us * dt:.4f} s")
np.save(os.path.join(save_path, 'delta_em_us.npy'), delta_em_us)

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(delta_theta_arr, wt_arr, "o:", color="gray")
ax.axhline(1, ls="--", lw=2, color="black")
if delta_em_us is not None:
    ax.axvline(delta_em_us, color="red", label=r"Detected $\Delta \text{EM}$")
    ax.legend()

ax.set_xlabel(r"$\Delta\theta$ [muestras]")
ax.set_ylabel("Wilcoxon T")

plt.show()
