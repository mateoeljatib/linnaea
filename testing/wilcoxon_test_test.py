import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from tqdm import tqdm

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
) -> float | None:
    """First downward crossing of `level` by the robust trend of y.

    The trend is a median filter (robust to plateau scatter and outliers);
    the crossing is interpolated between the two samples that bracket it.
    `smooth` defaults to one autocorrelation time of the tail.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if smooth is None:
        finite = y[np.isfinite(y)]
        tail = finite[-max(len(finite) // 4, 1):]
        smooth = max(min(_autocorr_time(tail), len(y) // 3), 3)

    ys = median_filter(y, size=smooth)

    d = ys - level
    cross = np.where((d[:-1] > 0) & (d[1:] <= 0))[0]
    if cross.size == 0:
        return None

    i = int(cross[0])
    return float(x[i] - d[i] * (x[i + 1] - x[i]) / (d[i + 1] - d[i]))

delta_em = 2.0
path = f"magnitudes/traj_x_DeltaEM_{str(delta_em).replace('.', '-')}_cuadratic.npy"
data = np.load(path)
dt = 5e-3
expected_delta = delta_em / dt

detected_delta_list = []
# nbins_list = np.arange(20, 200, 1)#[20, 50, 100, 150]
nbins_list = [50]
for nbins in tqdm(nbins_list):
    delta_theta_arr_us, wt_arr = compute_wilcoxon_test(
        data, dt=dt, nbins=nbins, end_scale=10.0, xi0_c = 0.0
    )
    try:
        detected_delta = robust_crossing(delta_theta_arr_us, wt_arr)
    except:
        detected_delta = np.nan
    detected_delta_list.append(detected_delta)


# fig, ax = plt.subplots(figsize=(10, 6))
#
# ax.plot(nbins_list, detected_delta_list)
#
# plt.show()
    

# fig, ax = plt.subplots(figsize=(10, 6))
#
# ax.plot(delta_theta_arr_us, wt_arr, "o:", color="gray")
# ax.axhline(1, ls="--", lw=2, color="black")
# # ax.axvline(expected_delta, color="red", ls="--", label=r"Expected $\Delta \text{EM}$")
# ax.axvline(detected_delta, color="red", label=r"Detected $\Delta \text{EM}$")
#
# ax.legend()
# plt.show()
