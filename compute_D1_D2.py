import numpy as np
from tqdm import tqdm

def compute_D1_D2_one_theta(xi0, xi1, xi_edges, xi_min_counts: int):
    """Conditional first/second moments of the increment for one time slice.

    The increment ``xi1 - xi0`` is binned by the initial value ``xi0`` and
    averaged within each bin. Bins with fewer than ``xi_min_counts`` samples
    are set to NaN.

    Parameters
    ----------
    xi0, xi1 : ndarray of shape (n_trajectories,)
        Trajectory values at the initial and final time of the slice.
    xi_edges : ndarray of shape (n_xi_bins + 1,)
        Monotonic bin edges over the xi axis.
    xi_min_counts : int
        Minimum samples per bin; bins below this are masked with NaN.

    Returns
    -------
    M1, M2 : ndarray of shape (n_xi_bins,)
        First and second conditional Kramers-Moyal moments, per bin.
    counts : ndarray of shape (n_xi_bins,)
        Number of samples falling in each bin.
    """
    n_xi_bins = len(xi_edges) - 1
    idx = np.digitize(xi0, xi_edges) - 1

    xi_inc = xi1 - xi0
    valid = (idx >= 0) & (idx < n_xi_bins) & np.isfinite(xi_inc)
    idx, xi_inc = idx[valid], xi_inc[valid]

    counts = np.bincount(idx, minlength=n_xi_bins).astype(float)
    sum1 = np.bincount(idx, weights=xi_inc, minlength=n_xi_bins)
    sum2 = np.bincount(idx, weights=xi_inc**2, minlength=n_xi_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        M1 = sum1 / counts
        M2 = sum2 / counts

    poor = counts < xi_min_counts
    M1[poor] = np.nan
    M2[poor] = np.nan
    return M1, M2, counts


def compute_D1_D2(
    data: np.ndarray,
    delta_theta_us: int,
    dt: float,
    n_xi_bins: int,
    xi_min_counts: int,
    n_theta_centers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the drift D1 and diffusion D2 coefficients over time slices.

    For each of ``n_theta_centers`` time slices, the conditional moments of the
    increment over a lag of ``delta_theta_us`` samples are computed on a global
    xi grid and converted to the Kramers-Moyal coefficients D1 and D2 (the
    latter with a finite-time correction).

    Parameters
    ----------
    data : ndarray of shape (n_trajectories, theta_size)
        Ensemble of trajectories sampled at a fixed time step ``dt``.
    delta_theta_us : int
        Time lag of the increment, in samples.
    dt : float
        Physical time step between consecutive samples.
    n_xi_bins : int
        Number of bins for the xi axis.
    xi_min_counts : int
        Minimum samples per bin; bins below this are masked with NaN.
    n_theta_centers : int
        Number of time slices at which to estimate the coefficients.

    Returns
    -------
    xi_centers : ndarray of shape (n_xi_bins,)
        Bin centers of the xi axis.
    theta_centers : ndarray of shape (n_theta_centers,)
        Physical times of the slices.
    D1_ij, D2_ij : ndarray of shape (n_theta_centers, n_xi_bins)
        Estimated drift and diffusion coefficients.

    Raises
    ------
    ValueError
        If ``n_theta_centers`` exceeds the number of available time slices.
    """
    _, theta_size = data.shape
    delta_theta = delta_theta_us * dt

    # Global xi-grid (shared across time so the meshgrid is rectangular).
    xi_edges = np.linspace(data.min(), data.max(), n_xi_bins + 1)
    xi_centers = 0.5 * (xi_edges[:-1] + xi_edges[1:])

    last_valid_theta = theta_size - delta_theta_us
    if n_theta_centers >= last_valid_theta:
        raise ValueError(
            f"n_theta_centers ({n_theta_centers}) must be smaller than the "
            f"number of available time slices ({last_valid_theta})."
        )

    theta_idx_arr = np.linspace(0, last_valid_theta - 1, n_theta_centers).astype(int)

    D1_ij = np.full((theta_idx_arr.size, n_xi_bins), np.nan)
    D2_ij = np.full((theta_idx_arr.size, n_xi_bins), np.nan)

    for i, theta_idx in tqdm(enumerate(theta_idx_arr)):
        xi0 = data[:, theta_idx]
        xi1 = data[:, theta_idx + delta_theta_us]
        M1, M2, _ = compute_D1_D2_one_theta(xi0, xi1, xi_edges, xi_min_counts)

        D1 = M1 / delta_theta
        # finite-time correction: subtract (delta_theta * D1)^2 from M2
        D2 = (M2 - (delta_theta * D1) ** 2) / (2.0 * delta_theta)
        D1_ij[i] = D1
        D2_ij[i] = D2

    theta_centers = theta_idx_arr * dt
    return xi_centers, theta_centers, D1_ij, D2_ij
