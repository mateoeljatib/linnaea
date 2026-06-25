import numpy as np
from tqdm import tqdm

SQRT_2_DIV_PI = np.sqrt(2 / np.pi)

def wilcoxon_test_2samp(p1, p2):
    """
    Test if p(df)1 it is statistically distributed as p(df)2.
    """
    m = p1.size
    n = p2.size

    sp1 = np.sort(p1)
    sp2 = np.sort(p2)

    # Q = np.sum(sp2[:, np.newaxis] > sp1) is faster but memory inefficient
    Q = 0
    for val2 in sp2:
        Q += np.sum(val2 >= sp1)

    Q_mean = n * m / 2
    Q_sigma = np.sqrt(n * m * (n + m + 1) / 12)
    T = np.abs(Q - Q_mean) / (Q_sigma * SQRT_2_DIV_PI)

    return T

def compute_wilcoxon_test(
    data, dt, nbins, end_scale, xi0_c = 0.0
):
    to_us = 1/dt  # convert to unit of samples

    # calculate delta_theta_arr_us
    start_scale_us = np.round(dt * to_us)
    end_scale_us = np.round(end_scale * to_us)
    delta_theta_arr_us = np.arange(start_scale_us, end_scale_us, 1)

    max_idx = int(2 * delta_theta_arr_us[-1])
    if max_idx >= data.shape[1]:
        max_end_scale = (data.shape[1] - 1) / 2 * dt
        raise ValueError(f"end_scale={end_scale} too large, max allowed is {max_end_scale}")

    wt_arr = np.zeros((delta_theta_arr_us.size,))
    for ii, delta_theta_us in enumerate(
        tqdm(
            delta_theta_arr_us,
            desc="# Δ ",
            bar_format=r"{desc}: |{bar}{r_bar}",
        )
    ):
        xi0 = data[:, 0+0*delta_theta_us] # xi(theta_0)
        xi1 = data[:, 0+1*delta_theta_us] # xi(theta_1) | theta_1 = theta_0 + Delta theta
        xi2 = data[:, 0+2*delta_theta_us] # xi(theta_2) | theta_2 = theta_1 + Delta theta

        # bins1
        count1, bins1_edges = np.histogram(xi1, bins=nbins)
        bins1_width = bins1_edges[1] - bins1_edges[0]

        bins1 = [
            (bins1_edges[i], bins1_edges[i] + bins1_width) for i in range(count1.size)
        ]

        bins0_width = bins1_width
        # bin0 (only one, i.e. idx_c0)
        # idx_c0 = np.abs(xi0) < bins0_width
        idx_c0 = np.abs(xi0 - xi0_c) < bins0_width

        # mean of the wilcoxon test stats over all bins1
        tmp = wilcoxon_test_multiple_bins(xi1, xi2, bins1, idx_c0)

        wt_arr[ii] = tmp

    return delta_theta_arr_us, wt_arr

def wilcoxon_test_multiple_bins(xi1, xi2, bins1, idx_c0):
    T_list = []
    for b1 in bins1:
        # - P(u_2|u_1) i.e. xi2 only where xi1 belongs to the nth bin
        idx_c1 = (xi1 > b1[0]) & (xi1 < b1[1])
        inc2_c1 = xi2[idx_c1]

        # - P(u_2|u_1,u_0=0)
        inc2_c1_c0 = xi2[idx_c1 & idx_c0]

        if inc2_c1.size > 40_000:  # we don't need that much data
            inc2_c1 = np.random.choice(inc2_c1, 40_000, replace=False)
        elif (
            inc2_c1.size < 30
        ):  # we need that much data (to be sure that we have enough data to calculate mean and have a Gaussian behaviour)
            continue  # skip this bin

        if inc2_c1_c0.size > 20_000:
            inc2_c1_c0 = np.random.choice(inc2_c1_c0, 20_000, replace=False)
        elif inc2_c1_c0.size < 30:
            continue

        # - test if P(u_2|u_1,u_0=0) is compatible with P(u_2|u_1)
        T = wilcoxon_test_2samp(inc2_c1, inc2_c1_c0)
        T_list.append(T)

    if len(T_list) == 0:
        raise np.nan
    else:
        return np.mean(T_list)
