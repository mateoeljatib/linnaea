import numpy as np


def simulate(
    delta_em,
    *,
    gamma=1.0,
    sigma0=1.0,
    sigma2=0.3,
    alpha=0.05,
    dt=5e-3,
    T=40.0,
    npart=3000,
    seed=0,
    dtype=np.float64,
):
    """
    Langevin with linear, time-independent drift  -gamma*x  and diffusion
        D2(x, t) = (sigma0**2 + sigma2**2 * x**2) * (1 + alpha * t)
    (quadratic in position, linear in time). Noise convention <G G'> = 2 delta.

    delta_em > 0  ->  driving noise is a colored OU process with correlation
                      time tau_em = delta_em (finite Einstein-Markov time).
    delta_em <= 0 ->  white limit; multiplicative Stratonovich SDE.

    Returns the trajectory matrix
        x.shape == (npart, n_steps)        # row = particle, column = time
    Multiplicative, non-autonomous noise -> no exact linear discretization;
    the core is a Heun integrator. dtype=np.float32 halves storage memory.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))

    def diff(xv, tv):
        """Diffusion coefficient D2(x, t), clipped to be non-negative."""
        return np.maximum((sigma0**2 + sigma2**2 * xv * xv) * (1.0 + alpha * tv), 0.0)

    xs = np.empty((n_steps, npart), dtype=dtype)
    x = rng.standard_normal(npart) * np.sqrt(sigma0**2 / gamma)  # additive-baseline IC

    if delta_em <= 0.0:  # white limit (Stratonovich Heun)
        sqdt = np.sqrt(dt)
        for k in range(n_steps):
            t = k * dt
            dW = sqdt * rng.standard_normal(npart)
            g1 = np.sqrt(2.0 * diff(x, t))
            xp = x - gamma * x * dt + g1 * dW  # Euler predictor
            g2 = np.sqrt(2.0 * diff(xp, t + dt))
            x = x - 0.5 * dt * gamma * (x + xp) + 0.5 * (g1 + g2) * dW
            xs[k] = x
        return xs.T

    tau_em = delta_em  # colored OU noise (exact 1D step)
    a = np.exp(-dt / tau_em)
    var_eta = 1.0 / tau_em  # stationary variance (-> 2*delta white limit)
    b = np.sqrt(var_eta * (1.0 - a * a))
    eta = rng.standard_normal(npart) * np.sqrt(var_eta)  # stationary IC for the noise

    for k in range(n_steps):
        t = k * dt
        eta_new = a * eta + b * rng.standard_normal(npart)  # eta at t + dt
        f1 = -gamma * x + np.sqrt(diff(x, t)) * eta  # Heun on the x-ODE
        xp = x + dt * f1
        f2 = -gamma * xp + np.sqrt(diff(xp, t + dt)) * eta_new
        x = x + 0.5 * dt * (f1 + f2)
        eta = eta_new
        xs[k] = x
    return xs.T  # view (npart, n_steps), no copy


def plot_trajectories(x, dt, n=6, seed=None):
    """Plot n random trajectories from the (npart, n_steps) matrix. Returns (fig, ax)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(n, x.shape[0]), replace=False)
    t = np.arange(x.shape[1]) * dt
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i in idx:
        ax.plot(t, x[i], lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_surfaces(*, gamma, sigma0, sigma2, alpha, x_range, t_range, nx, nt):
    """Plot the surfaces D1(x, t) and D2(x, t). Returns (fig, axes)."""
    import matplotlib.pyplot as plt

    x = np.linspace(*x_range, nx)
    t = np.linspace(*t_range, nt)
    X, T = np.meshgrid(x, t)

    D1 = -gamma * X
    D2 = (sigma0**2 + sigma2**2 * X * X) * (1.0 + alpha * T)

    fig = plt.figure(figsize=(11, 4.5))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(X, T, D1, cmap="viridis")
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("D1(x, t)")
    ax1.set_title("Drift")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(X, T, D2, cmap="viridis")
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("D2(x, t)")
    ax2.set_title("Diffusion")

    fig.tight_layout()
    plt.show()
    return fig, (ax1, ax2)


# --------------------------------------------------------------------------
if __name__ == "__main__":
    params = dict(
        gamma=1.0,
        sigma0=1.0,
        sigma2=0.3,
        alpha=0.05,
        dt=5e-3,
        T=40.0,
        npart=10_000,
        seed=42,
    )
    # deltas = [0.1, 0.5, 2.0, 5.0, 10.0]  # one matrix per Delta_EM
    deltas = [2.0]  # one matrix per Delta_EM
    save = True
    matrices = {}
    for de in deltas:
        # x = simulate(de, **params)
        # matrices[de] = x
        # if save:
        #     dem_str = f"{de}".replace(".", "-")
        #     np.save(f"magnitudes/traj_x_DeltaEM_{dem_str}_cuadratic.npy", x)
        # print(
        #     f"Delta_EM={de:<4}  shape={x.shape}  {x.dtype}  "
        #     f"{x.nbytes / 2**20:6.1f} MiB" + ("  -> saved" if save else "")
        # )
        # plot_trajectories(x, dt=5e-3)
        plot_surfaces(
            gamma=params["gamma"],
            sigma0=params["sigma0"],
            sigma2=params["sigma2"],
            alpha=params["alpha"],
            x_range=(-4.0, 4.0),
            t_range=(0, params["T"]),
            nx=60,
            nt=60,
        )
