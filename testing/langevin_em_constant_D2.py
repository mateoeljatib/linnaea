import numpy as np


def expm(M, n_taylor=24):
    """exp(M) por scaling-and-squaring + Taylor (robusto en bloques de Jordan)."""
    M = np.asarray(M, dtype=float)
    d = M.shape[0]
    nrm = np.max(np.sum(np.abs(M), axis=1))
    if nrm == 0.0:
        return np.eye(d)
    s = max(0, int(np.ceil(np.log2(nrm))) + 1)
    Ms = M / (2.0 ** s)
    term, out = np.eye(d), np.eye(d)
    for k in range(1, n_taylor + 1):
        term = term @ Ms / k
        out = out + term
        if np.max(np.abs(term)) < 1e-18:
            break
    for _ in range(s):
        out = out @ out
    return out


def lyapunov(A, Q):
    """Resuelve A S + S A^T = -Q (covarianza estacionaria) via Kronecker."""
    n = A.shape[0]
    K = np.kron(np.eye(n), A) + np.kron(A, np.eye(n))
    return np.linalg.solve(K, -Q.reshape(-1)).reshape(n, n)


def van_loan(A, B, dt):
    """Discretizacion exacta: Phi = expm(A dt) y Sigma = covarianza del ruido por paso."""
    n = A.shape[0]
    M = np.zeros((2 * n, 2 * n))
    M[:n, :n] = -A
    M[:n, n:] = B @ B.T
    M[n:, n:] = A.T
    F = expm(M * dt)
    Phi = F[n:, n:].T
    Sigma = Phi @ F[:n, n:]
    return Phi, 0.5 * (Sigma + Sigma.T)


def _chol(S, jitter=1e-12):
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        return np.linalg.cholesky(S + jitter * np.eye(S.shape[0]))


def simulate(delta_em, *, tau_v=1.0, D=1.0, dt=5e-3, T=40.0,
             npart=3000, seed=0, dtype=np.float64):
    """
    Evoluciona npart particulas y devuelve la matriz de velocidades

        v.shape == (npart, n_steps)      # fila = particula, columna = tiempo

    delta_em <= 0  ->  OU puro (limite blanco, Markoviano).
    Arranca en el estado estacionario (sin transitorio). dtype=np.float32
    reduce a la mitad la memoria del almacenamiento (el integrador opera en f64).
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))

    if delta_em <= 0.0:                                   # OU puro (referencia Markoviana)
        a = np.exp(-dt / tau_v)
        var = D * tau_v
        b = np.sqrt(var * (1.0 - a * a))
        v = np.empty((n_steps, npart), dtype=dtype)
        x = rng.standard_normal(npart) * np.sqrt(var)
        for k in range(n_steps):
            x = a * x + b * rng.standard_normal(npart)
            v[k] = x
        return v.T

    tau_c = delta_em
    A = np.array([[-1.0 / tau_v, 1.0], [0.0, -1.0 / tau_c]])
    B = np.array([[0.0], [np.sqrt(2.0 * D) / tau_c]])

    Phi, Sigma = van_loan(A, B, dt)
    L = _chol(Sigma)
    X = _chol(lyapunov(A, B @ B.T)) @ rng.standard_normal((2, npart))   # CI estacionaria

    v = np.empty((n_steps, npart), dtype=dtype)
    for k in range(n_steps):
        X = Phi @ X + L @ rng.standard_normal((2, npart))
        v[k] = X[0]
    return v.T          # vista (npart, n_steps), sin copia


def plot_trayectorias(v, dt, n=6, seed=None):
    """Grafica n trayectorias al azar de la matriz (npart, n_steps). Devuelve (fig, ax)."""
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    idx = rng.choice(v.shape[0], size=min(n, v.shape[0]), replace=False)
    t = np.arange(v.shape[1]) * dt
    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i in idx:
        ax.plot(t, v[i], lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("v(t)")
    fig.tight_layout()
    plt.show()
    return fig, ax


# --------------------------------------------------------------------------
if __name__ == "__main__":
    params = dict(tau_v=1.0, D=1.0, dt=5e-3, T=40.0, npart=10000, seed=42)
    deltas = [0.1, 0.5, 2.0, 5.0, 10.0]          # una matriz por Delta_EM
    guardar = True

    matrices = {}
    for de in deltas:
        v = simulate(de, **params)
        matrices[de] = v
        if guardar:
            dem_str = f"{de}".replace('.', '-')
            np.save(f"magnitudes/tray_vel_DeltaEM_{dem_str}.npy", v)
        print(f"Delta_EM={de:<4}  shape={v.shape}  {v.dtype}  "
              f"{v.nbytes / 2**20:6.1f} MiB" + ("  -> guardado" if guardar else ""))
        # plot_trayectorias(v, dt=5e-3)
