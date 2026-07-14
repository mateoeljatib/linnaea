import numpy as np

x_matrix_periodic = np.load("./magnitudes/ghost_sim/x_matrix_HIT512_HPMR_st05.npy")

def non_periodic_trajectories(positions):
    """Función prestada por Facu. postition tiene que tener shape (3, npart, times)"""
    Xs = np.copy(positions).astype(np.float32)
    real_steps = np.copy(positions).astype(np.float32)
    n = Xs.shape[2]
    real_steps = 1 * ( 
        (positions[:, :, 1:] % (2 * np.pi) - positions[:, :, :-1] % (2 * np.pi)) > np.pi
    ) - 1 * ( 
        (positions[:, :, :-1] % (2 * np.pi) - positions[:, :, 1:] % (2 * np.pi)) > np.pi
    )   
    real_steps = ( 
        positions[:, :, 1:] % (2 * np.pi) - positions[:, :, :-1] % (2 * np.pi)
    ) - 2 * np.pi * real_steps
    for t in range(n - 1): 
        Xs[:, :, t + 1] = Xs[:, :, t] + real_steps[:, :, t]
    return Xs

x_non_periodic = non_periodic_trajectories(x_matrix_periodic.transpose(2, 1, 0))
x = x_non_periodic - x_non_periodic[:, :, [0]]
d2 = np.sum(x**2, axis=0)


#np.save("./magnitudes/ghost_sim/x_matrix_non_periodic_HIT512_HPMR_st05.npy", x_non_periodic)
np.save("./magnitudes/ghost_sim/d2.npy", d2)
