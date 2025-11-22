import numpy as np

MU = 3.986004418e14


def rv_to_kepler(r, v):
    r_norm = np.linalg.norm(r)

    L = np.cross(r, v)
    L_norm = np.linalg.norm(L)
    n_orb = L / L_norm

    sin_i = np.sqrt(n_orb[0] ** 2 + n_orb[1] ** 2)
    cos_i = n_orb[2]
    i = np.atan2(sin_i, cos_i)

    z = np.array([0, 0, 1])
    N = np.cross(z, L)

    Omega = np.atan2(N[1], N[0]) if i != 0 else 0

    v2 = np.dot(v, v)
    e = ((v2 - MU / r_norm) * r - np.dot(r, v) * v) / MU
    e_norm = np.linalg.norm(e)

    e_1 = e / e_norm if e_norm != 0 else N

    d_1 = N
    d_2 = np.cross(n_orb, N)

    cos_w = np.dot(e_1, d_1)
    sin_w = np.dot(e_1, d_2)
    w = np.atan2(sin_w, cos_w)

    a = - MU / 2 / (v2 / 2 - MU / r_norm)

    e_2 = np.cross(n_orb, e_1)

    cos_nu = np.dot(e_1, r)
    sin_nu = np.dot(e_2, r)
    nu = np.atan2(sin_nu, cos_nu)

    return np.array([a, e_norm, i, w, Omega, nu])
