import numpy as np
from numpy import sin, cos

MU = 3.986004418e14


def rv_to_kepler(r, v):
    r_norm = np.linalg.norm(r)

    L = np.cross(r, v)
    L_norm = np.linalg.norm(L)
    n_orb = L / L_norm

    sin_i = np.sqrt(n_orb[0] ** 2 + n_orb[1] ** 2)
    cos_i = n_orb[2]
    i = np.arctan2(sin_i, cos_i)

    z = np.array([0, 0, 1])
    N = np.cross(z, L)

    Omega = np.arctan2(N[1], N[0]) if i != 0 else 0
    if Omega < 0:
        Omega += 2 * np.pi

    v2 = np.dot(v, v)
    e = ((v2 - MU / r_norm) * r - np.dot(r, v) * v) / MU
    e_norm = np.linalg.norm(e)

    e_1 = e / e_norm if e_norm != 0 else N

    d_1 = N
    d_2 = np.cross(n_orb, N)

    cos_w = np.dot(e_1, d_1)
    sin_w = np.dot(e_1, d_2)
    w = np.arctan2(sin_w, cos_w)

    a = - MU / 2 / (v2 / 2 - MU / r_norm)

    e_2 = np.cross(n_orb, e_1)

    cos_nu = np.dot(e_1, r)
    sin_nu = np.dot(e_2, r)
    nu = np.arctan2(sin_nu, cos_nu)

    return np.array([a, e_norm, i, w, Omega, nu])


def kepler_to_rv(kepler):
    a, e, i, w, Omega, nu = kepler

    p = a * (1 - e * e)

    sin_nu = sin(nu)
    cos_nu = cos(nu)

    r_orb = np.array([
        p * cos_nu / (1 + e * cos_nu),
        p * sin_nu / (1 + e * cos_nu),
        0
    ])

    tmp = np.sqrt(MU / p)

    v_orb = np.array([
        - tmp * sin_nu,
        tmp * (e + cos_nu),
        0
    ])

    rot_mat = np.zeros((3, 3))

    rot_mat[0, 0] = cos(Omega) * cos(w) - np.sin(Omega) * np.sin(w) * cos(i)
    rot_mat[0, 1] = -cos(Omega) * sin(w) - sin(Omega) * cos(w) * cos(i)
    rot_mat[0, 2] = sin(Omega) * sin(i)

    rot_mat[1, 0] = sin(Omega) * cos(w) + cos(Omega) * sin(w) * cos(i)
    rot_mat[1, 1] = -sin(Omega) * sin(w) + cos(Omega) * cos(w) * cos(i)
    rot_mat[1, 2] = -cos(Omega) * sin(i)

    rot_mat[2, 0] = sin(w) * sin(i)
    rot_mat[2, 1] = cos(w) * sin(i)
    rot_mat[2, 2] = cos(i)

    r = np.dot(rot_mat, r_orb)
    v = np.dot(rot_mat, v_orb)

    return r, v
