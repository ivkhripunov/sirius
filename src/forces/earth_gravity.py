import numpy as np

MU = 3.986004418e14
R_earth = 6378137

J2 = 0.0010826269
J3 = -0.0000025327
J4 = -0.0000016196


def central_grav(r):
    r_norm = np.linalg.norm(r)

    return - MU / r_norm ** 3 * r


def J2_grav(r):
    r_norm = np.linalg.norm(r)

    j2_const = - 3 * J2 * MU * R_earth ** 2 / (2 * r_norm ** 5)
    j2_grav = j2_const * r * (1 - 5 * (r[2] / r_norm) ** 2)
    j2_grav[2] += 2 * j2_const * r[2]

    return j2_grav


def J3_grav(r):
    r_norm = np.linalg.norm(r)

    j3_const = -5 * J3 * MU * R_earth ** 3 / (2 * r_norm ** 7)
    return np.array([
        j3_const * r[0] * (3 * r[2] - 7 * r[2] ** 3 / r_norm ** 2),
        j3_const * r[1] * (3 * r[2] - 7 * r[2] ** 3 / r_norm ** 2),
        j3_const * (6 * r[2] ** 2 - 7 * r[2] ** 4 / r_norm ** 2 - 3 * r_norm ** 2 / 5),
    ])


def J4_grav(r):
    r_norm = np.linalg.norm(r)

    j4_const = 15 * J4 * MU * R_earth ** 4 / (8 * r_norm ** 7)
    j4_grav = j4_const * r * (
            1 - 14 * r[2] ** 2 / r_norm ** 2 + 21 * r[2] ** 4 / r_norm ** 4)
    j4_grav[2] += j4_const * r[2] * (4 - 38 * r[2] ** 2 / (3 * r_norm ** 2))

    return j4_grav
