import numpy as np

from src.forces.earth_gravity import central_grav, J2_grav, J3_grav, J4_grav
from src.math.runge_kutta import rk


def ballistics(state, t):
    r = state[:3]
    acc = central_grav(r) + J2_grav(r)

    return np.array([*state[3:], *acc])


def propagate(state_init, t_init, t_final, step=10):
    return rk(ballistics, state_init, t_init, t_final, step)
