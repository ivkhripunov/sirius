import numpy as np


def rk_step(f, s, t, h):
    k_1 = f(s, t)
    k_2 = f(s + h * k_1 / 2, t + h / 2)
    k_3 = f(s + h * k_2 / 2, t + h / 2)
    k_4 = f(s + h * k_3, t + h)

    return s + h / 6 * (k_1 + 2 * (k_2 + k_3) + k_4), t + h


def rk(f, s_init, t_init, t_final, max_step):
    n = int((t_final - t_init) / max_step) + 1

    times = np.zeros(n)
    times[0] = t_init

    sol = np.zeros((n, np.size(s_init)))
    sol[0] = s_init

    for i in range(1, n - 1):
        sol[i], times[i] = rk_step(f, sol[i - 1], times[i - 1], max_step)

    sol[-1], times[-1] = rk_step(f, sol[-2], times[-2], t_final - times[-2])

    return sol, times
