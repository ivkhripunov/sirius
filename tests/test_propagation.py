import matplotlib.pyplot as plt
import numpy as np

from src.forces.earth_gravity import J2, R_earth, MU
from src.propagation import propagate
from src.kepler_elements import rv_to_kepler
from src.plot_kepler import plot_kepler

state_init = np.array([-11957371.5217699557542801,
                       385936.2176575958728790, - 1058529.5938599361106753, - 548.6566157170911993,
                       - 3515.9242334314676555,
                       4576.9840499053161693
                       ])

T = 2 * np.pi / 0.0004705130080798279

t_init = 0
t_final = 10 * T

res, times = propagate(state_init, t_init, t_final, step=10)

kepl_res = np.array([rv_to_kepler(rv[:3], rv[3:]) for rv in res])

print(kepl_res)

plot_kepler(kepl_res, times)

a, e, i, w_0, Omega_0, nu_0 = kepl_res[0]
p = a * (1 - e * e)
n = np.sqrt(MU / a) / a
dOmega_dt = -3 * n * R_earth ** 2 * J2 / (2 * p ** 2) * np.cos(i)
dw_dt = 3 * n * R_earth ** 2 * J2 / (4 * p ** 2) * (4 - 5 * np.sin(i) ** 2)

Omega_shift = np.mean(kepl_res[:, 4] - dOmega_dt * times)
w_shift = np.mean(kepl_res[:, 3] - dw_dt * times)

print(n)

fig_1, ax_1 = plt.subplots()
ax_1.plot(times, kepl_res[:, 4], )
ax_1.plot(times, Omega_shift + dOmega_dt * times)

fig_2, ax_2 = plt.subplots()
ax_2.plot(times, kepl_res[:, 3])
ax_2.plot(times, w_shift + dw_dt * times)

plt.show()
