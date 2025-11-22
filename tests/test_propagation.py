import numpy as np

from src.propagation import propagate
from src.kepler_elements import rv_to_kepler
from src.plot_kepler import plot_kepler

state_init = np.array([-11957371.5217699557542801,
                       385936.2176575958728790, - 1058529.5938599361106753, - 548.6566157170911993,
                       - 3515.9242334314676555,
                       4576.9840499053161693
                       ])

t_init = 0
t_final = 86400 * 3

res, times = propagate(state_init, t_init, t_final, step=10)

kepl_res = np.array([rv_to_kepler(rv[:3], rv[3:]) for rv in res])

print(kepl_res)

plot_kepler(kepl_res, times)


