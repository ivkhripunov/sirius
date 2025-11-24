from src.kepler_elements import rv_to_kepler, kepler_to_rv

import numpy as np

p = 11067790
e = 0.83285
a = p / (1 - e * e)
i = np.deg2rad(87.87)
Omega = np.deg2rad(227.89)
w = np.deg2rad(53.38)
nu = np.deg2rad(92.335)

kepler = [a, e, i, w, Omega, nu]

r, v = kepler_to_rv(kepler)

print(kepler)

print(rv_to_kepler(r, v))
