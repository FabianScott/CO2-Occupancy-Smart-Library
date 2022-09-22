import numpy as np
from Functions import optimise_mass_balance

N_total = np.random.randint(200, 600, 10)
C = np.random.randint(400, 600, (10, 27))
V = np.random.randint(50, 200, 27)
Q = 1.25*V

print(optimise_mass_balance(C, N_total, Q, V, m_range=(5, 15), precision=1))
