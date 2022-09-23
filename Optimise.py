import numpy as np
from Functions import optimise_mass_balance_m
from Functions import optimise_mass_balance_Q

N_total = np.random.randint(200, 600, 10)
C = np.random.randint(400, 600, (10, 27))
V = np.random.randint(50, 200, 27)
Q = np.random.random(27)*0.1
print(Q)

print(optimise_mass_balance_m(C, N_total, Q, V, m_range=(5, 15), precision=1))
print(optimise_mass_balance_Q(C, N_total, Q, V))
