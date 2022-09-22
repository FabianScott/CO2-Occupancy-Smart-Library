import numpy as np
from collections import defaultdict
from Functions import mass_balance


# Create scalars
N_total = np.random.randint(1, 800, 1)[0]
time_step = 5
m = np.random.randint(13, 20, 1)[0]
C_out = np.random.randint(350, 500, 1)[0]
alpha = 0.05
# Create vectors
Ci = np.random.randint(700, 1000, (2, 27))
Ci0 = np.random.randint(350, 450, (2, 27))
V = np.random.randint(50, 200, 27)
Q = 1.25*V

N_estimate = mass_balance(Ci, Q, V, N_total, alpha=1)
print(N_estimate, np.sum(N_estimate), N_total)


