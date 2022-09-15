import numpy as np
from Functions import basic_weighting


# Create random data
irrelevant_columns = [1, 12, 25, 26]
relevant_columns = list(set(range(27))-set(irrelevant_columns))
Ci = np.random.randint(400, 1000, 27)[relevant_columns]
Ci0 = np.random.randint(350, 450, 27)[relevant_columns]
N_total = np.random.randint(1, 800, 1)[0]

# Apply the simple formula
N = basic_weighting(Ci, Ci0, N_total, decimals=2)
print(N, N_total, np.sum(N))
# Include weighting of maximum capacity
M = np.random.randint(5, 20, 27)[relevant_columns]
N = basic_weighting(Ci, Ci0, N_total, decimals=2, include_max=True, M=M)
print(N)
print(np.sum(N))
print(np.sum(M)/(np.average(M)*len(M)))
