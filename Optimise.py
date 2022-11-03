import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint, minimize_scalar, differential_evolution
from Functions import data_for_optimising, exponential_moving_average, log_likelihood

filename = 'data/data2.csv'
device_list = data_for_optimising(filename)

q_min, q_max = (0, 10)
m_min, m_max = (10, 20)
c_min, c_max = (350, 450)

bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))
v = 150
dT = 2
parameters = []
for i, device in enumerate(device_list):
    if device:
        device = np.array(device)
        c = np.array(device[:, 1], dtype=float)
        n = np.ones(len(c)) * 20

        minimised = minimize(
            log_likelihood,
            x0=np.array([5, 15, 400]),
            args=(c, n, v, dT, ),
            bounds=bounds
        )
        parameters.append(minimised.x)
    elif i != 0:
        print(f'No data from zone {i}')
print(np.array(parameters))
