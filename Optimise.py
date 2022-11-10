import random
import numpy as np
import pandas as pd
from constants import id_map
from scipy.stats import norm
from datetime import timedelta, datetime
from scipy.optimize import minimize, LinearConstraint, minimize_scalar, differential_evolution
from Functions import data_for_optimising, exponential_moving_average, log_likelihood, \
    round_dt, string_to_datetime, calculate_co2_estimate


filename = 'data/data2.csv'
device_data_list = data_for_optimising(filename, interval_smoothing_length=15)

q_min, q_max = (0, 10)
m_min, m_max = (10, 20)
c_min, c_max = (350, 450)

bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))
v = 150
verbosity = True
dT = 2
parameters = []
np.random.seed(42)
for i, device in enumerate(device_data_list):
    if device:
        device = np.array(device)
        c = np.array(device[:, 1], dtype=float)
        n = np.random.normal(loc=100, scale=3, size=len(c))

        minimised = minimize(
            log_likelihood,
            x0=np.array([q_max-q_min, m_max-m_min, c_max-c_min]),
            args=(c, n, v, dT, verbosity,),
            bounds=bounds
        )
        parameters.append(minimised.x)
    elif i != 0:
        print(f'No data from zone {i}')
print(np.array(parameters))

print(calculate_co2_estimate(parameters[0], c, n, v, dT) - c[1:])

