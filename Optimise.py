import random
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, minimize_scalar, differential_evolution
from Functions import process_data, update_data, get_replacement_and_derivative

random.seed(42)
V = 150
N_true = 500

df = pd.read_csv('data1.csv')
new_data = process_data(df, time_indexes=[0, 3], minutes=10000000)
old_data = np.zeros((27, 2))
old_time = np.array(new_data[:int(27 * 2), 0]).reshape((27, 2))
C, current_time = update_data(new_data, old_data, old_time)


def objective_function_scalar(x, n_true):
    Q, m, C_out = x
    Cr, dC = get_replacement_and_derivative(C, C_out)
    n = sum((Q * (C[:, 0] - Cr) + V * dC) / m)
    return np.abs(n_true - n)


def objective_function_vector(x, n_true):
    C_out, m = x[:2]
    Q = np.array(x[2:])

    n = sum((Q * (C[:, :, 0] - Cr) + V * dC) / m)
    return np.abs(n_true - n)


q_min, q_max = (0, 10)
m_min, m_max = (10, 20)
c_min, c_max = (350, 450)
x0 = np.array((random.randint(q_min, q_max), random.randint(m_min, m_max), random.randint(c_min, c_max)))
print(differential_evolution(
    objective_function_scalar,
    x0=x0,
    args=(N_true, ),
    bounds=[(q_min, q_max), (m_min, m_max), (c_min, c_max)]
))
