import random
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize, LinearConstraint, minimize_scalar, differential_evolution
from Functions import process_data, exponential_moving_average

random.seed(42)
V = 150
N_true = 500

df = pd.read_csv('data/zone23.csv')
df['device_id'] = ['DA00110034' for _ in df.values[:, 0]]
new_data = process_data(df, time_indexes=[0, 1], id_index=3, minutes=10000000)


def objective_normal(x, C, N, V, dt, uncertainty=50, percent=0.03):
    """
    Calculates the log log_likelihood of the current parameters, by
    finding the pdf of the normal distribution with mean as the
    measured CO2 level and standard deviation from the specifications.
    Parameters being optimised are:
        m       CO2 per person
        C_out   CO2 concentration outdoors
        Q       Airflow rate with outdoors (and neighbouring zones, to be implemented)
    :param x:               parameters being optimised
    :param C:               measured CO2 levels
    :param N:               number of people
    :param V:               volume of zone
    :param dt:              time step
    :param percent:         percent uncertainty of sensors
    :param uncertainty:     minimum uncertainty of sensors
    :return:
    """
    m, C_out, Q = x
    uncertainty, percent = uncertainty / 2, percent / 2  # it is the 95 % confidence, therefor 2 sd's
    Ci, N = C[:-1], N[1:]  # Remove first N, as there is no previous CO2

    C_est = (Q * dt * C_out + V * Ci - m * N * dt) / (Q * dt + V)
    # compare to C[1:] as there is no first estimate
    sd = np.array([max(uncertainty, el * percent) for el in C[1:]])
    log_likelihood = sum(np.log(norm.pdf(C_est, loc=C[1:], scale=sd)))

    return log_likelihood


q_min, q_max = (0, 10)
m_min, m_max = (10, 20)
c_min, c_max = (350, 450)

print(exponential_moving_average(new_data[:20, 2], new_data[:20, 0]))
