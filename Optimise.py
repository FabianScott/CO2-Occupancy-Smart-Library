import pandas as pd
import numpy as np
from Functions import load_data, optimise_occupancy, load_occupancy, load_and_use_parameters

name_n1 = 'data/N_24_11new.csv'
name_n2 = 'data/N_30_11new.csv'
name_c1 = 'data/co2_24_11.csv'
name_c2 = 'data/co2_30_11.csv'
name_param = 'parameters/testing.csv'

dt = 15  # in minutes
V = np.ones(28) * 300   # Has little impact
q_min, q_max = (0.01, 5)
m_min, m_max = (1/360000, 5)
c_min, c_max = (300, 500)
bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

N1, start1, end1 = load_occupancy(name_n1)
device_data_list1 = load_data(name_c1, start1, end1, replace=True, interval_smoothing_length=dt, no_points=len(N1[-1]),
                              smoothing_type='Kalman')

parameters = optimise_occupancy(device_data_list1,
                                method='Nelder-Mead', N=N1, V=V, plot_result=False, verbosity=False,
                                filename_parameters=name_param, bounds=bounds)

N2, start2, end2 = load_occupancy(name_n2)

device_data_list2 = load_data(name_c2, start2, end2, replace=True, interval_smoothing_length=dt, no_points=len(N2[-1]),
                              smoothing_type='Kalman')
load_and_use_parameters(name_param, device_data_list2, N2, V, dt * 60)

