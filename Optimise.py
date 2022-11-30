import pandas as pd
from Functions import load_data, optimise_occupancy, load_occupancy, check_missing_data
filename1 = 'data/N_24_11.csv'

N = load_occupancy(filename1)

filename2 = 'data/co2_24_11.csv'
dT = 15  # in minutes
device_data_list = load_data(filename2, replace=True, interval_smoothing_length=dT, no_points=len(N[-1]))

q_min, q_max = (0.01, 0.02)
m_min, m_max = (10, 20)
c_min, c_max = (350, 450)

bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

parameters = optimise_occupancy(device_data_list, method='Nelder-Mead', N=N, plot_result=True, verbosity=False)


