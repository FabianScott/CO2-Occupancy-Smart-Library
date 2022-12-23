import pandas as pd
import numpy as np
from Functions import hold_out, load_data, load_occupancy
from sklearn.linear_model import LinearRegression

dates = ['2022_24_11', '2022_30_11',  '2022_09_12', '2022_14_12']   # '2022_07_12',

dt = 15*60  # in seconds
V = np.ones(28) * 300  # Has little impact
q_min, q_max = (0.01, 5)  # (.01/3600, 5/3600)
m_min, m_max = (0.01, 20)  # (7.675000000*(10**(-5)), 2*7.675000000*(10**(-5)))  # see equations in Maple
c_min, c_max = (300, 500)
bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

N_list, dd_list = [[] for _ in range(28)], [[] for _ in range(28)]

# Initially load all the data into the dd_list and N_list of lists
for date in dates:
    temp_name_c = 'data/co2_' + date + '.csv'
    temp_name_n = 'data/N_' + date + '.csv'

    N, start, end = load_occupancy(temp_name_n)
    device_data_list = load_data(temp_name_c, start, end, replace=True, interval=dt,
                                 no_points=len(N[-1]), smoothing_type='exponential')
    for i in range(28):
        N_list[i] = N_list[i] + list(N[i])
        dd_list[i] = dd_list[i] + device_data_list[i]

for N, co2 in zip(N_list, dd_list):
    reg = LinearRegression().fit(N, co2)

# hold-out method:
dd_list, N_list = hold_out(dates, V=V, dt=dt, plot=True, filename_parameters='testing', bounds=bounds)


