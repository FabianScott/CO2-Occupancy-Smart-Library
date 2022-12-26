import pandas as pd
import numpy as np
from Functions import hold_out, load_data, load_occupancy, load_lists
from sklearn.linear_model import LinearRegression

dates = ['2022_24_11', '2022_30_11',  '2022_09_12', '2022_14_12']   # '2022_07_12',

dt = 15*60  # in seconds
V = np.ones(28) * 300  # Has little impact
q_min, q_max = (0.01, 5)  # (.01/3600, 5/3600)
m_min, m_max = (0.01, 20)  # (7.675000000*(10**(-5)), 2*7.675000000*(10**(-5)))  # see equations in Maple
c_min, c_max = (300, 500)
bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

N_list, dd_list = load_lists(dates, dt)

for index, date in enumerate(dates):
    temp_dd, temp_N = [], []
    for device, occupancy in zip(dd_list, N_list):
        print(len(occupancy), len(device))
        temp_dd.append(device[:index] + device[index + 1:])
        temp_N.append(occupancy[:index] + occupancy[index + 1:])

# hold-out method:
# dd_list, N_list = hold_out(dates, V=V, dt=dt, plot=True, filename_parameters='testing', bounds=bounds)


