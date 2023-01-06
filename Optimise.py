import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Functions import simple_models_hold_out, hold_out, adjacent_co2,\
    load_data, load_occupancy, load_lists, plot_estimates

dates = ['2022_24_11', '2022_30_11', '2022_07_12',  '2022_09_12', '2022_14_12']
# dates = ['2022_24_11', '2022_30_11']

dt = 15*60  # in seconds
V = np.ones(28) * 300  # Has little impact
q_min, q_max = (0.01, 5)  # (.01/3600, 5/3600)
m_min, m_max = (0.01, 20)  # (7.675000000*(10**(-5)), 2*7.675000000*(10**(-5)))  # see equations in Maple
c_min, c_max = (300, 1000)
bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

# print(simple_models_hold_out(dates, dt=15 * 60, method='l', plot_scatter=True))
# hold-out method:
dd_list, N_list = hold_out(dates, V=V, dt=dt, plot=True, filename_parameters='testing')

