import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Functions import simple_models_hold_out, hold_out, adjacent_co2,\
    load_data, load_occupancy, load_lists, plot_estimates

dates = ['2022_24_11', '2022_30_11', '2022_07_12',  '2022_09_12', '2022_14_12']
# dates = ['2022_24_11', '2022_30_11']

dt = 15*60  # in seconds
V = np.ones(28) * 300  # Has little impact

# print(simple_models_hold_out(dates, dt=15 * 60, method='l', plot=True, plot_scatter=True))
# hold-out method:
dd_list, N_list = hold_out(dates, V=V, dt=dt, plot=True, filename_parameters='testing')

