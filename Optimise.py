import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Functions import simple_models_hold_out, hold_out, adjacent_co2, \
    load_data, load_occupancy, load_lists, plot_estimates, optimise_occupancy

dates = ['2022_24_11', '2022_30_11', '2022_07_12', '2022_09_12', '2022_14_12', '2023_18_01']
# dates = ['2022_24_11', '2022_30_11']

# print(simple_models_hold_out(dates, dt=15 * 60, method='l', plot=True, plot_scatter=True))
# hold-out method:
error_file = 'parameters/e.txt'
dd_list, N_list, E_list = hold_out(dates[:2], plot=1, filename_parameters='testing', optimise_N=0,
                                   error_file=error_file)


