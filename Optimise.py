import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from Functions import simple_models_hold_out, hold_out, adjacent_co2, residual_analysis, load_davide, \
    load_data, load_occupancy, load_lists, plot_estimates, optimise_occupancy, N_estimate, C_estimate, \
    return_average_parameters, sensitivity_plots, matrix_to_latex

dates = ['2022_24_11', '2022_30_11', '2022_07_12', '2022_09_12', '2022_14_12', '2023_18_01',
         '2023_19_01']  # , '2023_25_01']
# dates = dates[:2]

# hold-out method:
error_file = 'parameters/e.txt'
filename_parameters = 'testing'
filepath_plots = 'documents/plots/'
load_davide(optimise=True)
# Produce final result tables!!
dd_list, N_list, E_list, sensitivity_list = hold_out(dates, plot=True, filename_parameters=filename_parameters,
                                                     summary_errors=error_file, no_steps=20)
E_list_reg = simple_models_hold_out(dates, dt=15 * 60, method='l', plot=True, plot_scatter=0)
table_mean, table_std = residual_analysis(dd_list, N_list, E_list, E_list_reg, plot=False)
# Decide on plot for sensitivity, probably take average across periods?
avg_params, std_params = return_average_parameters(dates, filename_parameters)

print(matrix_to_latex(table_mean))
print(matrix_to_latex(table_std))
print(matrix_to_latex(avg_params, d=6))

