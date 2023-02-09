import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from constants import bounds
from Functions import simple_models_hold_out, hold_out, adjacent_co2, residual_analysis, load_davide, \
    load_data, load_occupancy, load_lists, plot_estimates, optimise_occupancy, N_estimate, C_estimate, \
    return_average_parameters, sensitivity_plots, matrix_to_latex, str_to_dt, error_fraction

dates = ['2022_24_11', '2022_30_11', '2022_07_12', '2022_09_12', '2022_14_12', '2023_18_01',
         '2023_19_01']  # , '2023_25_01']
# dates = dates[:2]

# hold-out method:
error_file = 'parameters/e.txt'
filename_parameters = 'testing'
filepath_plots = 'documents/plots/'


load_davide()

# dd_list, N_list, E_list, sensitivity_list, sensitivity_list_c = hold_out(dates, plot=False, filename_parameters=filename_parameters,
#                                                      summary_errors=error_file, no_steps=100, optimise_N=False)
# E_list_reg = simple_models_hold_out(dates, dt=15 * 60, method='l', plot=False, plot_scatter=False)
# table_mean, table_std, table_detect_noneg = residual_analysis(dd_list, N_list, E_list, E_list_reg, plot=False)
# # Decide on plot for sensitivity, probably take average across periods?
# avg_params, std_params = return_average_parameters(dates, filename_parameters)
# sensitivity_plots(sensitivity_list, filepath_plots=filepath_plots, avg_params=avg_params, avg_errors=table_mean[:, 1])
# sensitivity_plots(sensitivity_list_c, filepath_plots=filepath_plots, avg_params=avg_params, avg_errors=table_mean[:, 3], post_fix='CO2')
#
#
# zone_names = [21, 22, 23, 25, 26, 27]
# print(matrix_to_latex(table_mean))
# print(matrix_to_latex(np.hstack((np.array(zone_names).reshape(-1, 1), table_detect_noneg))))
# print(matrix_to_latex(table_std))
# print(matrix_to_latex(np.hstack((np.array(zone_names).reshape(-1, 1), avg_params)), d=2))
# print(matrix_to_latex(np.hstack((np.array(zone_names).reshape(-1, 1), avg_params / std_params)), d=2))
