import pandas as pd
import numpy as np
from Functions import load_data, optimise_occupancy, load_occupancy, load_and_use_parameters

dates = ['24_11', '30_11'] # , '09_12', '14_12']

dt = 15  # in minutes
V = np.ones(28) * 300  # Has little impact
q_min, q_max = (.01/3600, 5/3600)
m_min, m_max = (7.675000000*(10**(-5)), 2*7.675000000*(10**(-5)))  # see equations in Maple
c_min, c_max = (300, 500)
bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

# testing hold-out method:
for date in dates:
    N_list, dd_list = [[] for _ in range(28)], [[] for _ in range(28)]
    name_param = f'parameters/testing_{date}.csv'

    for d in dates:
        if d != date:
            temp_name_c = 'data/co2_' + d + '.csv'
            temp_name_n = 'data/N_' + d + '.csv'

            N, start, end = load_occupancy(temp_name_n)
            device_data_list = load_data(temp_name_c, start, end, replace=True, interval=dt,
                                         no_points=len(N[-1]), smoothing_type='Kalman')
            for i in range(28):
                N_list[i] = N_list[i] + list(N[i])
                dd_list[i] = dd_list[i] + device_data_list[i]
    # print(dd_list, '\n', N_list)

    parameters = optimise_occupancy(dd_list, method='Nelder-Mead', N=N_list, V=V, plot_result=True,
                                    verbosity=False, filename_parameters=name_param, bounds=bounds)

reg = False
if reg:
    name_n1 = 'data/N_24_11.csv'
    name_n2 = 'data/N_30_11.csv'
    name_c1 = 'data/co2_24_11.csv'
    name_c2 = 'data/co2_30_11.csv'
    name_param = 'parameters/testing.csv'

    N1, start1, end1 = load_occupancy(name_n1)
    device_data_list1 = load_data(name_c1, start1, end1, replace=True, interval=dt, no_points=len(N1[-1]),
                                  smoothing_type='Kalman')

    parameters = optimise_occupancy(device_data_list1,
                                    method='Nelder-Mead', N=N1, V=V, plot_result=False, verbosity=False,
                                    filename_parameters=name_param, bounds=bounds)

    N2, start2, end2 = load_occupancy(name_n2)

    device_data_list2 = load_data(name_c2, start2, end2, replace=True, interval=dt, no_points=len(N2[-1]),
                                  smoothing_type='Kalman')
    load_and_use_parameters(name_param, device_data_list2, N2, V, dt * 60)
