import pandas as pd
import numpy as np
from Functions import hold_out, load_data, load_occupancy, load_lists, plot_estimates
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
    zone_id = 0
    for device, occupancy in zip(dd_list, N_list):  # iterate over each zone
        if len(device[0]) < 1 or len(occupancy[0]) < 1:  # skip empty zones
            continue
        zone_id += 1

        counter = 0
        C_train, N_train = [], []
        for period_time_co2, period_N in zip(device, occupancy):     # iterate over each period in zone

            period_co2 = list(np.array(period_time_co2)[:, 1])
            if counter != index:
                C_train = C_train + period_co2
                N_train = N_train + period_N
            counter += 1

        C_test, N_test = np.array(device[index])[:, 1], occupancy[index]
        C_test, N_test = np.array(C_test, dtype=float).reshape(-1, 1), np.array(N_test, dtype=int).reshape(-1, 1)

        C_train = np.array(C_train).reshape(-1, 1)
        N_train = np.array(N_train).reshape(-1, 1)

        reg_N = LinearRegression().fit(C_train, N_train)
        N_est = np.round(reg_N.predict(C_test), 0)

        reg_co2 = LinearRegression().fit(N_train, C_train)
        C_est = reg_co2.predict(N_test)

        plot_estimates(C=C_test, C_est=C_est, N=N_test, N_est=N_est, dt=dt, zone_id=zone_id)

# hold-out method:
# dd_list, N_list = hold_out(dates, V=V, dt=dt, plot=True, filename_parameters='testing', bounds=bounds)


