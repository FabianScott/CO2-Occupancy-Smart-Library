import time
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from Functions import update_data, process_data, basic_weighting, \
    level_from_estimate, mass_balance, calculate_n_estimate, load_data

filename = 'data/co2_2022_07_12.csv'
end = datetime.now()
start = datetime.now() - timedelta(days=120)
dd_list = load_data(filename, start, end)

filename = 'data/co2_2022_09_12.csv'
end = datetime.now()
start = datetime.now() - timedelta(days=120)
dd_list_2 = load_data(filename, start, end, replace=False)

new_data = [[] for _ in dd_list]
old_data = [[] for _ in dd_list_2]

i = 0
for device_old, device_new in zip(dd_list, dd_list_2):
    if device_new:
        pass
    i += 1


new_data = None
old_data = np.zeros((27, 2))
old_time = np.array(new_data[:int(27 * 2), 0]).reshape((27, 2))
current_co2, current_time = update_data(new_data, old_data, old_time)
M, Ci0 = np.empty(27), np.empty(27)
M.fill(20)
Ci0.fill(300)
N = basic_weighting(current_co2[:, 0], Ci0, n_total=400, decimals=4, M=M, assume_unknown=True)
# print(f'Basis Weighting Result:')
# print(N)
# print(level_from_estimate(N, M))
Q = np.empty(27)
Q.fill(0.5)
V = np.empty(27)
V.fill(100)

N_mass = mass_balance(current_co2, Q, V, current_time=current_time, n_total=200, fill_neighbours=False)
print('Weighted Mass Balance Result:')
print(N_mass)

print(level_from_estimate(N_mass, M))
