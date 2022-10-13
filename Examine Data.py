import numpy as np
import pandas as pd
from Functions import update_data
from Functions import process_data
from Functions import basic_weighting
from Functions import level_from_estimate
from Functions import mass_balance
from Functions import summary_stats_datetime_difference

df = pd.read_csv('data1.csv')
new_data = process_data(df, time_indexes=[0, 3], minutes=10000000)
old_data = np.zeros((27, 2))
old_time = np.array(df['enqueuedTime'][:int(27 * 2)]).reshape((27, 2))
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
N_mass = mass_balance(current_co2, Q, V, n_total=200, fill_neighbours=False)
print('Weighted Mass Balance Result:')
print(N_mass)
