import random
import numpy as np
import pandas as pd
from constants import id_map
from scipy.stats import norm
from datetime import timedelta, datetime
from scipy.optimize import minimize, LinearConstraint, minimize_scalar, differential_evolution
from Functions import data_for_optimising, exponential_moving_average, log_likelihood, round_dt,  string_to_datetime

filename = 'data/data2.csv'
df = pd.read_csv(filename)
time_index = np.argmax(df.columns == 'telemetry.time')
co2_index = np.argmax(df.columns == 'telemetry.co2')
id_index = np.argmax(df.columns == 'deviceId')
# So indices correspond to zone number, the 0'th element will simply be empty
device_list = [[] for _ in range(28)]

relevant_time = [datetime(year=9990, month=12, day=1) for _ in range(28)]
interval_smoothing_length=15

for row in df.values:
    co2 = row[co2_index]
    time = string_to_datetime(row[time_index])
    device_id = id_map[row[id_index]]
    device_list[device_id].append([time, co2])
    if time < relevant_time[device_id]:    # smaller time is earlier
        relevant_time[device_id] = time

for i, device in enumerate(device_list[1:]):
    relevant_time[i+1] = round_dt(relevant_time[i+1], minutes=interval_smoothing_length, up=False) + timedelta(minutes=interval_smoothing_length)
    data = device_list[1]
    new_data = []
    index = 0
    while index < len(data):
        temp = []
        print(data, relevant_time)
        print(len(data), index)
        while data[index][time_index] < relevant_time[i+1]:
            temp.append(data[index])
            index += 1
        new_data.append([relevant_time, exponential_moving_average(temp, tau=interval_smoothing_length)])
        relevant_time += timedelta(minutes=interval_smoothing_length)
    pass

# device_list = data_for_optimising(filename, interval_smoothing_length=15)

q_min, q_max = (0, 10)
m_min, m_max = (10, 20)
c_min, c_max = (350, 450)

bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))
v = 150
dT = 2
parameters = []
for i, device in enumerate(device_list):
    if device:
        device = np.array(device)
        c = np.array(device[:, 1], dtype=float)
        n = np.ones(len(c)) * 20

        minimised = minimize(
            log_likelihood,
            x0=np.array([5, 15, 400]),
            args=(c, n, v, dT, ),
            bounds=bounds
        )
        parameters.append(minimised.x)
    elif i != 0:
        print(f'No data from zone {i}')
print(np.array(parameters))
