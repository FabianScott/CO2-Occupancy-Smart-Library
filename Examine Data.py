import numpy as np
import pandas as pd
from Functions import process_data
from Functions import update_data
from Functions import summary_stats_datetime_difference

df = pd.read_csv('data1.csv')
new_data = process_data(df, time_indexes=[0, 3], minutes=10000)
summary_stats_datetime_difference(new_data[:, 0], new_data[:, 3])

old_data = np.zeros((27, 2))
old_time = np.array(df['enqueuedTime'][:int(27*2)]).reshape((27, 2))

print(update_data(new_data, old_data, old_time))
