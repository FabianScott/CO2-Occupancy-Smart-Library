import pandas as pd
import numpy as np
from Functions import process_data
from Functions import update_data

df = pd.read_csv('data.csv')
new_data = process_data(df, 300)
old = np.zeros((27, 2))
print(old, update_data(new_data, old))
