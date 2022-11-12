import numpy as np
from matplotlib import pyplot as plt
from Functions import data_for_optimising, optimise_occupancy, simulate_office

dT = 15 * 60
filename = 'data/data2.csv'
device_data_list = data_for_optimising(filename, interval_smoothing_length=dT / 60)
parameters = optimise_occupancy(device_data_list)
print(np.array(parameters))

simulate_office()
