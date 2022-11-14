import numpy as np
from Functions import data_for_optimising, optimise_occupancy, simulate_office

dT = 15 * 60
filename = 'data/data2.csv'
device_data_list = data_for_optimising(filename, interval_smoothing_length=dT / 60)
# parameters = optimise_occupancy(device_data_list, N=[np.ones(2)*2 for _ in range(28)], method='Nelder-Mead', plot_result=True)
# print(np.array(parameters))

simulate_office()
