from Functions import load_data, optimise_occupancy, check_missing_data

filename = 'data/data3.csv'
dT = 15  # in minutes
device_data_list = load_data(filename, interval_smoothing_length=dT)

# check_missing_data(device_data_list, replace=True)
parameters = optimise_occupancy(device_data_list, method='Nelder-Mead', plot_result=True)
