import numpy as np
import pandas as pd
from copy import copy, deepcopy
from scipy.ndimage import uniform_filter1d
from sigfig import round
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.stats import norm, wilcoxon, probplot, shapiro
from scipy.optimize import differential_evolution
from sklearn.linear_model import LinearRegression
from constants import V, bounds, id_map, ppm_factor
from statsmodels.stats.contingency_tables import mcnemar


def hold_out(dates, m=15, dt=15 * 60, plot=False, use_adjacent=True, filename_parameters='testing', optimise_N=False,
             summary_errors=None, no_steps=0, smooth_co2=3):
    """
    Given a list of dates in the format yyyy_dd_mm, same as filenames
    for data, use the hold out method on each period of data, using it
    as a test period for the parameters optimised on the rest.
    :param dates:
    :param V:
    :param plot:
    :param filename_parameters:
    :param dt:
    :return:
    """

    N_list, dd_list = load_lists(dates, dt)
    # dimension 0 -> zone, dimension 1 -> (Q_adj, Q_out, C_out, m), dimension 2 -> (error_n, x_val)
    sensitivity_list_N = [[[[] for _ in dates] for _ in range(4)] for _ in dd_list]  # periods simply appended in series
    sensitivity_list_C = [[[[] for _ in dates] for _ in range(4)] for _ in dd_list]
    # dimension 0 -> zone, dimension 1 -> period, dimension 2 -> CO2,N error, dimension 3 -> proportion,avg.,list
    E_list = [[] for _ in N_list]
    E_summary_list = [[] for _ in N_list]
    adj_list = adjacent_co2(dd_list, use_adjacent=use_adjacent)
    # Use the index in the date list to hold out each period once
    for date_index, date in enumerate(dates):

        temp_dd, temp_N = [], []
        for device, occupancy in zip(dd_list, N_list):
            temp_dd.append(device[:date_index] + device[date_index + 1:])
            temp_N.append(occupancy[:date_index] + occupancy[date_index + 1:])

        filepath_parameters = f'parameters/{filename_parameters}_{date}.csv'
        parameters = optimise_occupancy(temp_dd, N_list=temp_N, optimise_N=optimise_N, smooth_co2=smooth_co2,
                                        filename_parameters=filepath_parameters, bounds=bounds)
        print(f'Coefficients for CO2:\n'
              f'{(1 - dt * sum(parameters[0, :2]))} * C_(i-1) +\n'
              f'{parameters[0, 0] * dt} * C_adj +\n'
              f'{parameters[0, 1] * dt} * C_out({parameters[0, 2]}) +\n'
              f'n*{dt * parameters[0, 3]}/V')
        print(f'Coefficients for N:\n'
              f'(C - {(1 - dt * sum(parameters[0, :2]))} * C_(i-1) -\n'
              f'{parameters[0, 0] * dt} * C_adj -\n'
              f'{parameters[0, 1] * dt} * C_out({parameters[0, 2]}))*V\n'
              f'/{dt * parameters[0, 3]}')
        max_N = get_max_N(temp_N)
        zone_id = 0
        param_id = 0  # quick fix
        for device, occupancy, C_adj, v, max_n in zip(dd_list, N_list, adj_list, V, max_N):
            if device[0] and occupancy[0]:
                C = [el[1] for el in device[date_index]]
                if smooth_co2:
                    C = uniform_filter1d(np.array(C), size=smooth_co2)
                N = occupancy[date_index]
                c_adj = C_adj[date_index]
                # print(C, N, occupancy)
                C_est = C_estimate_new(x=parameters[param_id], C=C, C_adj=c_adj, N=N, V=v, m=m, dt=dt)
                N_est = N_estimate(x=parameters[param_id], C=C, C_adj=c_adj, V=v, m=m, dt=dt)
                error_c = error_fraction(C[1:], C_est)
                error_n = error_fraction(N[1:], N_est)

                E_list[zone_id].append([error_c[2], error_n[2]])  # only list of errors
                E_summary_list[zone_id].append([error_c[:2], error_n[:2]])
                if plot:
                    plot_estimates(C=C[1:], C_est=C_est, N=N[1:], N_est=N_est, dt=dt, zone_id=zone_id,
                                   error_n=error_n[:2], error_c=error_c[1], start_time=device[date_index][0][0])

                if no_steps:
                    start = [el[0] for el in bounds]
                    end = [el[1] for el in bounds]
                    increments = [(el2 - el1) / no_steps for el1, el2 in zip(start, end)]
                    for sens_index in range(4):
                        # Reset the parameters to the optimal ones
                        temp_params = deepcopy(parameters[param_id])
                        for increment_factor in range(no_steps):
                            temp_params[sens_index] = increment_factor * increments[sens_index] + start[sens_index]
                            C_est = C_estimate(x=temp_params, C=C, C_adj=c_adj, N=N, V=v, m=m, dt=dt)
                            N_est = N_estimate(x=temp_params, C=C, C_adj=c_adj, V=v, m=m, dt=dt)
                            error_c_sens = error_fraction(C[1:], C_est)
                            error_n_sens = error_fraction(N[1:], N_est)
                            sensitivity_list_N[zone_id][sens_index][date_index].append(
                                [temp_params[sens_index], error_n_sens[1]])
                            sensitivity_list_C[zone_id][sens_index][date_index].append(
                                [temp_params[sens_index], error_c_sens[1]])

                param_id += 1
            zone_id += 1

    if summary_errors is not None:
        with open(summary_errors, 'w') as file:
            for i, zone_errors in enumerate(E_summary_list):
                file.write(f'{i} \n')
                for period in zone_errors:
                    for errors in period:
                        for el in errors[:2]:
                            file.write(str(el) + ' ')
                    file.write('\n')

    return dd_list, N_list, E_list, sensitivity_list_N, sensitivity_list_C


def simple_models_hold_out(dates, dt=15 * 60, method='l', plot=False, plot_scatter=False, n_zones=27):
    """
    Given dates, this function loads the device data and occupancy
    lists and uses them to hold-out validation of simpler methods.
    For Linear Regression the scatter plot and regression line can be
    shown in a plot, while the results in a specific zone can alway be
    plotted no matter the method. Curently there is Linear Regression
    and persistence.
    :param dates:
    :param dt:
    :param method:          only first letter in the name is needed, ('l', 'p')
    :param plot:
    :param plot_scatter:
    :return:
    """
    N_list, dd_list = load_lists(dates, dt)
    E_list = [[] for _ in range(n_zones + 1)]
    for index, date in enumerate(dates):
        zone_id = -1  # to keep track of what zone is being evaluated
        for device, occupancy in zip(dd_list, N_list):  # iterate over each zone
            zone_id += 1
            if len(device[0]) < 1 or len(occupancy[0]) < 1:  # skip empty zones
                continue

            counter = 0
            C_train, N_train = [], []
            for period_time_co2, period_N in zip(device, occupancy):  # iterate over each period in zone

                period_co2 = list(np.array(period_time_co2)[:, 1])
                if counter != index:
                    C_train = C_train + period_co2
                    N_train = N_train + period_N
                else:
                    start_time = device[0][0][0]
                counter += 1

            C_test, N_test = np.array(device[index])[:, 1], occupancy[index]

            if method.lower()[0] == 'l':
                C_test, N_test = np.array(list(C_test), dtype=float).reshape(-1, 1), np.array(list(N_test),
                                                                                              dtype=int).reshape(-1, 1)

                C_train = np.array(C_train).reshape(-1, 1)
                N_train = np.array(N_train).reshape(-1, 1)

                reg_N = LinearRegression().fit(C_train, N_train)
                N_est = np.round(reg_N.predict(C_test), 0)

                reg_co2 = LinearRegression().fit(N_train, C_train)
                C_est = reg_co2.predict(N_test)

                if plot_scatter:
                    plt.subplots_adjust(top=0.8)
                    plt.scatter(C_train, N_train)
                    plt.scatter(C_test, N_test, c='r')
                    plt.plot(C_train, reg_N.predict(C_train), c='c')
                    plt.xlabel('CO2 level')
                    plt.ylabel('Occupancy level')
                    plt.legend(['C train', 'C test', 'Line of best fit'])
                    plt.title(f'Linear regression scatter plot in zone {zone_id}\n'
                              f'Test point from period {start_time}\n'
                              f'Errors (Train, Test): {round(reg_N.score(C_train, N_train), 3), round(reg_N.score(C_test, N_test), 3)}')
                    plt.tight_layout()
                    plt.show()

                C_est, N_est = C_est.flatten(), N_est.flatten()
                C_test, N_test = C_test.flatten(), N_test.flatten()

                error_n = error_fraction(N_test, N_est)
                error_c = error_fraction(C_test, C_est)

                C_test, N_test = np.array([0] + [el for el in C_test]), np.array([0] + [el for el in N_test])

            elif method.lower()[0] == 'p':
                N_est = N_test[:-1]
                C_est = C_test[:-1]

                error_n = error_fraction([N_test[1:]], [N_est])
                error_c = error_fraction([C_test[1:]], [C_est])

            if plot:
                plot_estimates(C=C_test[1:], C_est=C_est, N=N_test[1:], N_est=N_est, dt=dt, zone_id=zone_id,
                               error_c=error_c[:2], error_n=error_n[:2], start_time=start_time, title='Lin Reg')
            E_list[zone_id].append([error_c[2], error_n[2]])

    return E_list


def return_average_parameters(dates, filename_parameters):
    """
    Given a list of dates and the filename used to store
    the optimised parameters, calculate the average in each zone
    and return this in the same format as the parameter files,
    along with the standard deviation.
    :param dates:
    :param filename_parameters:
    :return:
    """
    filenames_parameters = ['parameters/' + filename_parameters + '_' + date + '.csv' for date in dates]
    initial_frame = pd.read_csv(filenames_parameters[0]).values[:, 1:]
    all_parameters = np.zeros((initial_frame.shape[0], initial_frame.shape[1], len(filenames_parameters)))

    for i, parameter_file in enumerate(filenames_parameters):
        current_parameters = pd.read_csv(parameter_file).values[:, 1:]
        if i:
            initial_frame = initial_frame + current_parameters
        all_parameters[:, :, i] = current_parameters

    avg_parameters = initial_frame / len(filenames_parameters)
    return avg_parameters, np.std(all_parameters, axis=2)


# %% Loading
def load_occupancy(filename, n_zones=27, sep=';'):
    """
    Load the occupancy from a csv file created by Excel's
    vanilla csv function which says (comma delimited) despite
    being colon delimited.y
    :param filename:
    :param sep:
    :return:
    """
    df_N = pd.read_csv(filename, sep=sep)
    df_N.drop(df_N.columns[-1], axis=1, inplace=True)
    df_N.dropna(inplace=True)
    f = '%Y_%d_%m.%H.%M.%S'

    time_start = str_to_dt(filename[-14:-4] + '.' + df_N.values[0, 0], digits_to_remove=0, f=f)
    time_end = str_to_dt(filename[-14:-4] + '.' + df_N.values[-1, 0], digits_to_remove=0, f=f)
    zones = [name for name in df_N.columns[1:]]

    N = []
    for i in range(n_zones + 1):
        i = 'Z' + str(i)
        if i in zones:
            N.append(list(np.array(df_N[i].values, dtype=int)))
        else:
            N.append([])

    return N, time_start, time_end


def load_davide(save_filename=None, train_night_only=False, filepath_plots='documents/plots', smooth_co2=10):
    N_test, N_train, periods_test, periods_train, periods_adj_test, periods_adj_train = [], [], [], [], [], []
    C_all = []
    with open('data/davides_office.csv', 'r') as file:
        with open('data/davides_windows.csv', 'r') as file_windows:
            file.readline()
            temp_train, temp_test = [], []
            temp_N_train, temp_N_test = [], []
            temp_adj = []
            line_no = 0
            presence_set = set(range(22, 38)).union(range(129, 144)).union(range(463, 470))
            line_windows = file_windows.readline().split(',')
            t_windows = str_to_dt(line_windows[3][:16], f='%Y-%m-%d:%H:%M')
            for line in file.readlines():
                line = line.split(',')
                t = str_to_dt(line[3][:16], f='%Y-%m-%d:%H:%M')
                C_all.append(int(line[6]))
                presence = line_no in presence_set
                while t_windows < t and (len(line_windows) != 1):
                    t_windows = str_to_dt(line_windows[3][:16], f='%Y-%m-%d:%H:%M')
                    line_windows = file_windows.readline().split(',')
                adj = len(line_windows) != 1 and line_windows[6] == 'false'
                if train_night_only:
                    if 20 < t.hour or t.hour < 6:
                        if temp_test:
                            periods_test.append(temp_test)
                            N_test.append(temp_N_test)
                            temp_test, temp_N_test, temp_adj = [], [], []
                        temp_train.append([t, int(line[6])])
                        temp_N_train.append(presence)
                        temp_adj.append(adj)
                    else:
                        if temp_train:
                            periods_train.append(temp_train)
                            periods_test.append(temp_train)
                            N_train.append(temp_N_train)
                            N_test.append(temp_N_train)
                            temp_train, temp_N_train = [], []
                        temp_test.append([t, int(line[6])])
                        temp_N_test.append(presence)

                else:
                    temp_adj.append(adj)
                    temp_train.append([t, int(line[6])])
                    temp_N_train.append(presence)
                line_no += 1
    if train_night_only:
        dd_train = [periods_train]
        dd_test = [periods_test]
    else:
        dd_train = [[temp_train]]
        dd_test = [[temp_train]]
        N_train = [temp_N_train]
        N_test = [temp_N_train]

    v = 3 * 2 * 4
    dt = 10 * 60
    from constants import bounds_davide
    if temp_adj:
        parameters = optimise_occupancy(dd_train, [N_train], dt=dt, v=v, C_adj_list=[[temp_adj]], use_window=True,
                                        bounds=bounds_davide, smooth_co2=smooth_co2)
    else:
        parameters = optimise_occupancy(dd_train, [N_train], dt=dt, v=v, use_adjacent=False, bounds=bounds_davide, smooth_co2=smooth_co2)

    # C = [[el[1] for el in period_test] for period_test in periods_test]
    print(parameters)
    C_adj = adjacent_co2(dd_test, use_adjacent=False)[0]  # for 0'th device
    C_est, N_est = [], []
    C_flat, N_flat = [], []
    for c, n, c_adj in zip([C_all], N_test, C_adj):
        if smooth_co2:
            c = uniform_filter1d(np.array(c), size=10)
        C_flat = C_flat + [el for el in c[1:]]
        N_flat = N_flat + [el for el in n[1:]]
        C_est = C_est + [el for el in
                         C_estimate_new(parameters[0], C=np.array(c), C_adj=c_adj, N=np.array(n), V=v, dt=dt)]
        N_est = N_est + [el for el in
                         N_estimate(parameters[0], C=np.array(c), C_adj=c_adj, V=v, dt=dt)]

    error_c = error_fraction(C_flat, C_est)[:2]
    # print(np.average(np.abs(np.array(C_flat[:-1]) - np.array(C_flat[1:]))))
    errors_co2 = np.array(C_flat) - np.array(C_est)

    proportion_50_co2 = sum(np.abs(errors_co2) > 50)/len(errors_co2)
    proportion_100_co2 = sum(np.abs(errors_co2) > 100)/len(errors_co2)

    plt.hist(errors_co2, bins=100)
    plt.title(
        f'Histogram of CO2 errors in the office for MB\nError 50 ppm {round(proportion_50_co2, 4) * 100} % Error 100 ppm {round(proportion_100_co2, 4) * 100} %')
    plt.savefig(filepath_plots + '/davide_hist_co2.png')
    plt.show()

    N_est = np.array(N_est)
    N_flat = np.array(N_flat)
    plot_estimates(C_flat, C_est, N_flat, N_est, dt=dt, save_filename=save_filename,
                   title=f"CO2 prediction and occupancy detection in Davide's office\nAverage CO2 Error: {error_c[1]}"
                         f"\nProportion N error: {round(sum(N_est == N_flat)/len(N_flat),3)}",
                   legend_bar=['N True', 'N estimated'], legend_plot=['CO2 True', 'CO2 simulated'])

    print(f'Accuracy total: {round(sum(N_est == N_flat)/len(N_flat),3)}\n'
          f'Accuracy when present: {round(sum((N_est == N_flat)[N_flat == 1])/sum(N_flat == 1),3)}\n'
          f'Accuracy when not present: {round(sum((N_est == N_flat)[N_flat == 0])/sum(N_flat == 0),3)}\n')
    # no_0 = np.clip(N_est, 0, 10)
    # print(f'For no negative occupancy:\n Accuracy total: {round(sum(no_0 == N_flat) / len(N_flat),3)}\n'
    #       f'Accuracy when present: {round(sum((no_0 == N_flat)[N_flat == 1]) / sum(N_flat == 1),3)}\n'
    #       f'Accuracy when not present: {round(sum((no_0 == N_flat)[N_flat == 0]) / sum(N_flat == 0),3)}\n')
    return C_all, N_est, N_flat


def load_lists(dates, dt=15 * 60, filepath_and_prefix_co2='data/co2_', filepath_and_prefix_N='data/N_', n_zones=27):
    # The structure is:
    # Outer list corresponds to zone number and is called device
    # Each device is a list of periods
    # Each period is a (n x 2) list of (time, co2)
    N_list, dd_list = [[] for _ in range(n_zones + 1)], [[] for _ in range(n_zones + 1)]

    for date in dates:
        temp_name_c = filepath_and_prefix_co2 + date + '.csv'
        temp_name_n = filepath_and_prefix_N + date + '.csv'

        temp_N_list, start, end = load_occupancy(temp_name_n, n_zones=n_zones)
        temp_dd_list = load_data(temp_name_c, start, end, replace=True, dt=dt,
                                 no_points=len(temp_N_list[-1]), smoothing_type='exponential', n_zones=n_zones)
        for i in range(n_zones + 1):
            N_list[i].append((temp_N_list[i]))
            dd_list[i].append(temp_dd_list[i])

    return N_list, dd_list


def load_and_use_parameters(filepath_parameters, period_index, device_data_list, N, dt):
    # NEEDS UPDATING
    # Read the parameters for each zone, then calculate co2 and N estimates
    # based on the given data, plot in same manner as in optimise
    temp = pd.read_csv(filepath_parameters).values
    zone_ids = np.array(temp[:, 0], dtype=int)
    parameters = temp[:, 1:]

    for param, zone_id in zip(parameters, zone_ids):
        c = np.array(device_data_list[zone_id][period_index])[:, 1]
        n = N[zone_id][period_index]
        v = V[zone_id]

        C_est = calculate_co2_estimate(param, c, n, v, dt)
        N_est = calculate_n_estimate(param, c, v, dt)
        error_c = error_fraction(c, C_est)
        error_n = error_fraction(n, N_est)

        plot_estimates(c[1:], C_est, n[1:], N_est, dt, zone_id, device_data_list[zone_id][0][0], error_c[1],
                       error_n[:2])


def load_data(filename, start_time, end_time, dt=15 * 60, sep=',', format_time='%Y-%m-%d:%H:%M:%S.%f',
              digits_to_remove=1, filepath_averages='data/co2_time_average.csv', replace=1,
              no_points=None, smoothing_type='exponential', n_zones=27):
    """
    Given the filename of a csv file with three columns, one with
    device id's, one with co2 measurements and one with time of
    measurement, return a list containing the measurements from
    each device where the index in the list corresponds to the zone
    number it is from.
    :param filename:                    name of datafile
    :param start_time:                  threshold to cut off everything before
    :param end_time:                    threshold to cut off everything after
    :param filepath_averages:           file with replacement averages
    :param dt:                    time in seconds of the interval between measurements
    :param no_points:                   number of measurements of N
    :param replace:                     whether to replace missing data points or not
    :param digits_to_remove:            for formatting in datetime
    :param format_time:                 for formatting in datetime
    :param sep:                         for formatting in datetime
    :param smoothing_type:              for time steps with multiple measurements in between, options are exponential or Kalman
    :param n_zones:
    :return: device_data_list:          list of length n_zones with each item being the data for each device
    """
    dt = int(dt / 60)  # convert to minutes for this function
    df = pd.read_csv(filename, sep=sep)
    time_index = np.argmax(df.columns == 'telemetry.time')
    co2_index = np.argmax(df.columns == 'telemetry.co2')
    id_index = np.argmax(df.columns == 'deviceId')
    # To make indices correspond to zone number, the 0'th element will simply be empty
    device_data_list = [[] for _ in range(n_zones + 1)]

    # Format the datetime strings, map the device ids and find the starting time if needed
    for row in df.values:
        time = str_to_dt(row[time_index], digits_to_remove=digits_to_remove, f=format_time)
        if start_time < time < end_time:
            co2 = row[co2_index]
            device_id = id_map[row[id_index]]
            device_data_list[device_id].append([time, co2])
    try:
        zone_averages = pd.read_csv(filepath_averages).values[:, 1:]
    except FileNotFoundError:
        # 600 is a decent estimate
        zone_averages = np.ones((n_zones + 1, 96)) * 600

    for i, device in enumerate(device_data_list):
        # To start we want all measurements up to 15 minutes after the rounded first time
        # start_time = round_dt(start_time[i], minutes=interval_smoothing_length, up=False) \
        # + timedelta(minutes=interval_smoothing_length)
        data = np.array(device_data_list[i])

        # Skip sensors with less than 2 measurements, because with this no change can be detected
        if len(data) < 2:
            device_data_list[i] = []
            continue

        # sort data by time
        data = data[data[:, time_index].argsort()]
        # initialise variables
        new_data, index = [], 0
        temp_time = copy(start_time)
        # Keep looping until either there is data in every time slot
        while index < len(data) or (no_points is not None and len(new_data) < no_points):
            temp = []
            # ensure that if index is exceeds data length it does not crash, simply skips trying
            # to use data from the data
            if index < len(data):
                # append all data points created before relevant time to temp
                while data[index][time_index] < temp_time:
                    temp.append(data[index])
                    index += 1
                    if index == len(data):
                        break

            # Check if there was any data
            if temp:
                if smoothing_type[0].lower() == 'e':
                    co2_smoothed = exponential_moving_average(temp, tau=dt*3)
                elif smoothing_type[0].lower() == 'k':
                    co2_smoothed = kalman_estimates(np.array(temp)[:, 1])[0][-1]
                new_data.append((temp_time, co2_smoothed))
            else:  # Time and None if nothing recorded unless it is replaced by the average
                emp = None
                # print(f'Time {temp_time} missing from zone {i}')
                if replace:
                    # Find the position in the average time array with which to sub
                    column = int(temp_time.hour * 4 + temp_time.minute / dt)
                    emp = zone_averages[i, column] * (1 - replace) + replace * new_data[-1][1] \
                        if len(new_data) > 0 else zone_averages[i, column]

                new_data.append([temp_time, emp])
            # Increment relevant time
            temp_time = temp_time + timedelta(minutes=dt)

        device_data_list[i] = new_data

    return device_data_list


# %% Helpers

def matrix_to_latex(table, d=2, zone_index=True):
    """
    Given a 2D array like, return a string which can be
    copied directly into a LaTex table, one can specify the
    number of decimals to be rounded to.
    :param table:
    :param d:
    :param zone_index:
    :return: out
    """
    out = ''
    np.round(np.asarray(table), d)
    for row_mean in table:
        for i, el in enumerate(row_mean):
            if i == 0:
                out = out + f'{int(el)} & '
            else:
                out = out + f'{round(el, sigfigs=d)} & '
        out = out[:-1] + ' \\\\ \n'
    return out[:-7]


def get_max_N(N_list):
    """
    Find the maximum occupancy for each zone in an N_list
    and return in a list of same length as N_list
    :param N_list:
    :return:
    """
    max_N = [0 for _ in N_list]
    for i, device_N in enumerate(N_list):
        for period_N in device_N:
            if period_N:
                max_N[i] = max(period_N)
    return max_N


def exponential_moving_average(x, tau=900):
    """
    Given CO2 measurements with creation time, return the values smoothed
    based on the previous measurements. tau specifies the weight given to
    past measurements, the larger the more stable and further in the past
    is weighted higher and vice versa.
    :param x:   contains time as first column and CO2 as second
    :param tau: given in seconds
    :return:
    """
    x = np.array(x)
    C, t = x[:, 1], x[:, 0]
    # total seconds is necessary for robustness
    t = np.array([el.total_seconds() for el in (t[-1] - t)])
    smoothed = [C[0] for _ in C]
    for j in range(1, len(t)):
        w = np.exp(-(t[j - 1] - t[j]) / tau)

        smoothed[j] = smoothed[j - 1] * w + C[j] * (1 - w)

    # return the last element as this is the newest estimate
    return smoothed[-1]


def kalman_estimates(C, min_error=50, error_proportion=0.03):
    """
    Given a list of observations and the error values relevant for CO2,
    compute the kalman filtered value and error for each consecutive
    data point. No time is considered, only error.
    :param C:
    :param min_error:
    :param error_proportion:
    :return:
    """
    # Vocab:    E_est: estimate error
    #           EST:   estimate
    #           E_est_p: previous estimate error
    # remove first element to make code look nicer
    EST = C[0]
    C = C[1:]
    # initial error
    E_est = max(EST * error_proportion, min_error)
    E_est_list = np.array([E_est] + [0 for _ in range(len(C))], dtype=float)
    E_m_list = []
    KGs = np.empty(len(C))

    estimates = np.array([EST] + [0 for _ in range(len(C))])

    for i, m in enumerate(C):
        # Define previous and measurement errors:
        E_est_p = E_est_list[i]
        E_m = max(m * error_proportion, min_error)
        E_m_list.append(E_m)
        # Calculate the Kalman Gain (KG)
        EST_p = estimates[i]
        KG = E_est_p / (E_est_p + E_m)
        KGs[i] = KG

        # The new error can be calculated using the Kalman Gain as:
        E_est = (1 - KG) * E_est_p
        E_est_list[i + 1] = E_est
        # Calculate the new estimate using KG:
        EST = EST_p + KG * (m - EST_p)
        estimates[i + 1] = EST
        # print(f'{i}, KG={KG}, E_EST_p={E_est_p}, E_est={E_est} m={m}')

    return estimates, E_est_list


def residual_analysis(dd_list, N_list, E_list, E_list_reg, filepath_plots='documents/plots/', plot=True,
                      do_summary=False):
    from scipy.stats import norm, wilcoxon, probplot, shapiro
    from statsmodels.stats.contingency_tables import mcnemar
    all_co2, all_N, errors_co2, errors_N = [], [], [], []
    errors_co2_reg, errors_N_reg = [], []
    # errors by device (EBD) dimension 0 -> device, dimension 1 -> mean,std
    EBD_N, EBD_co2 = [[] for _ in dd_list], [[] for _ in dd_list]
    EBD_N_reg, EBD_co2_reg = [[] for _ in dd_list], [[] for _ in dd_list]
    detected, detected_reg = [[] for _ in dd_list], [[] for _ in dd_list]
    detected_noneg, detected_reg_noneg = [[] for _ in dd_list], [[] for _ in dd_list]

    all_detected, all_detected_reg = [], []

    dev_id = 0
    for device, device_N, device_errors, device_errors_reg in zip(dd_list, N_list, E_list, E_list_reg):
        C, C_est, C_est_reg, N, N_est, N_est_reg = [], [], [], [], [], []
        for period, period_N, period_errors, period_errors_reg in zip(device, device_N, device_errors,
                                                                      device_errors_reg):
            for co2, n, error_co2, error_N, error_co2_reg, error_N_reg in zip(np.array(period)[:, 1], period_N,
                                                                              period_errors[0], period_errors[1],
                                                                              period_errors_reg[0][1:],
                                                                              period_errors_reg[1][1:]):
                C.append(co2)
                C_est.append(co2 + error_co2)
                C_est_reg.append(co2 + error_co2_reg)
                N.append(n)
                N_est.append(n + error_N)
                N_est_reg.append(n + error_N_reg)

                all_co2.append(co2)
                all_N.append(n)
                errors_co2.append(error_co2)
                errors_N.append(error_N)
                EBD_co2[dev_id].append(abs(error_co2))
                EBD_N[dev_id].append(abs(error_N))
                detected[dev_id].append(bool(n) == bool(n + error_N))
                detected_noneg[dev_id].append(bool(n) == (n + error_N >= 0))

                errors_co2_reg.append(error_co2_reg)
                errors_N_reg.append(error_N_reg)
                EBD_N_reg[dev_id].append(abs(error_N_reg))
                EBD_co2_reg[dev_id].append(abs(error_co2_reg))
                detected_reg[dev_id].append(bool(n) == bool(n + error_N_reg))
                detected_reg_noneg[dev_id].append(bool(n) == (n + error_N_reg >= 0))

                all_detected.append(bool(n) and bool(n + error_N))
                all_detected_reg.append(bool(n) and bool(n + error_N_reg))
        if plot:
            if C_est and N_est:
                print(f'plotting {dev_id}..')
                plot_estimates(C, C_est, N, N_est, title=f'All data for zone {dev_id} Mass Balance',
                               x_vals=range(len(N)), width=0.5, x_label='Time steps',
                               save_filename=filepath_plots + f'All_errors_MB_{dev_id}')
                plot_estimates(C, C_est_reg, N, N_est_reg, title=f'All data for zone {dev_id} Linear Regression',
                               x_vals=range(len(N)), width=0.5, x_label='Time steps',
                               save_filename=filepath_plots + f'All_errors_LR_{dev_id}')
        dev_id += 1

    if plot:
        plt.scatter(all_co2, errors_co2, marker='.')
        cor_co2 = round(np.corrcoef(all_co2, errors_co2)[1, 0], 3)
        plt.title(f'CO2 residual plot MB\nCorrelation: {cor_co2}')
        plt.ylabel('CO2 Residual')
        plt.xlabel('CO2')
        plt.savefig(filepath_plots + 'co2_residual_MB', bbox_inches='tight')
        plt.show()

        plt.scatter(all_N, errors_N, marker='.')
        cor_N = round(np.corrcoef(all_N, errors_N)[1, 0], 3)
        plt.title(f'N residual plot MB\nCorrelation: {cor_N}')
        plt.ylabel('N Residual')
        plt.xlabel('N')
        plt.savefig(filepath_plots + 'N_residual_MB')
        plt.show()

        plt.scatter(all_co2, errors_N, marker='.')
        cor_N = round(np.corrcoef(all_co2, errors_N)[1, 0], 3)
        plt.title(f'CO2/N residual plot MB\nCorrelation: {cor_N}')
        plt.ylabel('N Residual')
        plt.xlabel('CO2')
        plt.savefig(filepath_plots + 'co2_N_residual_MB')
        plt.show()

        plt.scatter(all_co2, errors_co2_reg, marker='.')
        cor_co2 = round(np.corrcoef(all_co2, errors_co2_reg)[1, 0], 3)
        plt.title(f'CO2 residual plot reg\nCorrelation: {cor_co2}')
        plt.ylabel('CO2 Residual')
        plt.xlabel('CO2')
        plt.savefig(filepath_plots + 'co2_residual_LR', bbox_inches='tight')
        plt.show()

        plt.scatter(all_N, errors_N_reg, marker='.')
        cor_N = round(np.corrcoef(all_N, errors_N_reg)[1, 0], 3)
        plt.title(f'N residual plot MB\nCorrelation: {cor_N}')
        plt.ylabel('N Residual')
        plt.xlabel('N')
        plt.savefig(filepath_plots + 'N_residual_LR')
        plt.show()

        plt.scatter(all_co2, errors_N_reg, marker='.')
        cor_N = round(np.corrcoef(all_co2, errors_N_reg)[1, 0], 3)
        plt.title(f'CO2/N residual plot reg\nCorrelation: {cor_N}')
        plt.ylabel('N Residual')
        plt.xlabel('CO2')
        plt.savefig(filepath_plots + 'co2_N_residual_LR')
        plt.show()

        probplot(errors_N, dist="norm", plot=plt)
        shapiro_p = shapiro(errors_N).pvalue
        plt.title(f'Q-Q plot N errors Mass Balance\nShapiro p-value: {shapiro_p}')
        plt.savefig(filepath_plots + 'qq_n_MB')
        plt.show()

        probplot(errors_N_reg, dist="norm", plot=plt)
        shapiro_p = shapiro(errors_N_reg).pvalue
        plt.title(f'Q-Q plot N errors Linear Regression\nShapiro p-value: {shapiro_p}')
        plt.savefig(filepath_plots + 'qq_n_lr')
        plt.show()

        probplot(errors_co2, dist="norm", plot=plt)
        shapiro_p = shapiro(errors_co2).pvalue
        plt.title(f'Q-Q plot CO2 errors Mass Balance\nShapiro p-value: {shapiro_p}')
        plt.savefig(filepath_plots + 'qq_co2_MB')
        plt.show()

        probplot(errors_co2_reg, dist="norm", plot=plt)
        shapiro_p = shapiro(errors_co2_reg).pvalue
        plt.title(f'Q-Q plot CO2 errors Linear Regression\nShapiro p-value: {shapiro_p}')
        plt.savefig(filepath_plots + 'qq_co2_lr')
        plt.show()

        proportion_50_co2 = sum(np.abs(errors_co2) > 50)/len(errors_co2)
        proportion_100_co2 = sum(np.abs(errors_co2) > 100)/len(errors_co2)
        proportion_50_co2_reg = sum(np.abs(errors_co2_reg) > 50)/len(errors_co2_reg)
        proportion_100_co2_reg = sum(np.abs(errors_co2_reg) > 100)/len(errors_co2_reg)
        plt.hist(errors_co2, bins=100)
        plt.title(f'Histogram of CO2 errors for Mass Balance\nError +-50 ppm: {round(proportion_50_co2*100, 1)} % Error +- 100 ppm {round(proportion_100_co2*100, 1)} %')
        plt.savefig(filepath_plots + 'hist_co2.png')
        plt.show()

        plt.hist(errors_co2_reg, bins=100)
        plt.title(
            f'Histogram of CO2 errors for Linear Regression\nError +-50 ppm: {round(proportion_50_co2_reg * 100, 1)} % Error +- 100 ppm {round(proportion_100_co2_reg * 100, 1)} %')
        plt.savefig(filepath_plots + 'hist_co2_reg.png')
        plt.show()


    print('For N:')
    if do_summary:
        print('Summary for mass balance errors:')
        temp = pd.Series(np.abs(errors_N))
        print(temp.describe())
        print('Summary for linear regression errors:')
        temp = pd.Series(np.abs(errors_N_reg))
        print(temp.describe())
    wilcox = wilcoxon(all_N, errors_N_reg)
    print(f'P-values for wilcox rank sum test {wilcox.pvalue}')

    print('For co2:')
    if do_summary:
        print('Summary for mass balance errors:')
        temp = pd.Series(np.abs(errors_co2))
        print(temp.describe())
        print('Summary for linear regression errors:')
        temp = pd.Series(np.abs(errors_co2_reg))
        print(temp.describe())
    wilcox = wilcoxon(all_co2, errors_co2_reg)
    print(f'P-values for wilcox rank sum test {wilcox.pvalue}')

    df = pd.crosstab(all_detected, all_detected_reg)
    print(f'McNemar p-value for detecting occupancy:\n{mcnemar(df)}')

    table_mean, table_std = [], []
    table_detect_noneg = []
    zone_id = 0
    for error_n_mb, error_n_lr, error_co2_mb, error_co2_lr, error_detected_mb, error_detected_lr, el1, el2 in zip(EBD_N,
                                                                                                                  EBD_N_reg,
                                                                                                                  EBD_co2,
                                                                                                                  EBD_co2_reg,
                                                                                                                  detected,
                                                                                                                  detected_reg,
                                                                                                                  detected_noneg,
                                                                                                                  detected_reg_noneg):
        if error_n_mb:
            table_mean.append(
                [zone_id, np.average(error_n_mb), np.average(error_n_lr), np.average(error_co2_mb),
                 np.average(error_co2_lr), np.average(error_detected_mb), np.average(error_detected_lr)])
            table_std.append([zone_id, error_n_mb[1], error_n_lr[1], error_co2_mb[1], error_co2_lr[1]])
            table_detect_noneg.append([sum(el1) / len(el1), sum(el2) / len(el2)])
        zone_id += 1
    table_mean, table_std = np.asarray(table_mean), np.asarray(table_std)

    return table_mean, table_std, table_detect_noneg


def sensitivity_plots(sensitivity_list, avg_params, avg_errors, filepath_plots='documents/plots/', post_fix='N'):
    """
    Using the sensitivity list, do the plots

    :param sensitivity_list:
    :param avg_params:          matrix of average parameters found through optimisation
    :param avg_errors:          vector of average errors for parameters
    :param filepath_plots:
    :return:
    """
    # dimension 0 -> (Q_adj, Q_out, C_out, m), dimension 1 -> zone
    sens_plotting_list = [[] for _ in range(4)]
    legend = []
    for zone_id, zone in enumerate(sensitivity_list):
        if zone[0][0]:
            legend.append(f'Zone {zone_id}')
            legend.append('_Hidden label')
            for i in range(4):
                # temp = np.array([np.array(el) for el in zone[i]])
                sens_plotting_list[i].append(np.array(zone[i]).mean(axis=0))
    x_labels = ['Q_adj', 'Q_out', 'C_out', 'm']
    for sens_set, x_label, avg_param in zip(sens_plotting_list, x_labels, avg_params.T):
        for i, zone_line in enumerate(sens_set):
            plt.plot(zone_line[:, 0], zone_line[:, 1])
            plt.scatter(avg_param[i], avg_errors[i])
        plt.xlabel(x_label)
        plt.ylabel(f'{post_fix} error')
        plt.legend(legend)
        plt.title(f'Average {post_fix} error plotted against ' + x_label)
        if filepath_plots != '':
            plt.savefig(filepath_plots + 'sensitivity_' + x_label + '_' + post_fix)
        plt.show()


def plot_estimates(C, C_est, N, N_est, dt=60 * 15, x_vals=None, zone_id=None, start_time=None, error_c=None,
                   error_n=None, title='', width=4, alpha=0.4, x_label='', save_filename=None,
                   legend_bar=None, legend_plot=None):
    """
    Given the relevant parameters and the associated errors
    plot the results.
    :param C:
    :param C_est:
    :param N:
    :param N_est:
    :param zone_id:
    :param start_time:
    :param dt:              time step in seconds
    :param error_c:
    :param error_n:
    :return:
    """

    fig, ax1 = plt.subplots(1, 1)
    if x_vals is None:
        x_vals = np.arange(0, len(C_est) * dt / 60, dt / 60)
    ax1.plot(x_vals, C, color='b')
    ax1.plot(x_vals, C_est, color='c')
    plt.ylabel('CO2 concentration (ppm)')
    if x_label == '':
        plt.xlabel('Time (min)')
    if title == '':
        plt.title(
            f'Measured CO2 level vs estimate from optimisation in zone {zone_id}\nat start time {start_time}\nAvg'
            f'. CO2 error: {error_c}, N error: {error_n}\n Blue is true C, Red is true N'
        )
    else:
        plt.title(title)
    ax1.legend(['CO2', 'CO2 estimated'])

    ax2 = ax1.twinx()
    ax2.bar(x_vals, N, color='red', alpha=alpha - 0.2, width=width)
    ax2.bar(x_vals, N_est, color='orange', alpha=alpha, width=width)
    if legend_bar is not None:
        ax2.legend(legend_bar)
    if legend_plot is not None:
        ax1.legend(legend_plot, loc='upper center')
    ax2.legend(['N', 'N estimated'])
    plt.tight_layout()
    # ax1.legend(['CO2 true', 'CO2 Estimated'], loc='upper left', title='Metric: ppm')
    # ax2.legend(['N true', 'N Estimated'], loc='upper right', title='Rounded to integer')
    # ax1.set_xticklabels(ax1.get_xticks(), rotation=30)
    # ax2.set_xticklabels(ax2.get_xticks(), rotation=30)
    if save_filename is not None:
        plt.savefig(save_filename)
    plt.show()


def adjacent_co2(dd_list, n_map=None, use_adjacent=True):
    """
    Given the device data list find calculate the average of the
    two neighbouring zone's co2 level as measured at the time of
    every measurement. the n-map dictionary is made and if left
    blank will simply be imported from the constant.py file.
    :param dd_list:
    :param use_adjacent:        flag, if False set all values to 0
    :param n_map:
    :return:
    """
    r_list = [[] for _ in dd_list]
    if n_map is None:
        from constants import n_map

    for device_index, device in enumerate(dd_list):
        if not device:
            continue

        neighbours = n_map[device_index]

        for period_index, period in enumerate(device):
            period_replacement = []

            for time_index, el in enumerate(period):
                co2_neighbours = []
                for neighbour in neighbours:

                    if dd_list[neighbour][period_index]:  # check if there is data from the period in the
                        # neighbouring zone
                        co2_neighbours.append(dd_list[neighbour][period_index][time_index][1])
                if use_adjacent:
                    if co2_neighbours:
                        period_replacement.append(np.average(co2_neighbours))
                    elif device[period_index][time_index]:
                        period_replacement.append(device[period_index][time_index][1])
                else:
                    period_replacement.append(0)
            r_list[device_index].append(period_replacement)

    return r_list


def get_replacement_and_derivative(C, C_out, time_step=5 * 60, alpha=0.05, n_map=None):
    # Calculate vectors from CO2:
    dC = (C[:, 0] - C[:, 1]) / time_step
    Cr = np.empty(27)

    if n_map is None:
        from constants import n_map
    for zone_number in range(len(C[:, 0])):
        temp = []
        for neighbour in n_map[zone_number + 1]:
            neighbour -= 1
            if C[:, 0][neighbour] > 0:
                temp.append(C[:, 0][neighbour])
        Cr[zone_number] = alpha * C_out + (1 - alpha) * np.average(temp)
    Cr[C[:, 0] == 0] = 0

    return Cr, dC


def str_to_dt(t, chars_to_remove='T', digits_to_remove=1, f='%Y-%m-%d:%H:%M:%S.%f'):
    if digits_to_remove:
        t = t[:-digits_to_remove]
    for char in chars_to_remove:
        t = t.replace(char, ':')
    # Convert to datetime object
    return datetime.strptime(t, f)


def abs_distance(x, C, C_adj, N, V, m, dt, optimise_N=False, use_window=False):
    """
    Calculates the absolute difference between the estimated CO2
    and True CO2:
        m       CO2 per person
        C_out   CO2 concentration outdoors
        Q       Airflow rate with outdoors (and neighbouring zones, to be implemented)
    :param x:               parameters being optimised
    :param C:               measured CO2 levels
    :param N:               number of people
    :param V:               volume of zone
    :param dt:              time step
    :param optimise_N:
    :return:
    """

    C_est, N_est = [], []
    for c, n, c_adj in zip(C, N, C_adj):
        C_est.append(
            C_estimate_new(x, C=np.array(c), N=np.array(n), C_adj=c_adj, V=V, m=m, dt=dt, use_window=use_window))
        N_est.append(N_estimate(x, C=np.array(c), C_adj=c_adj, V=V, m=m, dt=dt))

    dist = 0
    if optimise_N:
        for n, n_est in zip(N, N_est):
            dist += abs(sum(n[1:] - np.array(n_est)))
    else:
        for c, c_est in zip(C, C_est):
            dist += sum(abs(np.array(c[1:]) - np.array(c_est)))

    # This will return the distance we are minimising
    return dist


def C_estimate(x, C, C_adj, N, V, dt=15 * 60, m=15, d=2, use_window=False):
    """
    Calculates the estimated CO2 given parameters
    :param x:               Q, m and C_out
    :param C:               measured CO2 levels
    :param C_adj:
    :param N:               number of people
    :param V:               volume of zone
    :param m:
    :param dt:              time step
    :param d:
    :return:
    """
    Q_adj, Q_out, C_out, m = x
    Q = Q_adj + Q_out

    N = np.array(N)
    C = np.array(C) / ppm_factor  # ppm
    if use_window:
        C_adj = np.array(C_adj)[1:] * C_out / ppm_factor
    else:
        C_adj = np.array(C_adj)[1:] / ppm_factor

    Ci, N = C[:-1], N[1:]  # Remove first N, as there is no previous CO2
    C_est = (1 - Q * dt) * Ci + \
            Q_adj * dt * C_adj + \
            Q_out * dt * C_out + \
            N * dt * m / V
    # print('new')
    # print((1 - Q * dt) * Ci, Q_adj * dt * C_adj, Q_out * dt * C_out, N * dt * m / V)
    C_est = np.array(C_est, dtype=float) * ppm_factor
    return np.round(C_est, decimals=d)


def N_estimate(x, C, C_adj, V, dt=15 * 60, m=15, d=0):
    """
    Given all necessary parameters, calculate the estimated
    number of occupants in a zone. Can take scalars and vector
    as long as C is a vector of length at least 2 containing
    previous and current CO2.
    :param x:
    :param C:
    :param C_adj:
    :param V:
    :param m:
    :param dt:
    :param d:
    :param max_n:
    :return:
    """
    Q_adj, Q_out, C_out, m = x
    Q = Q_adj + Q_out

    C_adj = np.array(C_adj)[1:] / ppm_factor
    C = np.array(C) / ppm_factor  # ppm
    Ci = C[:-1]
    C = C[1:]
    N = np.array(V * (C - (1 - Q * dt) * Ci -
                      Q_adj * dt * C_adj -
                      Q_out * dt * C_out) / (dt * m), dtype=float)

    # At least 0 people ignore max_n
    # N = [n if n > 0 else 0 for n in N]
    return np.round(N, d)


def C_estimate_new(x, C, C_adj, N, V, dt=15 * 60, m=15, d=2, use_window=False):
    """
    Calculates the estimated CO2 given parameters
    :param x:               Q, m and C_out
    :param C:               measured CO2 levels
    :param C_adj:
    :param N:               number of people
    :param V:               volume of zone
    :param m:
    :param dt:              time step
    :param d:
    :param use_window:
    :return:
    """

    Q_adj, Q_out, C_out, m = x
    Q = Q_adj + Q_out

    N = np.array(N)
    C = np.array(C)
    C_est = [C[0]]

    if use_window:
        C_adj = np.array(C_adj)[1:] * C_out
    else:
        C_adj = np.array(C_adj)[1:]

    for i, c in enumerate(C[1:]):
        c_est = (1 - Q * dt) * C_est[-1] + \
                Q_adj * dt * C_adj[i] + \
                Q_out * dt * C_out + \
                N[i] * dt * m / V
        # if N[i]:
        #     print('New')
        #     print((1 - Q * dt), Q_out * dt)
        #     print((1 - Q * dt) * C_est[-1])
        #     print(Q_out * dt * C_out)
        #     print(N[i] * dt * m / V)
        C_est.append(c_est)

    return np.round(C_est[1:], decimals=d)


def error_fraction(true_values, estimated_values, d=2):
    """
    Given the true and estimated values, return the proportion of
    time steps where they do not match, the average error and finally
    a list of the errors.
    Uses the new format of list of periods, and therefore unpacks
    those lists to do the calculation
    :param true_values:
    :param estimated_values:
    :param d:                   decimals in rounding
    :return:
    """

    n_false = 0
    n_total = 0
    error_list = []
    if len(true_values) != len(estimated_values):
        print('Dimension mismatch between true and estimated outer lists')
        print(len(true_values), len(estimated_values))
    for t, e in zip(true_values, estimated_values):
        n_false += not t == e
        error_list.append(t - e)
        n_total += 1
    return np.round(n_false / n_total, d), np.round(sum(np.abs(error_list)) / n_total, d), error_list


def optimise_occupancy(dd_list, N_list, m=15, dt=15 * 60, optimise_N=False, bounds=None, smooth_co2=10,
                       filename_parameters=None, use_adjacent=True, v=None, use_window=False, C_adj_list=None):
    """
    Given data in the format from the above function and potentially
    vectors representing the occupancy and volumes, find the optimal
    Q, m and CO2 concentration outdoors

    :param optimise_N:
    :param m:
    :param dd_list:             list of lists of data from each zone, 0'th element is empty
    :param N_list:              list of occupancy from each zone, assumes same order as device data
    :param dt:                  float, time step in seconds
    :param bounds:              tuple of tuple of bounds for the parameters
    :param filename_parameters: string of filename to store parameters in
    :return:
    """

    if C_adj_list is None:
        C_adj_list = adjacent_co2(dd_list, use_adjacent=use_adjacent)

    x = []
    for bound in bounds:
        x.append((bound[0] + bound[1]) / 2)

    parameters, zone_ids = [], []
    empty_zones = []
    np.random.seed(41)

    for i, device in enumerate(dd_list):
        # skip every iteration where zone has no occupancy/co2 data
        if device[0] and len(N_list[i][0]) > 0:
            zone_ids.append(i)
            # extract the C's, N's and C_adj's as lists of periods
            C = [[el[1] for el in period] for period in device]
            if smooth_co2:
                C = uniform_filter1d(np.array(C[0]), size=smooth_co2).reshape(1, -1)
            N = N_list[i]
            C_adj = C_adj_list[i]
            if v is None:
                v = np.array(V[i])
            minimised = differential_evolution(
                abs_distance,
                x0=x,
                args=(C, C_adj, N, v, m, dt, optimise_N, use_window),
                bounds=bounds,
                # method=method
            )

            C_est, N_est = [], []
            C_flat, N_flat = [], []
            for c, n, c_adj in zip(C, N, C_adj):
                if smooth_co2:
                    c = uniform_filter1d(np.array(c), size=10)
                C_flat = C_flat + [el for el in c[1:]]
                N_flat = N_flat + [el for el in n[1:]]
                C_est = C_est + [el for el in
                                 C_estimate_new(minimised.x, C=np.array(c), C_adj=c_adj, N=np.array(n), V=v, m=m,
                                                dt=dt)]
                N_est = N_est + [el for el in
                                 N_estimate(minimised.x, C=np.array(c), C_adj=c_adj, V=v, m=m, dt=dt)]
            error_c = error_fraction(C_flat, C_est)
            error_n = error_fraction(N_flat, N_est)
            print(f'Zone {i}:\nAverage CO2 Error: {error_c[1]}\n'
                  f'Occupancy error (proportion wrong, average error): {error_n[:2]}')

            parameters.append(minimised.x)
        elif i != 0:
            empty_zones.append(i)
    print(f'There is no data from {empty_zones}')
    if filename_parameters is not None:
        parameters = np.array(parameters)
        zone_ids = np.array(zone_ids)
        file_contents = np.empty((parameters.shape[0], parameters.shape[1] + 1))
        file_contents[:, 0] = zone_ids
        file_contents[:, 1:] = parameters
        df = pd.DataFrame(file_contents)
        df.to_csv(filename_parameters, index=False)

    return parameters


def round_dt(dt, minutes=15, up=False):
    delta = timedelta(minutes=minutes)
    if up:
        return datetime.min + np.ceil((dt - datetime.min) / delta) * delta
    else:
        return datetime.min + np.floor((dt - datetime.min) / delta) * delta


def level_from_estimate(N, M, treshs=(0.3, 0.7)):
    """
    Given occupancy estimate and maximum capacity arrays
    map them to an occupancy level (0-3) based on thresholds
    defined in t. If an N is exactly 0 assume unknown (0)
    :param N:       array of occupancy estimates
    :param M:       array of maximum capacity
    :param treshs:  tuple of two thresholds
    :return: output array of occupancy level
    """

    percentage = N / M
    output = [0 if p == 0 else (1 if p < treshs[0] else (2 if p < treshs[1] else 3)) for p in percentage]

    return output


# %% Redundant stuff
def calculate_co2_estimate(x, C, N, V, dt, d=2, no_steps=None, rho=1.22):
    """
    Calculates the estimated CO2 given parameters
    :param x:               Q, m and C_out
    :param C:               measured CO2 levels
    :param N:               number of people
    :param V:               volume of zone
    :param dt:              time step
    :param d:
    :param rho:
    :param no_steps:       to be iterated over for generation, assume same time step
    :return:
    """
    Q, m, C_out = x
    C = np.array(C)
    N = np.array(N)
    if no_steps is not None:  # C is then the first CO2 value
        C_est = [C]
        for i in range(no_steps - 1):
            # i is then the previous index in C_est
            C_est.append((dt * (Q * C_out + m * N[i + 1]) + V * C_est[i]) / (Q * dt + V))
    else:
        Ci, N = C[:-1], N[1:]  # Remove first N, as there is no previous CO2
        C_est = np.array((dt * (Q * C_out + m * N * rho) + V * Ci * rho) / (Q * dt + rho * V), dtype=np.longdouble)

    return np.round(C_est, decimals=d)


def calculate_n_estimate(x, C, V, dt, d=0, rho=1.22):
    """
    Given all necessary parameters, calculate the estimated
    number of occupants in a zone. Can take scalars and vector
    as long as C is a vector of length at least 2 containing
    previous and current CO2.
    :param x:
    :param C:
    :param V:
    :param dt:
    :param d:
    :param rho:
    :return:
    """
    Q, m, C_out = x
    C = np.array(C)

    Ci = C[:-1]
    C = C[1:]

    N = np.array((V * (C - Ci) * rho + Q * (C - C_out) * dt) / (dt * rho * m), dtype=float)
    # At least 0 people
    N = [n if n > 0 else 0 for n in N]
    return np.round(N, d)


def summary_stats_datetime_difference(time1, time2, p=True):
    """
    Given 2 numpy arrays of datetimes, compute the mean, median and
    standard deviation of their difference. If mean and median are negative
    this function calculates the reverse difference
    :param time1:       array of datetimes
    :param time2:       array of datetimes
    :param p:           bool: to print or not to print
    :return: m, M, sd   float: mean, median and standard deviation
    """
    obj = pd.to_timedelta(pd.Series(time2 - time1))
    m, M, sd = obj.mean(), obj.median, obj.std()
    if m < timedelta(seconds=0) and M < timedelta(seconds=0):
        obj = pd.to_timedelta(pd.Series(time1 - time2))
        m, M, sd = obj.mean(), obj.median, obj.std()
    if p:
        print(f'Mean: {m}\nMedian: {M}\nSD: {sd}')
    return m, M, sd


def process_data(df, minutes, time_indexes=None, id_index=1):
    """
    Function to call to format the data so it can be used in update_data
    1: convert the time string into datetime object
    2: map device ID's to zone numbers
    3: remove too old data points
    4: sort based on time
    :param df:              pandas dataframe of data from sql server
    :param minutes:         now - minutes is threshold for new data
    :param time_indexes:      iterable column indexes of time variables, first el is used to sort by
    :param id_index:        column index of device ID's
    :return: data:          numpy array of data in correct format
    """

    if time_indexes is None:
        time_indexes = [0]
    data = np.array(df.values)
    # Convert all time strings to datetime to perform arithmetics on them
    for i, time_index in enumerate(time_indexes):
        for j, t in enumerate(data[:, time_index]):
            # Convert to datetime object
            data[j, time_index] = str_to_dt(t)
            # Convert to number corresponding to zone
            if not i:  # Only map data once
                data[j, id_index] = id_map[data[j, id_index]]
    # Remove rows where time is before specified time
    time_cutoff = datetime.now() - timedelta(minutes=minutes)
    data = data[data[:, time_indexes[0]] > time_cutoff]
    # Sort by date so newest is at the top
    data = np.flip(data[data[:, time_indexes[0]].argsort()], axis=0)

    return data


def update_data(new_data, old_data, old_time, time_index=0, co2_index=1, id_index=2):
    """
    Given the new rows of data in the PROCESSED FORMAT find the
    devices that have produced new co2 data. If they have produced
    two outputs these will be stored in order. The old_data is then
    supplanted in the entries where there is new data.
    :param new_data:        (nxm) matrix of new data points
    :param old_data:        (27x2) matrix of data used in the previous iteration
    :param old_time:        (27x2) matrix of timestamps from previous iteration
    :param time_index:      int of the index where the time label is
    :param id_index:        int of the index in new data where the zone ID's are stored
    :param co2_index:       int of the index in new data where the co2 data is stored
    :return: output         (27x2) matrix of the most up to date data available
    :return: output_time    (27x2) List of timestamps of data
    """
    output = old_data
    output_time = old_time
    # Assumes newest data comes first
    for row in new_data:
        device_id = row[id_index] - 1  # convert to comply with 0 indexed arrays
        # If no data from the current device has been seen yet, input it in the first column of output
        if not output[device_id][0]:
            output_time[device_id][0] = row[time_index]
            output[device_id][0] = row[co2_index]
        # If there is data in the first column, input the data in the second column
        elif not output[device_id][1]:
            output_time[device_id][1] = row[time_index]
            output[device_id][1] = row[co2_index]
        # If there is data in both, do nothing

    return output, output_time


def simulate_office():
    parameter_mat = np.empty(shape=(4, 3))
    co2_scaling = 1
    co2_pp, c_out = 15, 380 / co2_scaling
    qi, qm, qw = 0.05, 4, 0.5
    parameter_mat[0] = np.array([0, co2_pp, c_out])  # No q
    parameter_mat[1] = np.array([qi, co2_pp, c_out])  # infiltration
    parameter_mat[2] = np.array([qi + qm, co2_pp, c_out])  # infiltration + mechanical ventilation
    parameter_mat[3] = np.array([qi + qm + qw, co2_pp, c_out])  # infiltration + mechanical ventilation + window

    no_steps = 10000
    no_hours = 1
    hour_scaling = 3600
    volume = 100
    no_people = 1
    Cg, Ng = 450 / co2_scaling, np.ones(no_steps) * no_people

    for parameter_set in parameter_mat:
        step = no_hours * hour_scaling / no_steps
        plt.plot(
            np.arange(0, no_hours * hour_scaling, step),
            co2_scaling * calculate_co2_estimate(parameter_set, Cg, Ng, V=volume, dt=step, no_steps=no_steps)
        )

    plt.legend([f'Nothing (Q={0})',
                f'Infiltration (Q={qi})',
                f'Mechanical (Q={qi + qm})',
                f'Window (Q={qi + qm + qw})'])
    plt.title(f'CO2 level vs time in an office of volume={volume} occupied by {no_people} person(s)\n '
              f'CO2 per person={co2_pp} CO2 outdoors={c_out}')
    plt.ylabel('CO2 concentration (ppm)')
    plt.xlabel(f'Time ({"hours" if hour_scaling == 1 else "seconds"})')
    plt.show()


def check_missing_data(device_data_list, replace=False, return_count=False, verbose=False):
    """
    Given the data in the format from 'data_for_optimising'
    count the number of missing data points, replace them
    with the previous data if specified
    :param device_data_list:
    :param replace:
    :param return_count:
    :param verbose:
    :return:
    """
    missing_list = []
    no_missing, no_replaced = 0, 0
    for i, data in enumerate(device_data_list):
        temp = []
        for j, el in enumerate(data):
            if not el:
                no_missing += 1
                if verbose:
                    print(f'Data from zone {i}, at index {j} is missing')
                missing_list.append((i, j))

                if replace and j > 0:
                    if data[j - 1]:  # only replace with previous as would be the case in real application
                        temp.append((j, data[j - 1]))
                        if verbose:
                            print(f'Data from zone {i}, at index {j} was replaced with the previous data point')
                        no_replaced += 1

        for j, dat in temp:  # update data with the replacement
            data[j] = dat
    print(
        f'There were {no_missing} missing points and {no_replaced} were replaced. Ratio: {no_replaced / no_missing if no_missing else no_missing}')
    return missing_list


def log_likelihood(x, C, N, V, dt, uncertainty=50, percent=0.03, verbose=True):
    """
    Calculates the log log_likelihood of the current parameters, by
    finding the pdf of the normal distribution with mean = the
    measured CO2 level and standard deviation from the specifications.
    Since we are calculating the log likelihood, we need to minimise.
    Parameters being optimised are:
        m       CO2 per person
        C_out   CO2 concentration outdoors
        Q       Airflow rate with outdoors (and neighbouring zones, to be implemented)
    :param x:               parameters being optimised
    :param C:               measured CO2 levels
    :param N:               number of people
    :param V:               volume of zone
    :param dt:              time step
    :param percent:         percent uncertainty of sensors
    :param uncertainty:     minimum uncertainty of sensors
    :param verbose:         to print or not to print
    :return:
    """
    uncertainty, percent = uncertainty / 2, percent / 2  # it is the 95 % confidence, therefor 2 sd's

    C_est = calculate_co2_estimate(x, C, N, V, dt)
    sd = np.array([uncertainty + el * percent for el in C[1:]])
    log_l = sum(np.log(norm.pdf(C_est, loc=C[1:], scale=sd)))
    if verbose:
        print(
            f'Average absolute difference: {np.average(np.abs(C_est - C[1:]))}')  # compare to C[1:] as there is no first estimate
        print(f'log_likelihood: {log_l}')
        print(f'Paramters: {x}')
        print(f'Average C: {np.average(C)}')
        print(f'Average C_est: {np.average(C_est)}\n\n')

    # This will return the negative log likelihood as we are minimising
    return -log_l


def basic_weighting(Ci, Ci0, n_total, decimals=0, M=None, assume_unknown=False):
    """
    Takes the vectors of CO2 and baseline CO2, then applies the
    simple weighting of the occupancy in the zones from the total
    number of occupants. Can also include the weighting based on
    maximum occupants per zone as specified by M. This method appears
    very prone to rounding errors. Can either ignore the unknown zones
    or spread the mean of the other zones to those unknown zones.
    :param Ci:              vector of current CO2
    :param Ci0:             vector of baseline CO2
    :param n_total:         integer of total occupants
    :param decimals:        number of decimals for rounding
    :param M:               vector of maximum occupancy per zone
    :param assume_unknown:  bool of what to do with unknown zones
    :return: N_estimate     vector of estimated N in each zone
    """
    Ci.flatten()
    Ci0.flatten()
    # Quick fix for zones with no data:
    Ci0[Ci == 0] = 0
    N_estimated = n_total * (Ci - Ci0) / sum(Ci - Ci0)
    if M is not None:
        N_estimated = N_estimated * M / np.average(M)
    if assume_unknown:
        # Calculate mean from the estimate where only zones with data are included
        mean = np.average(N_estimated[N_estimated != 0])
        # Use this to spread out the mean to those unknown zones
        n_unknown = len(N_estimated) - np.count_nonzero(N_estimated)
        N_estimated[N_estimated != 0] -= n_unknown / (len(N_estimated) - n_unknown) * mean
        N_estimated[N_estimated == 0] = mean
    return N_estimated.round(decimals)


def mass_balance_helper(X):
    C, dC, Cr, Q, V, m = X
    N = (Q * (C - Cr) + V * dC) / m
    return N


def mass_balance(C, Q, V, n_total, current_time=[], n_map=None, C_out=420, alpha=0.7, time_step=5 * 60, m=20,
                 decimals=0, M=None, fill_neighbours=False):
    """
    This function calculates the derivative of Ci, creates the replacement
    CO2 vector from the neighbour map (n_map) and calculates the estimated
    N by using the calculated N as a proportion of the total N

    :param C:           (27,2) vector of CO2 data from current and previous time_step
    :param Q:           (27,1) vector of airflow rates in each zone
    :param V:           (27,1) vector of volumes of zones
    :param current_time: (27,2) vector of time of creation for each co2 measurement
    :param n_map:       dictionary where key is zone no and values are neighbouring zones
    :param m:           float CO2 exhaled per person
    :param M:           (27,1) vector of maximum capacity in each zone, if left blank is not considered
    :param decimals:    int of decimals to round to
    :param time_step:   float time between measurements
    :param alpha:       float/vector of proportion of outdoor air in exchange
    :param n_total:     int number of people in the entire library
    :param C_out:       float CO2 concentration outdoors
    :param fill_neighbours: bool of whether to use average of neighbouring zones for estimation, do not use!!
    :return:            vector of estimated number of people per zone
    """
    Q.flatten()
    V.flatten()
    if n_map is None:
        n_map = {1: [2, 10], 2: [1, 3], 3: [2, 4], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10],
                 10: [1, 9], 11: [12, 20], 12: [11, 13], 13: [12, 14], 14: [13, 15], 15: [14, 16], 16: [15, 17],
                 17: [16, 18], 18: [17, 19], 19: [18, 20], 20: [11, 19], 21: [22, 27], 22: [21, 23], 23: [22, 24],
                 24: [23, 25], 25: [24, 26], 26: [25, 27], 27: [21, 26]
                 }
    if fill_neighbours:
        for zone_number in range(len(C[:, 0])):
            temp = []
            if C[:, 0][zone_number] == 0:
                for neighbour in n_map[zone_number + 1]:
                    neighbour -= 1
                    if C[:, 0][neighbour] > 0:
                        temp.append(C[:, 0][neighbour])
                C[:, 0][zone_number] = np.average(temp)

    if len(current_time) > 0:
        time_step = []
        for i, times in enumerate(current_time):
            delta = times[0] - times[1]
            time_step.append(delta.seconds)
        time_step = np.array(time_step)

    Cr, dC = get_replacement_and_derivative(C, C_out, time_step, alpha)

    N = (Q * (C[:, 0] - Cr) + V * dC) / m
    N_estimated = N / np.sum(N) * n_total
    if M is not None:
        M.flatten()
        N_estimated = N_estimated * M / np.average(M)
    return N_estimated.round(decimals)


if __name__ == '__main__':
    filepath_plots = 'documents/plots/'
    temp, N_est, N_flat = load_davide(save_filename=filepath_plots + 'davide_plot.png', smooth_co2=5)

