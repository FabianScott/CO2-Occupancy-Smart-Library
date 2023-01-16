import numpy as np
import pandas as pd
from copy import copy
from datetime import datetime
from datetime import timedelta
from scipy.stats import norm
from constants import id_map
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def hold_out(dates, V, m=15, plot=False, filename_parameters='testing', bounds=None, dt=15):
    """
    Given a list of dates in the format yyyy_dd_mm, same as filenames
    for data, use the hold out method on each period of data, using it
    as a test period for the parameters optimised on the rest.
    :param dates:
    :param V:
    :param plot:
    :param filename_parameters:
    :param bounds:
    :param dt:
    :return:
    """

    if bounds is None:
        from constants import bounds

    N_list, dd_list = load_lists(dates, dt)
    adj_list = adjacent_co2(dd_list)
    # Use the index in the date list to hold out each period once
    for index, date in enumerate(dates):

        temp_dd, temp_N = [], []
        for device, occupancy in zip(dd_list, N_list):
            temp_dd.append(device[:index] + device[index + 1:])
            temp_N.append(occupancy[:index] + occupancy[index + 1:])

        fname_param = f'parameters/{filename_parameters}_{date}.csv'
        parameters = optimise_occupancy(temp_dd, N_list=temp_N, V_list=V, plot_result=False,
                                        filename_parameters=fname_param, bounds=bounds)

        if plot:
            zone_id = 0
            param_id = 0  # lazy quick fix
            for device, occupancy, C_adj, v in zip(dd_list, N_list, adj_list, V):
                if device[0] and occupancy[0]:
                    C = [el[1] for el in device[index]]
                    N = occupancy[index]
                    c_adj = C_adj[index]
                    # print(C, N, occupancy)
                    C_est = [C_estimate(x=parameters[param_id], C=C, C_adj=c_adj, N=N, V=v, m=m, dt=dt)]
                    N_est = [N_estimate(x=parameters[param_id], C=C, C_adj=c_adj, V=v, m=m, dt=dt)]
                    error_c = error_fraction([C[1:]], C_est)[1]
                    error_n = error_fraction([N[1:]], N_est)
                    plot_estimates(C=[C], C_est=C_est, N=[N], N_est=N_est, dt=dt, zone_id=zone_id,
                                   error_n=error_n, error_c=error_c, start_time=device[index][0][0])
                    param_id += 1
                zone_id += 1

    return dd_list, N_list


def simple_models_hold_out(dates, dt=15 * 60, method='l', plot=False, plot_scatter=False):
    """
    Given dates, this function loads the device data and occupancy
    lists and uses them to to hold-out validation of simpler methods.
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
    E_list = [[] for _ in range(28)]
    for index, date in enumerate(dates):
        zone_id = 0  # to keep track of what zone is being evaluated
        for device, occupancy in zip(dd_list, N_list):  # iterate over each zone
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
                    plt.title(f'Linear regression scatter plot in zone {zone_id}\n'
                              f'Test point from period {start_time}\n'
                              f'R^2 value (Train, Test): {round(reg_N.score(C_train, N_train), 3), round(reg_N.score(C_test, N_test), 3)}')
                    plt.show()

                C_est, N_est = C_est.flatten(), N_est.flatten()
                C_test, N_test = C_test.flatten(), N_test.flatten()

                error_n = error_fraction([N_test], [N_est])
                error_c = error_fraction([C_test], [C_est])

                C_test, N_test = np.array([0] + [el for el in C_test]), np.array([0] + [el for el in N_test])

            elif method.lower()[0] == 'p':
                N_est = N_test[:-1]
                C_est = C_test[:-1]

                error_n = error_fraction([N_test[1:]], [N_est])
                error_c = error_fraction([C_test[1:]], [C_est])

            if plot:
                plot_estimates(C=[C_test], C_est=[C_est], N=[N_test], N_est=[N_est], dt=dt, zone_id=zone_id,
                               error_c=error_c, error_n=error_n, start_time=start_time)
            print(zone_id)
            E_list[zone_id].append((error_c, error_n))
            zone_id += 1

    return E_list


# %% Loading
def load_occupancy(filename, sep=';'):
    """
    Load the occupancy from a csv file created by Excel's
    vanilla csv function which says (comma delimited) despite
    being colon delimited.
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
    for i in range(28):
        i = 'Z' + str(i)
        if i in zones:
            N.append(list(np.array(df_N[i].values, dtype=int)))
        else:
            N.append([])

    return N, time_start, time_end


def load_lists(dates, dt):
    # The structure is:
    # Outer list corresponds to zone number and is called device
    # Each device is a list of periods
    # Each period is a (n x 2) list of (time, co2)
    N_list, dd_list = [[] for _ in range(28)], [[] for _ in range(28)]

    for date in dates:
        temp_name_c = 'data/co2_' + date + '.csv'
        temp_name_n = 'data/N_' + date + '.csv'

        N, start, end = load_occupancy(temp_name_n)
        device_data_list = load_data(temp_name_c, start, end, replace=True, interval=dt,
                                     no_points=len(N[-1]), smoothing_type='exponential')
        for i in range(28):
            N_list[i].append((N[i]))
            dd_list[i].append(device_data_list[i])

    return N_list, dd_list


def load_and_use_parameters(filepath_parameters, period_index, device_data_list, N, V, dt):
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
        error_c = error_fraction(c, C_est)[1]
        error_n = error_fraction(n, N_est)

        plot_estimates(c, C_est, n, N_est, dt, zone_id, device_data_list[zone_id][0][0], error_c, error_n)


def load_data(filename, start_time, end_time, interval=15 * 60, sep=',', format_time='%Y-%m-%d:%H:%M:%S.%f',
              digits_to_remove=1,
              filepath_averages='data/co2_time_average.csv', replace=1, no_points=None, smoothing_type='exponential'):
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
    :param interval:                    time in seconds of the interval between measurements
    :param no_points:                   number of measurements of N
    :param replace:                     whether to replace missing data points or not
    :param digits_to_remove:            for formatting in datetime
    :param format_time:                 for formatting in datetime
    :param sep:                         for formatting in datetime
    :param smoothing_type:              for time steps with multiple measurements in between, options are exponential or Kalman
    :return: device_data_list:          list of length 28 with each item being the data for each device
    """
    interval = int(interval / 60)  # convert to minutes for this function
    df = pd.read_csv(filename, sep=sep)
    time_index = np.argmax(df.columns == 'telemetry.time')
    co2_index = np.argmax(df.columns == 'telemetry.co2')
    id_index = np.argmax(df.columns == 'deviceId')
    # To make indices correspond to zone number, the 0'th element will simply be empty
    device_data_list = [[] for _ in range(28)]

    # Format the datetime strings, map the device ids and find the starting time if needed
    for row in df.values:
        time = str_to_dt(row[time_index], digits_to_remove=digits_to_remove, f=format_time)
        if start_time < time < end_time:
            co2 = row[co2_index]
            device_id = id_map[row[id_index]]
            device_data_list[device_id].append([time, co2])
    try:
        zone_averages = pd.read_csv(filepath_averages).values
    except FileNotFoundError:
        # 600 is a decent estimate
        zone_averages = np.ones((28, 96)) * 600

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
                    co2_smoothed = exponential_moving_average(temp, tau=interval)
                elif smoothing_type[0].lower() == 'k':
                    co2_smoothed = kalman_estimates(np.array(temp)[:, 1])[0][-1]

                new_data.append((temp_time, co2_smoothed))
            else:  # Time and None if nothing recorded unless it is replaced by the average
                emp = None
                # print(f'Time {temp_time} missing from zone {i}')
                if replace:
                    # Find the position in the average time array with which to sub
                    column = int(temp_time.hour * 4 + temp_time.minute / interval)
                    emp = zone_averages[i, column] * (1 - replace) + replace * new_data[-1][1] \
                        if len(new_data) > 0 else zone_averages[i, column]
                new_data.append([temp_time, emp])
            # Increment relevant time
            temp_time = temp_time + timedelta(minutes=interval)

        device_data_list[i] = new_data

    return device_data_list


# %% Helpers

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


def plot_estimates(C, C_est, N, N_est, dt, zone_id=None, start_time=None, error_c=None, error_n=None):
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
    if error_c is None:
        error_c = error_fraction(C, C_est)
    if error_n is None:
        error_n = error_fraction(N, N_est)

    x_dim = int(np.ceil(np.sqrt(len(C_est))))
    y_dim = int(np.ceil(len(C_est) / x_dim))
    fig, axs = plt.subplots(x_dim, y_dim)
    plt.title(
        f'Measured CO2 level vs estimate from optimisation in zone {zone_id}\nat start time {start_time}\nAvg'
        f'. CO2 error: {error_c}, N error: {error_n}\n B is true C, R is true N')
    axs = np.asarray(axs)

    for i, ax1 in enumerate(axs.flatten()):
        x_vals = np.arange(0, len(C_est[i]) * dt / 60, dt / 60)
        ax1.plot(x_vals, C[i][1:], color='b')
        ax1.plot(x_vals, C_est[i], color='c')
        plt.ylabel('CO2 concentration (ppm)')
        plt.xlabel('Time (min)')

        ax2 = ax1.twinx()
        ax2.bar(x_vals, N[i][1:], color='orange', alpha=0.4, width=4)
        ax2.bar(x_vals, N_est[i], color='red', alpha=0.4, width=4)

        # ax1.legend(['CO2 true', 'CO2 Estimated'], loc='upper left', title='Metric: ppm')
        # ax2.legend(['N true', 'N Estimated'], loc='upper right', title='Rounded to integer')
        # ax1.set_xticklabels(ax1.get_xticks(), rotation=30)
        # ax2.set_xticklabels(ax2.get_xticks(), rotation=30)
        plt.subplots_adjust(top=0.8)

    plt.show()


def adjacent_co2(dd_list, n_map=None):
    """
    Given the device data list find calculate the average of the
    two neighbouring zone's co2 level as measured at the time of
    every measurement. the n-map dictionary is made and if left
    blank will simply be imported from the constant.py file.
    :param dd_list:
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

                for neighbour in neighbours:
                    co2_neighbours = []

                    if dd_list[neighbour][
                        period_index]:  # check if there is data from the period in the neighbouring zone
                        co2_neighbours.append(dd_list[neighbour][period_index][time_index][1])

                if co2_neighbours:
                    period_replacement.append(np.average(co2_neighbours))
                elif device[period_index][time_index]:
                    period_replacement.append(device[period_index][time_index][1])
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


def abs_distance(x, C, C_adj, N, V, m, dt, verbose=False, zone=1):
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
    :param verbose:         to print or not to print
    :return:
    """

    C_est = []
    for c, n, c_adj in zip(C, N, C_adj):
        C_est.append(C_estimate(x, C=np.array(c), N=np.array(n), C_adj=c_adj, V=V, m=m, dt=dt))

    dist = 0
    for c, c_est in zip(C, C_est):
        dist += sum(np.abs(c[1:] - np.array(c_est)))

    if verbose:
        print(f'Zone {zone}:')
        print(
            f'Average absolute difference: {np.average(np.abs(C_est - C[1:]))}')  # compare to C[1:] as there is no first estimate
        print(f'Parameters: {x}')
        print(f'Average C: {np.average(C)}')
        print(f'Average C_est: {np.average(C_est)}\n\n')

    # This will return the distance we are minimising
    return dist


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


def C_estimate(x, C, C_adj, N, V, dt, m, d=2, rho=1.22):
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
    :param rho:
    :return:
    """
    Q_adj, Q_out, C_out, m = x
    Q = Q_adj + Q_out

    C = np.array(C)
    N = np.array(N)
    C_adj = np.array(C_adj)[1:]

    Ci, N = C[:-1], N[1:]  # Remove first N, as there is no previous CO2
    Q = Q_adj + Q_out
    C_est = (1 - Q * dt) * Ci + \
            Q_adj * dt * C_adj + \
            Q_out * dt * C_out + \
            N * dt * m / V

    return np.round(C_est, decimals=d)


def N_estimate(x, C, C_adj, V, m, dt, d=0, rho=1.22):
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
    :param rho:
    :return:
    """
    Q_adj, Q_out, C_out, m = x
    Q = Q_adj + Q_out

    C_adj = np.array(C_adj)[1:]
    C = np.array(C)
    Ci = C[:-1]
    C = C[1:]
    N = np.array(V * (C - (1 - Q * dt) * Ci -
                   Q_adj * dt * C_adj -
                   Q_out * dt * C_out) / (dt * m), dtype=float)

    # At least 0 people
    N = [n if n > 0 else 0 for n in N]
    return np.round(N, d)


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


def error_fraction(true_values, estimated_values, d=2):
    """
    Given the true and estimated values, return the proportion of
    time steps where they do not match and the average error.
    Uses the new format of list of periods, and therefore unpacks
    those lists to do the calculation
    :param true_values:
    :param estimated_values:
    :param d:                   decimals in rounding
    :return:
    """

    n_false = 0
    n_total = 0
    error_size = 0
    if len(true_values) != len(estimated_values):
        print('Dimension mismatch between true and estimated outer lists')
    for T, E in zip(true_values, estimated_values):
        if len(T) != len(E):
            # print(f'Dimension mismatch between true and estimated period lists: {len(T), len(E)}')
            if len(T) - 1 == len(E):
                # print('Corrected, hopefully')
                T = T[1:]
        for t, e in zip(T, E):
            n_false += not t == e
            error_size += abs(t - e)
            n_total += 1

    return np.round(n_false / n_total, d), np.round(error_size / n_total, d)


def optimise_occupancy(dd_list, N_list, V_list, m=15, dt=15 * 60, bounds=None, verbosity=False,
                       plot_result=False, filename_parameters=None):
    """
    Given data in the format from the above function and potentially
    vectors representing the occupancy and volumes, find the optimal
    Q, m and CO2 concentration outdoors

    :param dd_list:             list of lists of data from each zone, 0'th element is empty
    :param N_list:              list of occupancy from each zone, assumes same order as device data
    :param V_list:              list of volumes for each zone
    :param dt:                  float, time step in seconds
    :param bounds:              tuple of tuple of bounds for the parameters
    :param verbosity:           to print or not to print
    :param method:              optimisation method for scipy's minimise
    :param plot_result:         to show plots of the result or not
    :param filename_parameters: string of filename to store parameters in
    :return:
    """
    C_adj_list = adjacent_co2(dd_list)
    if bounds is None:
        from constants import bounds

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
            # exctract the C's, N's and C_adj's as lists of periods
            C = [[el[1] for el in period] for period in device]
            N = N_list[i]
            C_adj = C_adj_list[i]

            V = np.array(V_list[i])

            minimised = differential_evolution(
                abs_distance,
                x0=x,
                args=(C, C_adj, N, V, m, dt, verbosity, i,),
                bounds=bounds,
                # method=method
            )

            C_est, N_est = [], []
            for c, n, c_adj in zip(C, N, C_adj):
                C_est.append(C_estimate(minimised.x, C=np.array(c), C_adj=c_adj, N=np.array(n), V=V, m=m, dt=dt))
                N_est.append(N_estimate(minimised.x, C=np.array(c), C_adj=c_adj, V=V, m=m, dt=dt))
            error_c = error_fraction(C, C_est)[1]
            error_n = error_fraction(N, N_est)
            print(f'Zone {i}:\nAverage CO2 Error: {error_c}\n'
                  f'Occupancy error (proportion wrong, average error): {error_n}')

            if plot_result:
                plot_estimates(C, C_est, N, N_est, dt, i, device[:][0][0], error_c, error_n)

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
