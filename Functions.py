import numpy as np
import pandas as pd
from copy import copy
from datetime import datetime
from datetime import timedelta
from scipy.stats import norm
from constants import id_map
from scipy.optimize import minimize
import matplotlib.pyplot as plt


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


def get_replacement_and_derivative(C, C_out, time_step=5 * 60, alpha=0.05, n_map=None):
    if n_map is None:
        n_map = {1: [2, 10], 2: [1, 3], 3: [2, 4], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10],
                 10: [1, 9], 11: [12, 20], 12: [11, 13], 13: [12, 14], 14: [13, 15], 15: [14, 16], 16: [15, 17],
                 17: [16, 18], 18: [17, 19], 19: [18, 20], 20: [11, 19], 21: [22, 27], 22: [21, 23], 23: [22, 24],
                 24: [23, 25], 25: [24, 26], 26: [25, 27], 27: [21, 26]
                 }

    # Calculate vectors from CO2:
    dC = (C[:, 0] - C[:, 1]) / time_step
    Cr = np.empty(27)
    for zone_number in range(len(C[:, 0])):
        temp = []
        for neighbour in n_map[zone_number + 1]:
            neighbour -= 1
            if C[:, 0][neighbour] > 0:
                temp.append(C[:, 0][neighbour])
        Cr[zone_number] = alpha * C_out + (1 - alpha) * np.average(temp)
    Cr[C[:, 0] == 0] = 0

    return Cr, dC


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
            data[j, time_index] = string_to_datetime(t)
            # Convert to number corresponding to zone
            if not i:  # Only map data once
                data[j, id_index] = id_map[data[j, id_index]]
    # Remove rows where time is before specified time
    time_cutoff = datetime.now() - timedelta(minutes=minutes)
    data = data[data[:, time_indexes[0]] > time_cutoff]
    # Sort by date so newest is at the top
    data = np.flip(data[data[:, time_indexes[0]].argsort()], axis=0)

    return data


def string_to_datetime(t, chars_to_remove='T', digits_to_remove=1, f='%Y-%m-%d:%H:%M:%S.%f'):
    if digits_to_remove:
        t = t[:-digits_to_remove]
    for char in chars_to_remove:
        t = t.replace(char, ':')
    # Convert to datetime object
    return datetime.strptime(t, f)


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


def abs_distance(x, C, N, V, dt, verbose=True, zone=1):
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
    :param verbose:         to print or not to print
    :return:
    """

    C_est = calculate_co2_estimate(x, C, N, V, dt)
    dist = sum(np.abs(C[1:] - C_est))
    if verbose:
        print(f'Zone {zone}:')
        print(
            f'Average absolute difference: {np.average(np.abs(C_est - C[1:]))}')  # compare to C[1:] as there is no first estimate
        print(f'Parameters: {x}')
        print(f'Average C: {np.average(C)}')
        print(f'Average C_est: {np.average(C_est)}\n\n')

    # This will return the negative log likelihood as we are minimising
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
    Ci = C[:-1]
    C = C[1:]
    N = (V * (C - Ci) * rho + Q * (C - C_out) * dt) \
        / (dt * rho * m)

    return np.round(N, d)


def error_fraction(true_values, estimated_values, d=2):
    """
    Given the true and estimated values, return the proportion of
    time steps where they do not match and the average error.
    :param true_values:
    :param estimated_values:
    :return:
    """

    true_values = true_values[1:]
    n_false = 0
    error_size = 0
    for i, el in enumerate(estimated_values):
        n_false += not true_values[i] == el
        error_size += abs(true_values[i] - el)

    return np.round(n_false / len(true_values),d), np.round(error_size / len(true_values),d)


def round_dt(dt, minutes=15, up=False):
    delta = timedelta(minutes=minutes)
    if up:
        return datetime.min + np.ceil((dt - datetime.min) / delta) * delta
    else:
        return datetime.min + np.floor((dt - datetime.min) / delta) * delta


def load_data(filename, interval_smoothing_length=15, sep=',', format_time='%Y-%m-%d:%H:%M:%S.%f', digits_to_remove=1,
              filepath_averages='data/co2_time_average.csv', replace=False):
    """
    Given the filename of a csv file with three columns, one with
    device id's, one with co2 measurements and one with time of
    measurement, return a list containing the measurements from
    each device where the index in the list corresponds to the zone
    number it is from.
    :param filename:
    :param interval_smoothing_length:
    :return:
    """

    df = pd.read_csv(filename, sep=sep)
    time_index = np.argmax(df.columns == 'telemetry.time')
    co2_index = np.argmax(df.columns == 'telemetry.co2')
    id_index = np.argmax(df.columns == 'deviceId')
    # To make indices correspond to zone number, the 0'th element will simply be empty
    device_data_list = [[] for _ in range(28)]

    start_time = [datetime(year=9990, month=12, day=1) for _ in range(28)]

    for row in df.values:
        co2 = row[co2_index]
        time = string_to_datetime(row[time_index], digits_to_remove=digits_to_remove,f=format_time)
        device_id = id_map[row[id_index]]
        device_data_list[device_id].append([time, co2])
        if time < start_time[device_id]:  # smaller time is earlier
            start_time[device_id] = time

    zone_averages = pd.read_csv(filepath_averages).values

    for i, device in enumerate(device_data_list):

        # To start we want all measurements up to 15 minutes after the rounded first time
        start_time[i] = round_dt(start_time[i], minutes=interval_smoothing_length, up=False) \
                             + timedelta(minutes=interval_smoothing_length)
        data = np.array(device_data_list[i])

        # Skip sensors with less than 2 measurements, because with this no change can be detected
        if len(data) < 2:
            device_data_list[i] = []
            continue

        # sort data by time
        data = data[data[:, time_index].argsort()]
        # initialise variables
        new_data, index = [], 0

        while index < len(data):

            temp = []
            # print(f'The index is now: {index}, relevant time is: {start_time[i]}')
            # append all data points created before relevant time to temp
            while data[index][time_index] < start_time[i] and index < len(data) - 1:
                temp.append(data[index])
                index += 1

            # Check if there was any data
            if temp:
                new_data.append([start_time[i], exponential_moving_average(temp, tau=interval_smoothing_length)])
            else:  # Time and None if nothing recorded
                emp = None
                if replace:
                    # Find the position in the average time array with which to sub
                    column = int(start_time[i].hour * 4 + start_time[i].minute / interval_smoothing_length)
                    emp = zone_averages[i, column]
                new_data.append([start_time[i], emp])
            # Increment relevant time
            start_time[i] = start_time[i] + timedelta(minutes=interval_smoothing_length)
        device_data_list[i] = new_data

    return device_data_list


def optimise_occupancy(device_data_list, N=None, V=None, dt=15 * 60, bounds=None, verbosity=True, method=None,
                       plot_result=False):
    """
    Given data in the format from the above function and potentially
    vectors representing the occupancy and volumes, find the optimal
    Q, m and CO2 concentration outdoors
    :param device_data_list:
    :param N:
    :param V:
    :param dt: 
    :param bounds: 
    :param verbosity:
    :param method:
    :return:
    """
    if bounds is None:
        q_min, q_max = (0.01, 0.2)
        m_min, m_max = (10, 20)
        c_min, c_max = (300, 450)

        bounds = ((q_min, q_max), (m_min, m_max), (c_min, c_max))

    x = np.array([bounds[0][0] - (bounds[0][0] - bounds[0][1]) / 2,
                  bounds[1][0] - (bounds[1][0] - bounds[1][1]) / 2,
                  bounds[2][0] - (bounds[2][0] - bounds[2][1]) / 2, ])
    if V is None:
        V = np.ones(len(device_data_list)) * 250



    parameters = []
    np.random.seed(41)
    for i, device in enumerate(device_data_list):
        if device:
            c = []
            # Do fix this
            for el in device:
                if el[1] is not None:
                    c.append(el[1])
                else:   # to be decided how to weight the previous measurement vs time average
                    if c[-1]:
                        c.append(c[-1])
                    else:
                        # calculate the average co2 concentration at a time of day and use
                        c.append(550)

            c = np.array(c, dtype=float)
            v = V[i]
            if N is None:
                m1 = len(c)
                l1 = [0 for _ in range(m1)]
                n = np.array(l1, dtype=int)

            else:
                n = N[i]

            minimised = minimize(
                abs_distance,
                x0=x,
                args=(c, n, v, dt, verbosity, i,),
                bounds=bounds,
                method=method
            )

            C_est = calculate_co2_estimate(minimised.x, c, n, v, dt)
            N_est = calculate_n_estimate(minimised.x, c, v, dt)
            error_c = error_fraction(c, C_est)[1]
            error_n = error_fraction(n, N_est)
            print(f'Average CO2 Error: {error_c}\n'
                  f'Occupancy error (proportion wrong, average error): {error_n}')

            if plot_result:
                ax1 = plt.subplot()
                x_vals = np.arange(0, len(C_est) * dt / 60, dt / 60)
                ax1.plot(x_vals, c[1:])
                ax1.plot(x_vals, C_est)
                plt.ylabel('CO2 concentration (ppm)')
                plt.xlabel('Time (min)')

                # x_vals = np.arange(0, len(N_est))
                ax2 = ax1.twinx()
                ax2.plot(x_vals, n[1:], color='y')
                ax2.scatter(x_vals, N_est, color='r', s=0.5)

                ax1.legend(['CO2 true', 'CO2 Estimated'], loc='upper left', title='Metric: ppm')
                ax2.legend(['N true', 'N Estimated'], loc='upper right', title='Rounded to integer')

                plt.title(f'Measured CO2 level vs estimate from optimisation in zone {i}\nAvg. CO2 error: {error_c}, N error: {error_n}')
                plt.show()
            parameters.append(minimised.x)
        elif i != 0:
            print(f'No data from zone {i}')
    return parameters


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
                    if data[j-1]:   # only replace with previous as would be the case in real application
                        temp.append((j, data[j-1]))
                        if verbose:
                            print(f'Data from zone {i}, at index {j} was replaced with the previous data point')
                        no_replaced += 1

        for j, dat in temp:     # update data with the replacement
            data[j] = dat
    print(f'There were {no_missing} missing points and {no_replaced} were replaced. Ratio: {no_replaced/no_missing}')
    return missing_list