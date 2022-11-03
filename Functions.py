import numpy as np
import pandas as pd
from copy import copy
from datetime import datetime
from datetime import timedelta
from scipy.stats import norm
from constants import id_map


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
        n_unknown = len(N_estimated)-np.count_nonzero(N_estimated)
        N_estimated[N_estimated != 0] -= n_unknown/(len(N_estimated)-n_unknown) * mean
        N_estimated[N_estimated == 0] = mean
    return N_estimated.round(decimals)


def mass_balance_helper(X):
    C, dC, Cr, Q, V, m = X
    N = (Q * (C - Cr) + V * dC) / m
    return N


def mass_balance(C, Q, V, n_total, current_time=[], n_map=None, C_out=420, alpha=0.7, time_step=5*60, m=20, decimals=0, M=None, fill_neighbours=False):
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


def get_replacement_and_derivative(C, C_out, time_step=5*60, alpha=0.05, n_map=None):
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


def string_to_datetime(t):
    t = t.replace('T', ':')[:-8]
    # Convert to datetime object
    return datetime.strptime(t, '%Y-%m-%d:%H:%M:%S')


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
    output = [0 if p == 0 else(1 if p < treshs[0] else (2 if p < treshs[1] else 3)) for p in percentage]

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


def exponential_moving_average(C, t, tau=900):
    """
    Given CO2 measurements with creation time, return the values smoothed
    based on the previous measurements. Time window can be specified.
    :param C:
    :param t:
    :param tau:
    :return:
    """
    t = np.array([el.seconds for el in (t[0] - t)])
    smoothed = [C[0] for _ in C]
    for j in range(1, len(t)):
        w = np.exp(-(t[j] - t[j - 1]) / tau)
        smoothed[j] = smoothed[j - 1] * w + C[j] * (1 - w)
    return smoothed


def log_likelihood(x, C, N, V, dt, uncertainty=50, percent=0.03):
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
    :return:
    """
    m, C_out, Q = x
    uncertainty, percent = uncertainty / 2, percent / 2  # it is the 95 % confidence, therefor 2 sd's
    Ci, N = C[:-1], N[1:]  # Remove first N, as there is no previous CO2

    C_est = (Q * dt * C_out + m * N * dt) / (Q * dt + V) + Ci
    # compare to C[1:] as there is no first estimate
    sd = np.array([max(uncertainty, el * percent) for el in C[1:]])
    log_l = sum(np.log(norm.pdf(C_est, loc=C[1:], scale=sd)))

    # since the function is doing optimisation, we want it to find the largest
    # number
    return log_l


def data_for_optimising(filename, newest_first=True, interval_smoothing=0):
    """
    Given the filename of a csv file with three columns, one with
    device id's, one with co2 measurements and one with time of
    measurement, return a list containing the measurements from
    each device where the index in the list corresponds to the zone
    number it is from.
    :param filename:
    :param newest_first:
    :param interval_smoothing:
    :return:
    """
    df = pd.read_csv(filename)
    time_index = np.argmax(df.columns == 'telemetry.time')
    co2_index = np.argmax(df.columns == 'telemetry.co2')
    id_index = np.argmax(df.columns == 'deviceId')
    # So indices correspond to zone number, the 0'th element will simply be empty
    device_list = [[] for _ in range(28)]

    first_time = df.values[0][time_index]

    for row in df.values:
        co2 = row[co2_index]
        time = string_to_datetime(row[time_index])
        device_list[id_map[row[id_index]]].append([time, co2])

    if not newest_first:
        for device in device_list:
            if device:
                device.sort(key=lambda x: (x[time_index]))

    if interval_smoothing:

        pass

    return device_list
