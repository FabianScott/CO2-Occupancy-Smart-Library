import numpy as np
import pandas as pd
from copy import copy
from datetime import datetime
from datetime import timedelta


def basic_weighting(Ci, Ci0, n_total, decimals=0, include_max=False, M=None):
    """
    Takes the vectors of CO2 and baseline CO2, then applies the
    simple weighting of the occupancy in the zones from the total
    number of occupants. Can also include the weighting based on
    maximum occupants per zone as specified by M. This method appears
    very prone to rounding errors
    :param Ci: vector of current CO2
    :param Ci0: vector of baseline CO2
    :param n_total: integer of total occupants
    :param decimals: number of decimals for rounding
    :param include_max: bool
    :param M: vector of maximum occupancy per zone
    :return:
    """
    Ci.flatten()
    Ci0.flatten()
    N_estimated = n_total * (Ci - Ci0) / sum(Ci - Ci0)
    if include_max:
        N_estimated = N_estimated * M / np.average(M)
    return N_estimated.round(decimals)


def mass_balance(C, Q, V, n_total, n_map=None, C_out=400, alpha=0.05, time_step=5, m=20, decimals=0, M=None):
    """
    This function calculates the derivative of Ci, creates the replacement
    CO2 vector from the neighbour map (n_map) and calculates the estimated
    N by using the calculated N as a proportion of the total N

    :param C:           (27,2) vector of CO2 data from current and previous time_step
    :param Q:           (27,1) vector of airflow rates in each zone
    :param V:           (27,1) vector of volumes of zones
    :param n_map:       dictionary where key is zone no and values are neighbouring zones
    :param m:           float CO2 exhaled per person
    :param M:           (27,1) vector of maximum capacity in each zone, if left blank is not considered
    :param decimals:    int of decimals to round to
    :param time_step:   float time between measurements
    :param alpha:       float/vector of proportion of outdoor air in exchange
    :param n_total:     int number of people in the entire library
    :param C_out:       float CO2 concentration outdoors
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
    dC = (C[:, 0] - C[:, 1]) / time_step
    Cr = [alpha * C_out + (1 - alpha) * np.average(C[:, 0][np.array(n_map[el]) - 1]) for el in range(1, 1 + len(C[:, 0]))]
    N = (Q * (C[:, 0] - Cr) + V * dC) / m
    N_estimated = N / np.sum(N) * n_total
    if M is not None:
        M.flatten()
        N_estimated = N_estimated * M / np.average(M)
    return N_estimated.round(decimals)


def optimise_mass_balance_m(C, n_total, Q, V, n_map=None, m_range=(5, 30), precision=0.1, C_out=400, alpha=0.05,
                            time_step=5):
    """
    Given a set of CO2 measurements in each of the 27 zones and the total n
    at the corresponding time, this function finds the m within a given range
    which results in the error (N_est-N_tot)^2 being minimised.
    As of 22/10, running on random numbers results in the largest m possible
    as the calculated N is an order of 10^2 off calculating the correct number
    of people.

    :param C:           (n x 27) matrix of CO2 measurements
    :param n_total:     (27 x 1) vector of true total people count
    :param precision:   step between potential m's
    :param Q:           vector of airflow
    :param V:           vector of volumes
    :param n_map:       dict of value: zones neighbouring key: zone
    :param m_range:     range of potential m's
    :param C_out:       int of CO2 concentration outdoors
    :param alpha:       proportion of air from outdoors
    :param time_step:   time between measurements
    :return:            m which minimises the error
    """
    assert m_range[0] > 0
    if n_map is None:
        n_map = {1: [2, 10], 2: [1, 3], 3: [2, 4], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10],
                 10: [1, 9], 11: [12, 20], 12: [11, 13], 13: [12, 14], 14: [13, 15], 15: [14, 16], 16: [15, 17],
                 17: [16, 18], 18: [17, 19], 19: [18, 20], 20: [11, 19], 21: [22, 27], 22: [21, 23], 23: [22, 24],
                 24: [23, 25], 25: [24, 26], 26: [25, 27], 27: [21, 26]
                 }

    best_m, lowest_cost = 0, np.inf
    dC = (C[:-1] - C[1:]) / time_step
    Cr = []
    for i, row in enumerate(C):
        if i + 1 < len(C):
            Cr.append(np.array([alpha * C_out + (1 - alpha) * np.average(row[np.array(n_map[i]) - 1]) for i in
                                range(1, len(row) + 1)]))

    Cr = np.array(Cr)
    for m in range(m_range[0], m_range[1] + 1, precision):

        N = (Q * (C[1:] - Cr) + V * dC) / m
        err = np.sum((np.sum(N, axis=1) - n_total[:-1]) ** 2)
        if err < lowest_cost:
            lowest_cost = copy(err)
            best_m = m

    return best_m


def optimise_mass_balance_Q(C, n_total, Q, V, m=20, n_map=None, learning_rate=0.001, C_out=400, alpha=0.05,
                            time_step=5):
    """
    Given a set of CO2 measurements in each of the 27 zones and the total n
    at the corresponding time, this function finds the Q which results in the
    smallest error (N_est-N_tot)^2. One constraint is of course it being a
    positive number. Currently overflows numerically, not to be used as is.

    :param m:           int co2 exhaled per person
    :param C:           (n x 27) matrix of CO2 measurements
    :param n_total:     (27 x 1) vector of true total people count
    :param learning_rate:   step between potential m's
    :param Q:           vector of airflow
    :param V:           vector of volumes
    :param n_map:       dict of value: zones neighbouring key: zone
    :param C_out:       int of CO2 concentration outdoors
    :param alpha:       proportion of air from outdoors
    :param time_step:   time between measurements
    :return:            m which minimises the error
    """
    if n_map is None:
        n_map = {1: [2, 10], 2: [1, 3], 3: [2, 4], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10],
                 10: [1, 9], 11: [12, 20], 12: [11, 13], 13: [12, 14], 14: [13, 15], 15: [14, 16], 16: [15, 17],
                 17: [16, 18], 18: [17, 19], 19: [18, 20], 20: [11, 19], 21: [22, 27], 22: [21, 23], 23: [22, 24],
                 24: [23, 25], 25: [24, 26], 26: [25, 27], 27: [21, 26]
                 }

    # Calculate vectors from CO2:
    dC = (C[:-1] - C[1:]) / time_step
    Cr = []
    for i, row in enumerate(C):
        if i + 1 < len(C):
            Cr.append(np.array([alpha * C_out + (1 - alpha) * np.average(row[np.array(n_map[i]) - 1]) for i in
                                range(1, len(row) + 1)]))
    Cr = np.array(Cr)

    # Optimisation begins, for every data point
    for i, n in enumerate(n_total[:-1]):
        c = C[i]
        cr = Cr[i]
        dc = dC[i]
        n_est = (Q * (c - cr) + V * dc) / m
        err_base = np.sum((np.sum(n_est) - n) ** 2)
        grad = np.zeros(27)
        for j, q in enumerate(Q):
            Q_temp = copy(Q)
            Q_temp[j] = q*(1 - learning_rate)
            n_temp = (Q_temp * (c - cr) + V * dc) / m
            err_temp = np.sum((np.sum(n_temp) - n) ** 2)
            print(err_temp-err_base)
            grad[j] = (err_temp - err_base) / learning_rate
        # print(Q)
        Q = np.array([q - q*grad[j] if q - q*grad[j] > 0 else q for j, q in enumerate(Q)])
    return Q


def process_data(df, minutes, time_index=0, id_index=1):
    """
    Function to call to format the data so it can be used in Python.
    First job is to convert the time string into datetime object
    Second one is mapping device ID's to zone numbers
    Third one is removing too old data points
    Fourth one is sorting based on time
    :param df:              pandas dataframe of data from sql server
    :param minutes:         now - minutes is threshold for new data
    :param time_index:      column index of time variables
    :param id_index:        column index of device ID's
    :return: data:          numpy array of data in correct format
    """

    data = np.array(df.values)
    id_map = {'DA00110043': 1, 'DA00110044': 2, 'DA00110045': 3, 'DA00130002': 4, 'DA00130001': 5, 'DA00110031': 6,
              'DA00110041': 7, 'DA00110047': 8, 'DA00110035': 9, 'DA00110049': 10, 'DA00110032': 11, 'AMNO-03': 12,
              'DA00130004': 13, 'DA00110033': 14, 'AMNO-04': 15, 'DA00110037': 16, 'DA00130003': 17, 'AMNO-01': 18,
              'AMNO-02': 19, 'DA00110040': 20, 'DA00100001': 21, 'DA00110036': 22, 'DA00110034': 23, 'DA00120001': 24,
              'DA00110039': 25, 'DA00110042': 26, 'DA00110038': 27, }     # Must be verified
    # Convert all time strings to datetime to perform maths on them
    for i, t in enumerate(data[:, time_index]):
        t = t.replace('T', ':')[:-8]
        # Convert to datetime object
        data[i, time_index] = datetime.strptime(t, '%Y-%m-%d:%H:%M:%S')
        # Convert to number corresponding to zone
        data[i, id_index] = id_map[data[i, id_index]]

    # Remove rows where time is before specified time
    time_cutoff = datetime.now() - timedelta(minutes=minutes)
    data = data[data[:, time_index] > time_cutoff]
    # Sort by date so newest is at the top
    data = np.flip(data[data[:, 0].argsort()], axis=0)

    return data


def update_data(new_data, old_data, id_index=1, co2_index=2):
    """
    Given the new rows of data in the processed format find the
    devices that have produced new co2 data. If they have produced
    two outputs these will be stored in order. The old_data is then
    supplanted in the entries where there is new data.
    :param new_data:    (nx3) matrix of new data points
    :param old_data:    (27x2) matrix of data used in the previous iteration
    :param id_index:    int of the index in new data where the zone ID's are stored
    :param co2_index:   int of the index in new data where the co2 data is stored
    :return: output     (27x2) matrix of the most up to date data available
    """
    output = np.zeros((27, 2))
    for row in new_data:
        # If no data from the current device has been seen yet, input it in the first column of output
        device_id = row[id_index] - 1   # 0 indexed arrays
        if not output[device_id, 0]:
            output[device_id, 0] = row[co2_index]
        # If there is data in the first column, input the data in the second column
        elif not output[device_id, 1]:
            output[device_id, 1] = row[co2_index]
    output[output == 0] = old_data[output == 0]

    return output
