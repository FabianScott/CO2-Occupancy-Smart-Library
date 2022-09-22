import numpy as np
from copy import copy


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

    N_estimated = n_total * (Ci - Ci0) / sum(Ci - Ci0)
    if include_max:
        N_estimated = N_estimated * M / np.average(M)
    return N_estimated.round(decimals)


def mass_balance(Ci, Q, V, n_total, n_map=None, C_out=400, alpha=0.05, time_step=5, m=20, decimals=0, M=None):
    """
    This function calculates the derivative of Ci, creates the replacement
    CO2 vector from the neighbour map (n_map) and calculates the estimated
    N by using the calculated N as a proportion of the total N

    :param M:           vector of maximum capacity in each zone, if left blank is not considered
    :param Ci:          vector of CO2 data from current and previous time_step
    :param Q:           vector of airflow rates in each zone
    :param V:           vector of volumes of zones
    :param n_map:       dictionary where key is zone no and values are neighbouring zones
    :param m:           float CO2 exhaled per person
    :param decimals:    int of decimals to round to
    :param time_step:   float time between measurements
    :param alpha:       float/vector of proportion of outdoor air in exchange
    :param n_total:     int number of people in the entire library
    :param C_out:       float CO2 concentration outdoors
    :return:            vector of estimated number of people per zone
    """
    if n_map is None:
        n_map = {1: [2, 10], 2: [1, 3], 3: [2, 4], 4: [2, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 10],
                 10: [1, 9], 11: [12, 20], 12: [11, 13], 13: [12, 14], 14: [13, 15], 15: [14, 16], 16: [15, 17],
                 17: [16, 18], 18: [17, 19], 19: [18, 20], 20: [11, 19], 21: [22, 27], 22: [21, 23], 23: [22, 24],
                 24: [23, 25], 25: [24, 26], 26: [25, 27], 27: [21, 26]
                 }
    dC = (Ci[0] - Ci[1]) / time_step
    Cr = [alpha * C_out + (1 - alpha) * np.average(Ci[0][np.array(n_map[el])-1]) for el in range(1, 1 + len(Ci[0]))]
    N = (Q * (Ci[0] - Cr) + V * dC) / m
    N_estimated = N / np.sum(N) * n_total
    if M is not None:
        N_estimated = N_estimated * M / np.average(M)
    return N_estimated.round(decimals)


def optimise_mass_balance(C, n_total, Q, V, n_map=None, m_range=(5, 30), precision=0.1, C_out=400, alpha=0.05,
                          time_step=5, ):
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
