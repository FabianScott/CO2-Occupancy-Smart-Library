import numpy as np


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


def mass_balance(Ci, Q, V, n_total, n_map, C_out=400, alpha=0.05, time_step=5, m=20, decimals=0):
    """
    This function calculates the derivative of Ci, creates the replacement
    CO2 vector from the neighbour map (n_map) and calculates the estimated
    N by using the calculated N as a proportion of the total N

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
    dC = (Ci[0] - Ci[1]) / time_step
    Cr = [alpha * C_out + (1 - alpha) * np.average(Ci[0][n_map[el]]) for el in Ci[0]]
    N = (Q * (Ci[0] - Cr) + V * dC) / m
    N_estimate = N / np.sum(N) * n_total
    return N_estimate.round(decimals)


