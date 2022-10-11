import pandas as pd
import datetime
import numpy as np

# Read CO2 Level
CO2 = pd.read_csv("data/CO2_data_library.csv", sep=";", parse_dates=["Timestamp"])

# Group data by every 30 minute (Fill rows) and drop nans. Dropna is reducing n_rows from 6608 -> 6334 which is 137 hours of missing data
CO2 = CO2.groupby(pd.Grouper(key="Timestamp", freq="30min")).mean().dropna()

#  Lets look at the CO2 level for a specific time window
time_mask = (CO2.index.hour >= 8) & (CO2.index.hour <= 9)
CO2[time_mask]

# Adding occupancy
Occ = pd.read_csv("data/Occu_data_library.csv", sep=";", parse_dates=["Timestamp"])

# Sum main entrance and bookstore
Occ = Occ.groupby(Occ.Timestamp).sum()
Occ["change"] = Occ["in"] - Occ["out"]


def current_people(timestamp, before, change, reset_time=4):
    """
    :param before:              what the count was at previous timestamp
    :param change:              change of count
    :param reset_time:          timestamp where the current amout of people in library is reset to zero. Has to be an integer for 24 hour clock
    """
    if timestamp.hour in reset_time:
        return 0
    else:
        return before + change


occ = np.zeros(len(Occ))
occ[0] = 200  # The change between 16:00 and 03:00 first day
for i in range(1, len(occ)):
    occ[i] = current_people(
        Occ.index[i], occ[i - 1], Occ.change.iloc[i], reset_time=range(4)
    )

Occ["current"] = occ

print((Occ["current"] < 0).sum())
