import pandas as pd
import seaborn as sns
from utils.hist_tools import add_curr_occ, plot_window, get_reading


sns.set_theme(style="darkgrid")

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

# Assumptions of current occupancy:
#   - (resetting at 4 o'clock)
#   - never having negative occupancy
Occ = add_curr_occ(Occ, reset_hour=4)

# Combining
df = pd.concat([Occ["Current Occupancy"], CO2], axis=1).dropna()

plot_window(df, "2021-08-23", "2021-09-30")
plot_window(df, "2021-11-01", "2021-12-13")
plot_window(df, "2021-12-15")

# Using the new estimated current occupancy
# We can approximate the CO2 level for every zone
zero_level = df[df["Current Occupancy"] == 0].mean()[1:]

# There is a 3% measure accuracy
# https://www.connectedbaltics.com/wp-content/uploads/2017/12/AirWitsCO2_brochure-1pageENG.pdf
zero_level = pd.concat(
    [zero_level.rename("ppm"), (zero_level * 0.03).rename("acc. +/-")], axis=1
)
print("Mean ppm when occupancy is estimated to 0: ", zero_level)

# Get nearest reading of time witten as "yyyy-mm-dd hh:mm:ss"
timestamp, occupancy, co2 = get_reading(df, "2021-11-05 12:20:00")
print(
    f"Got the following readings at {str(timestamp)}: \n CO2: \n{co2}\n Occ: {float(occupancy)}"
)
