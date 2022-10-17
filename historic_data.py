import pandas as pd
from utils.hist_tools import plot_window, get_reading, make_co2_occ_df

df = make_co2_occ_df(
    CO2_path="data/CO2_data_library.csv",
    Occ_path="data/Occu_data_library.csv",
)

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

timestamp, occupancy, co2 = get_reading(df, "2021-11-05 12:20:00")
