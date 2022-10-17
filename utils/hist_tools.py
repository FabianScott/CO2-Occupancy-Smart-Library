import datetime
from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def make_co2_occ_df(
    CO2_path: str = "data/CO2_data_library.csv",
    Occ_path: str = "data/Occu_data_library.csv",
) -> pd.DataFrame:
    # Read CO2 Level
    CO2 = pd.read_csv(CO2_path, sep=";", parse_dates=["Timestamp"])

    # Group data by every 30 minute (Fill rows) and drop nans. Dropna is reducing n_rows from 6608 -> 6334 which is 137 hours of missing data
    CO2 = CO2.groupby(pd.Grouper(key="Timestamp", freq="30min")).mean().dropna()

    #  Lets look at the CO2 level for a specific time window
    time_mask = (CO2.index.hour >= 8) & (CO2.index.hour <= 9)
    CO2[time_mask]

    # Adding occupancy
    Occ = pd.read_csv(Occ_path, sep=";", parse_dates=["Timestamp"])

    # Sum main entrance and bookstore
    Occ = Occ.groupby(Occ.Timestamp).sum()
    Occ["change"] = Occ["in"] - Occ["out"]

    # Assumptions of current occupancy:
    #   - (resetting at 4 o'clock)
    #   - never having negative occupancy
    Occ = add_curr_occ(Occ, reset_hour=4)

    # Combining
    df = pd.concat([Occ["Current Occupancy"], CO2], axis=1).dropna()

    return df


def curr_occ(timestamp, before, change, reset_time):
    """
    :param before:              what the count was at previous timestamp
    :param change:              change of count
    :param reset_time:          timestamp where the current amout of people in library is reset to zero. Has to be an integer for 24 hour clock
    """
    if timestamp.hour == reset_time:
        return 0
    else:
        current = before + change
        if current < 0:
            current = 0
        return current


def add_curr_occ(df: pd.DataFrame, reset_hour: int = 4) -> pd.DataFrame:
    occ = np.zeros(len(df))
    occ[0] = df.iloc[: 17 + (2 * 4)][
        "out"
    ].sum()  # The change between 16:00 and 0X:00 first day
    for i in range(1, len(occ)):
        occ[i] = curr_occ(
            df.index[i], occ[i - 1], df.change.iloc[i], reset_time=reset_hour
        )

    df["Current Occupancy"] = occ

    return df


def nomalize_df(df):
    return (df - df.min()) / (df.max() - df.min())


def plot_window(df, start_time=None, end_time=None, normalize=False):
    sns.set_theme(style="darkgrid")
    if start_time == None:
        start_time = df.index[0]  # Just a early year

    if end_time == None:
        end_time = df.index[-1]

    time_mask = (df.index >= start_time) & (df.index <= end_time)

    df = df[time_mask]
    x = df.index

    y1 = df.iloc[:, 1:].mean(axis=1)
    y2 = df["Current Occupancy"]

    if normalize == True:
        y1 = nomalize_df(y1)
        y2 = nomalize_df(y2)

    fig, ax1 = plt.subplots(dpi=600, figsize=(15, 6))
    ax2 = ax1.twinx()
    ax1.plot(x, y1, "g-")
    ax2.plot(x, y2, "b-")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("CO2 [ppm]", color="g")
    ax2.set_ylabel("Occupancy [no. people]", color="b")
    time_int = f"{str(start_time).split(' ')[0]}-{str(end_time).split(' ')[0]}"
    plt.title(f"Time period: {time_int}")
    plt.savefig(f"docs/plots/{time_int}.png")


def get_reading(df, datetime):
    """
    :param df:          datafram where first column is current occupancy and the rest is co2 level from the different censors
    :param datetime:    time witten as "yyyy-mm-dd hh:mm:ss"

    :return timestamp:  nearest timestamp
    :return occupancy:  occupancy level at nearest timestamp
    :co2:               co2 level for each zone at nearest timestamp
    """
    datetime = pd.to_datetime(datetime)
    loc = df.index.get_indexer([datetime], method="nearest")
    timestamp = df.index[loc]
    occupancy = df.iloc[loc, 0]
    co2 = df.iloc[loc, 1:]

    print(
        f"Got the following readings at {str(timestamp)}: \n CO2: \n{co2}\n Occ: {float(occupancy)}"
    )

    return timestamp, occupancy, co2
