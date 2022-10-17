import datetime
from tracemalloc import start
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


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
    datetime = pd.to_datetime(datetime)
    loc = df.index.get_indexer([datetime], method="nearest")
    timestamp = df.index[loc]
    occupancy = df.iloc[loc, 0]
    co2 = df.iloc[loc, 1:]

    return timestamp, occupancy, co2
