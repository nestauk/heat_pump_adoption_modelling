# File: heat_pump_adoption_modelling/analysis/MCS/plot_mcs.py
"""Produce plots relating to heat pump costs/capacities:
- Median costs by year and tech type
- Mean ASHP SCOP by year and flow temp
- Costs by ASHP capacity
- ASHP capacities by property archetype
- ASHP costs by property archetype

Current draft of paper can be found here: https://docs.google.com/document/d/1Es-eWt5kLTbsZRtazi23IbpaPIdPDe29juqWGnAGKHc/edit?usp=sharing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.getters.load_mcs import load_inflation
from heat_pump_adoption_modelling.pipeline.preprocessing.mcs_epc_joining import (
    join_mcs_epc_data,
)

# Load data for plotting


def plottable_data(file_path=PROJECT_DIR / "outputs/mcs_epc.csv"):
    """Load merged MCS/EPC data in a suitable form for plotting
    (with additional variables: "built_post_1950" indicating
    whether or not the property was built after 1950, and
    "inflated_cost" indicating inflation-adjusted installation cost).

    Parameters
    ----------
    file_path : str
        Path to saved merged data.
        If none, data is loaded from scratch.

    Return
    ----------
    merged : pd.DataFrame
        Merged MCS-EPC data with added variables.
    """

    if file_path is None:
        merged = join_mcs_epc_data()
    else:
        merged = pd.read_csv(file_path)

    # Add inflated cost variable
    inflation = load_inflation()
    merged = merged.merge(inflation, how="left", on="year")
    merged["inflated_cost"] = merged["cost"] * merged["multiplier"]

    # Add indicator variable for properties built after 1950
    ages_dict = {
        "England and Wales: before 1900": False,
        "1900-1929": False,
        "1930-1949": False,
        "1950-1966": True,
        "1965-1975": True,
        "1976-1983": True,
        "1983-1991": True,
        "1991-1998": True,
        "1996-2002": True,
        "2003-2007": True,
        "2007 onwards": True,
        "unknown": pd.NA,
        np.nan: pd.NA,
        None: pd.NA,
    }

    merged["built_post_1950"] = merged["CONSTRUCTION_AGE_BAND"].apply(
        lambda x: ages_dict[x]
    )

    return merged


# Define functions to assist with "archetype" boxplots


def bool_not(series):
    """Given a series of bools and NAs, return a series
    with NAs in the same places but bools inverted.

    Parameters
    ----------
    series : pd.Series
        A series of bools and NAs.

    Return
    ----------
    result : pd.Series
        A series with NAs in the same places and bools inverted.
    """

    # Initialise with a series of NAs
    result = pd.Series(np.repeat(pd.NA, len(series)))

    result.loc[series == True] = False
    result.loc[series == False] = True

    return result


def generate_archetype_dict(df):
    """Given a dataframe, produce a dictionary of archetype labels
    to readable names and logical vectors indicating the rows of
    the dataframe that conform to that archetype.
    Archetypes are defined in the top table of sheet "DAv2" here:
    https://docs.google.com/spreadsheets/d/1HkUYneexFXDTgMyJs_UaUbBmXGUum6I1K4L2GP2rEjQ/edit?usp=sharing

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with "PROPERTY_TYPE", "BUILT_FORM" and "built_post_1950"
        variables.

    Return
    ----------
    result : dict
        A dictionary from archetype numbers to names and logical vectors.
    """

    archetype_dict = {
        "archetype_1": {
            "name": "Post-1950 flats",
            "condition": (df.PROPERTY_TYPE == "Flat") & (df.built_post_1950),
        },
        "archetype_2": {
            "name": "Pre-1950 flats",
            "condition": (df.PROPERTY_TYPE == "Flat") & bool_not(df.built_post_1950),
        },
        "archetype_3": {
            "name": "Post-1950 bungalows",
            "condition": (df.PROPERTY_TYPE == "Bungalow") & (df.built_post_1950),
        },
        "archetype_4": {
            "name": "Post-1950 semi-detached,\nterraced and maisonettes",
            "condition": (
                (df.PROPERTY_TYPE == "Maisonette")
                | (
                    (df.PROPERTY_TYPE == "House")
                    & (
                        df.BUILT_FORM.isin(
                            [
                                "Semi-Detached",
                                "Mid-Terrace",
                                "End-Terrace",
                                "Enclosed End-Terrace",
                                "Enclosed Mid-Terrace",
                            ]
                        )
                    )
                )
            )
            & (df.built_post_1950),
        },
        "archetype_5": {
            "name": "Post-1950 detached",
            "condition": (
                (df.PROPERTY_TYPE == "House")
                & (df.BUILT_FORM == "Detached")
                & (df.built_post_1950)
            ),
        },
        "archetype_6": {
            "name": "Pre-1950 semi-detached,\nterraced and maisonettes",
            "condition": (
                (df.PROPERTY_TYPE == "Maisonette")
                | (
                    (df.PROPERTY_TYPE == "House")
                    & (
                        df.BUILT_FORM.isin(
                            ["Semi-Detached", "Mid-Terrace", "End-Terrace"]
                        )
                    )
                )
            )
            & bool_not(df.built_post_1950),
        },
        "archetype_7": {
            "name": "Pre-1950 bungalows",
            "condition": (df.PROPERTY_TYPE == "Bungalow")
            & bool_not(df.built_post_1950),
        },
        "archetype_8": {
            "name": "Pre-1950 detached",
            "condition": (
                (df.PROPERTY_TYPE == "House")
                & (df.BUILT_FORM == "Detached")
                & bool_not(df.built_post_1950)
            ),
        },
    }

    return archetype_dict


#### PLOTS


def plot_median_costs(df):
    """Plot time series of median inflation-adjusted
    installation costs over time (from 2015 as sample sizes
    before this are reasonably small) for both AS and GSHPs.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing "year", "tech_type" and "inflated_cost"
        variables.

    Return
    ----------
        No objects returned, but figure is saved as median_costs.png.
    """

    cost_data = (
        df.groupby(["year", "tech_type"])["inflated_cost"]
        .median()
        .loc[2015:2021]
        .unstack()
    )

    fig, ax = plt.subplots()

    ax.plot(
        cost_data.index,
        cost_data["Ground/Water Source Heat Pump"].tolist(),
        label="Ground/Water Source Heat Pump",
        c="#EB548C",
        marker="o",
        markersize=4,
    )

    ax.plot(
        cost_data.index,
        cost_data["Air Source Heat Pump"].tolist(),
        label="Air Source Heat Pump",
        c="#1AC9E6",
        marker="o",
        markersize=4,
    )

    ax.set_ylim(0, 22500)
    ax.grid(axis="y", color="0.8")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median inflation-adjusted cost of installation (£2021)")
    ax.legend(title="Heat pump type")
    ax.set_title(
        "Median inflation-adjusted cost of\nMCS certified heat pump installations"
    )

    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "outputs/figures/median_costs.png")


def scop_trend_plot(df):
    """Plot time series of mean SCOP of installed heat pumps
    for flow temperatures of 35, 40, 45, 50, 55 (the most
    common flow temperatures) over time (2016 onwards to ensure
    sample sizes are sufficient).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing "year", "flow_temp" and "scop" variables.

    Return
    ----------
        No objects returned, but figure is saved as mean_scop.png.
    """

    flow_temps = [35, 40, 45, 50, 55]

    # Filter to relevant data
    df = df.loc[
        df["flow_temp"].isin(flow_temps)
        & ~df["scop"].isna()
        & (df["year"] > 2015)
        & (df["tech_type"] == "Air Source Heat Pump")
    ]

    # Group and process data
    plotting_df = (
        df.groupby(["year", "flow_temp"])["scop"].agg(["mean", "count"]).unstack()
    )

    fig, ax = plt.subplots()

    colours = ["purple", "blue", "green", "orange", "red"]

    # Plot line for each flow temp
    # Commented section adds sample sizes as labels;
    # excluded from current version of plot
    for i in range(0, 5):
        ax.plot(
            plotting_df.index,
            plotting_df["mean"][flow_temps[i]],
            label=flow_temps[i],
            c=colours[i],
        )
        # for year in plotting_df.index:
        #     ax.annotate(
        #         text=plotting_df["count"].loc[year, flow_temps[i]],
        #         xy=(year - 0.23, plotting_df["mean"].loc[year, flow_temps[i]] + 0.02),
        #         c=colours[i],
        #     )

    ax.set_ylim(3, 4.2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean SCOP")
    ax.set_title("Mean SCOP of installed MCS-certified ASHPs over time")
    ax.grid(color="0.8")

    # Put a legend to the right of the current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(
        title="Flow temp. ($\degree$C)", loc="center left", bbox_to_anchor=(1, 0.5)
    )

    plt.savefig("outputs/figures/mean_scop.png")


def ashp_capacity_cost_boxplot(df):
    """Produce boxplot of ASHP inflation-adjusted installation costs
    vs their capacities for capacities between 5 and 16 inclusive
    (the most common capacities), only considering data from 2019 onwards
    (the most reliable cost data).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing "year", "tech_type", "capacity" and
        "inflated_cost" variables.

    Return
    ----------
        No objects returned, but figure is saved as capacity_cost_boxplots.png.
    """

    # Filter to only recent and ASHP data
    plottable_data = df.loc[
        (df["year"] >= 2019) & (df["tech_type"] == "Air Source Heat Pump")
    ]

    capacity_list = range(5, 17)

    # Form a list of lists of cost data for each capacity
    data_list = []
    for capacity in capacity_list:
        filtered_df = plottable_data.loc[plottable_data["capacity"] == capacity]
        costs = list(filtered_df["inflated_cost"].dropna())
        data_list.append(costs)

    fig, ax = plt.subplots()

    ax.boxplot(
        data_list,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "black",
            "alpha": 0.3,
        },
    )

    ax.grid(axis="y", color="0.8")
    ax.set_ylim(0, 50000)
    ax.set_xlabel("HP capacity")
    ax.set_ylabel("Inflation-adjusted cost (£2021)")
    ax.set_title("MCS-certified ASHP installation costs by capacity (2019-21)")

    # Add labels to x axis
    plt.setp(ax, xticklabels=capacity_list)
    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "outputs/figures/capacity_cost_boxplots.png")


def ashp_archetypes_boxplot(df, variable):
    """Produce boxplot of ASHP inflation-adjusted installation costs
    for each archetype (only considering data from 2019 onwards).

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing "year", "tech_type",
        "PROPERTY_TYPE", "BUILT_FORM" and "built_post_1950" variables,
        as well as the variable specified in the "variable" parameter.
    variable : str
        Tested with either "inflated_cost" or "capacity".

    Return
    ----------
        No objects returned, but figure is saved as ashp_cost_archetype_boxplots.png
        if variable is "inflated_cost" and ashp_capacity_archetype_boxplots.png if
        variable is "capacity".
    """

    ashp_data_list = []

    archetype_dict = generate_archetype_dict(df)
    # off_scale = 0
    # total = 0

    for info in archetype_dict.values():
        filtered_df = df.loc[
            (df["year"] >= 2019)
            & (df["tech_type"] == "Air Source Heat Pump")
            & info["condition"]
        ]
        values = list(filtered_df[variable].dropna())
        ashp_data_list.append(values)
        # off_scale += sum([value > 30000 for value in values])
        # total += len(values)

    ashp_data_list = ashp_data_list[::-1]

    fig, ax = plt.subplots()

    ax.boxplot(
        ashp_data_list,
        vert=False,
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "black",
            "alpha": 0.3,
        },
    )

    if variable == "inflated_cost":
        ax.set_xlim(0, 30000)
        ax.set_xlabel("Inflation-adjusted cost of installation (£2021)")
        ax.set_ylabel("Property type")
        ax.set_title("MCS-certified ASHP installation costs by property type (2019-21)")
    elif variable == "capacity":
        ax.set_xlim(0, 30)
        ax.set_xlabel("ASHP capacity")
        ax.set_ylabel("Property type")
        ax.set_title("MCS-certified ASHP capacities by property type (2019-21)")

    ax.grid(axis="x", color="0.8")

    # percentage_off_scale = round(off_scale / total * 100, 2)

    # plt.figtext(0.99, 0.01,
    #     "Only 2019-21 data used due to increased data quality post-2019.\n{}% of points lie outside axis limits.".format(
    #         percentage_off_scale
    #     ), horizontalalignment="right", fontsize = "small"
    # )

    # Add labels to y axis (requires reversing the list as labels are added
    # from bottom to top)
    plt.setp(ax, yticklabels=[info["name"] for info in archetype_dict.values()][::-1])
    plt.tight_layout()

    if variable == "inflated_cost":
        plt.savefig(PROJECT_DIR / "outputs/figures/ashp_cost_archetype_boxplots.png")
    elif variable == "capacity":
        plt.savefig(
            PROJECT_DIR / "outputs/figures/ashp_capacity_archetype_boxplots.png"
        )


# ---------------------------------------------------------------------------------


def main(file_path=PROJECT_DIR / "outputs/mcs_epc.csv"):
    """Main function: Plots MCS-EPC data.

    Parameters
    ----------
    file_path : str
        Path to merged MCS-EPC data.
        If None, data is loaded from scratch.

    Return
    ----------
        No objects returned, but figures are saved as specified in
        individual functions.
    """

    print("Loading data...")
    data = plottable_data(file_path=file_path)
    print("Plotting...")
    plot_median_costs(data)
    scop_trend_plot(data)
    ashp_capacity_cost_boxplot(data)
    ashp_archetypes_boxplot(data, "inflated_cost")
    ashp_archetypes_boxplot(data, "capacity")


if __name__ == "__main__":
    # Execute only if run as a script
    main()
