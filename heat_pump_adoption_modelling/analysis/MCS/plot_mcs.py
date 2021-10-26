import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.MCS.load_mcs import load_domestic_hps
from heat_pump_adoption_modelling.pipeline.MCS.mcs_epc_joining import join_mcs_epc_data


def plottable_dhps():
    dhps = load_domestic_hps()
    dhps["year"] = dhps.date.dt.year
    dhps_noexhaust = dhps[dhps.tech_type != "Exhaust Air Heat Pump"]

    return dhps_noexhaust


def plottable_merged(file_path=PROJECT_DIR / "outputs/mcs_epc.csv"):
    if file_path is None:
        merged = join_mcs_epc_data()
    else:
        merged = pd.read_csv(file_path)

    merged["year"] = pd.to_datetime(merged.date).dt.year

    return merged


# median installation cost per year
def plot_median_costs(df):

    cost_data = (
        df.groupby(["year", "tech_type"])["cost"].median().loc[2015:2021].unstack()
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
    ax.set_ylabel("Median cost of installation (£)")
    ax.legend(title="Heat pump type")
    ax.set_title(
        "Median cost of MCS certified heat pump installations\n(years with >100 installations only)"
    )

    plt.savefig(PROJECT_DIR / "outputs/figures/median_costs.png")


# numbers of installations each year
def plot_installation_numbers(df):

    numbers_data = df.groupby(["year", "tech_type"]).size().unstack().loc[2014:2021]

    fig, ax = plt.subplots()

    numbers_data.plot(kind="bar", ax=ax, zorder=3, color=["#1AC9E6", "#EB548C"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Installations")
    ax.grid(axis="y", color="0.8", zorder=0)
    ax.legend(title="Heat pump type")
    ax.set_title("Numbers of MCS certified heat pumps installed")
    plt.xticks(rotation=0)
    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "outputs/figures/installation_numbers.png")


def add_property_category(df):
    df["property_category"] = np.nan
    df.loc[
        [string in ["Flat", "Maisonette"] for string in df.PROPERTY_TYPE],
        "property_category",
    ] = "Flat/Maisonette"
    df.loc[
        [
            (string1 in ["House", "Bungalow"])
            & (
                string2
                in [
                    "Enclosed Mid-Terrace",
                    "Mid-Terrace",
                    "Enclosed End-Terrace",
                    "End-Terrace",
                ]
            )
            for string1, string2 in zip(df.PROPERTY_TYPE, df.BUILT_FORM)
        ],
        "property_category",
    ] = "Terraced House"
    df.loc[
        [
            (string1 in ["House", "Bungalow"])
            & (string2 in ["Detached", "Semi-Detached"])
            for string1, string2 in zip(df.PROPERTY_TYPE, df.BUILT_FORM)
        ],
        "property_category",
    ] = "Detached/Semi-Detached House"

    return df


c1 = "#142459"
c2 = "#ee9a3a"
c3 = "#820401"
c4 = "#a9a9a9"


def plot_property_medians(df, hp_type="Air Source Heat Pump"):
    if hp_type not in ["Air Source Heat Pump", "Ground/Water Source Heat Pump"]:
        print("invalid hp type")
    else:
        df = df[df.tech_type == hp_type]

        property_category_medians = (
            df.groupby(["year", "property_category"])["cost"]
            .median()
            .unstack()
            .loc[2016:2021]
        )

        fig, ax = plt.subplots()

        ax.plot(
            property_category_medians.index,
            property_category_medians["Detached/Semi-Detached House"].tolist(),
            label="Detached/Semi-Detached House",
            c=c1,
            marker="o",
            markersize=4,
        )
        ax.plot(
            property_category_medians.index,
            property_category_medians["Flat/Maisonette"].tolist(),
            label="Flat/Maisonette",
            c=c2,
            marker="o",
            markersize=4,
        )
        ax.plot(
            property_category_medians.index,
            property_category_medians["Terraced House"].tolist(),
            label="Terraced House",
            c=c3,
            marker="o",
            markersize=4,
        )
        ax.set_xticks(list(range(2016, 2022)))
        ax.grid(color="0.8")
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Year")
        ax.set_ylabel("Median cost of installation (£)")
        ax.legend(title="Dwelling type")
        short_dict = {
            "Air Source Heat Pump": "ASHP",
            "Ground/Water Source Heat Pump": "G/WSHP",
        }
        ax.set_title(
            "Median cost of MCS certified "
            + short_dict[hp_type]
            + " installations by dwelling type"
        )

        plt.savefig(
            "outputs/figures/property_medians_"
            + short_dict[hp_type].lower().replace("/", "").replace(" ", "")
            + ".png"
        )


def plot_property_counts(df, hp_type="Air Source Heat Pump"):
    if hp_type not in ["Air Source Heat Pump", "Ground/Water Source Heat Pump", "any"]:
        print("invalid hp type")
    else:
        if hp_type in ["Air Source Heat Pump", "Ground/Water Source Heat Pump"]:
            df = df[df.tech_type == hp_type]

        df["property_category"] = df["property_category"].fillna("Other/Unknown")

        property_category_counts = (
            df.groupby(["year", "property_category"])["cost"]
            .count()
            .unstack()
            .loc[2014:2021]
        )

        fig, ax = plt.subplots()

        ax.plot(
            property_category_counts.index,
            property_category_counts["Detached/Semi-Detached House"].tolist(),
            label="Detached/Semi-Detached House",
            marker="o",
            markersize=4,
            c=c1,
        )
        ax.plot(
            property_category_counts.index,
            property_category_counts["Flat/Maisonette"].tolist(),
            label="Flat/Maisonette",
            marker="o",
            markersize=4,
            c=c2,
        )
        ax.plot(
            property_category_counts.index,
            property_category_counts["Terraced House"].tolist(),
            label="Terraced House",
            marker="o",
            markersize=4,
            c=c3,
        )
        ax.plot(
            property_category_counts.index,
            property_category_counts["Other/Unknown"].tolist(),
            label="Other/Unknown",
            marker="o",
            markersize=4,
            c=c4,
        )
        ax.set_xticks(list(range(2014, 2022)))
        ax.set_ylim(bottom=0)
        ax.grid(color="0.8")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of installations")
        ax.legend(title="Dwelling type")
        short_dict = {
            "Air Source Heat Pump": "ASHP",
            "Ground/Water Source Heat Pump": "G/WSHP",
            "any": "heat pump",
        }
        ax.set_title(
            "Number of MCS certified "
            + short_dict[hp_type]
            + " installations by dwelling type"
        )
        plt.tight_layout()

        plt.savefig(
            "outputs/figures/property_counts_"
            + short_dict[hp_type].lower().replace("/", "").replace(" ", "")
            + ".png"
        )


#### Geographic distribution - local authorities
def regional_plot(df):
    df.groupby("local_authority").agg(
        number=("cost", lambda col: col.count()),
        median_cost=("cost", lambda col: col.median()),
    ).sort_values("median_cost", ascending=False)
    # tbc...


# some LAs with not many installations
# some non-standard local authority names -
# e.g. "Belfast City Council, Lisburn & Castlereagh..."


def main():
    """Main function: Plots MCS data."""
    dhps_noexhaust = plottable_dhps()
    plot_median_costs(dhps_noexhaust)
    plot_installation_numbers(dhps_noexhaust)

    merged = plottable_merged()
    merged = add_property_category(merged)
    plot_property_counts(merged)
    plot_property_counts(merged, hp_type="Ground/Water Source Heat Pump")
    plot_property_counts(merged, hp_type="any")
    plot_property_medians(merged)
    plot_property_medians(merged, hp_type="Ground/Water Source Heat Pump")


if __name__ == "__main__":
    # Execute only if run as a script
    main()
