import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.MCS.load_mcs import load_inflation
from heat_pump_adoption_modelling.pipeline.MCS.mcs_epc_joining import join_mcs_epc_data


# /load in merged df
# /merge with inflation (check?)
# /line: median inflation-adjusted cost over time (2015-21) by tech
# /line: mean scop over time (2016-21) by flow temp (35-55)
# /v. boxplot: inf-adj cost by hp capacity (5-12, 14, 16), 2019-21 data
# /add archetypes
# /h. boxplot: capacities by archetype (2019-21 data)
# /h. boxplot: inf-adj costs by archetype (2019-21 data)


def plottable_data(file_path=PROJECT_DIR / "outputs/mcs_epc.csv"):

    if file_path is None:
        merged = join_mcs_epc_data()
    else:
        merged = pd.read_csv(file_path)

    inflation = load_inflation()
    merged = merged.merge(inflation, how="left", on="year")
    merged["inflated_cost"] = merged["cost"] * merged["multiplier"]

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


def plot_median_costs(df):

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

    flow_temps = [35, 40, 45, 50, 55]

    # Filter to relevant data
    df = df.loc[df.flow_temp.isin(flow_temps) & ~df.scop.isna() & (df.year > 2015)]

    # Group and process data
    plotting_df = df.groupby(["year", "flow_temp"])["scop"].agg("mean").unstack()

    fig, ax = plt.subplots()

    colours = ["purple", "blue", "green", "orange", "red"]

    for i in range(0, 5):
        ax.plot(
            plotting_df.index,
            plotting_df[flow_temps[i]],
            label=flow_temps[i],
            c=colours[i],
        )

    ax.set_ylim(3, 4.4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean SCOP")
    ax.set_title("Mean SCOP of installed heat pumps over time")
    ax.grid(color="0.8")
    ax.legend(title="Flow temp. ($\degree$C)")

    plt.savefig("outputs/figures/scop.png")


def ashp_capacity_cost_boxplot(df):

    # Filter to only recent and ASHP data
    plottable_data = df.loc[
        (df.year >= 2019) & (df.tech_type == "Air Source Heat Pump")
    ]

    capacity_list = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16]

    data_list = []
    for capacity in capacity_list:
        filtered_df = plottable_data.loc[plottable_data.capacity == capacity]
        costs = list(filtered_df.inflated_cost.dropna())
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

    ax.set_ylim(0, 50000)
    ax.set_xlabel("HP capacity")
    ax.set_ylabel("Inflation-adjusted cost (£2021)")
    ax.set_title("HP installation costs by capacity (2019-21)")

    plt.setp(ax, xticklabels=capacity_list)
    plt.tight_layout()

    plt.savefig(PROJECT_DIR / "outputs/figures/capacity_cost_boxplots.png")


def bool_not(series):

    result = pd.Series(np.repeat(pd.NA, len(series)))

    result.loc[series == True] = False
    result.loc[series == False] = True

    return result


def generate_archetype_dict(df):

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
                            ["Semi-Detached", "Mid-Terrace", "End-Terrace"]
                        )
                    )
                )  # enclosed??
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


def ashp_archetypes_boxplot(df, variable):

    ashp_data_list = []

    archetype_dict = generate_archetype_dict(df)

    for info in archetype_dict.values():
        filtered_df = df.loc[
            (df.year >= 2019)
            & (df.tech_type == "Air Source Heat Pump")
            & info["condition"]
        ]
        values = list(filtered_df[variable].dropna())
        ashp_data_list.append(values)

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
        ax.set_title("ASHP installation costs by property type (2019-21)")
    elif variable == "capacity":
        ax.set_xlim(0, 30)
        ax.set_xlabel("ASHP capacity")
        ax.set_ylabel("Property type")
        ax.set_title("ASHP capacities by property type (2019-21)")

    plt.setp(ax, yticklabels=[info["name"] for info in archetype_dict.values()][::-1])
    plt.tight_layout()

    if variable == "inflated_cost":
        plt.savefig(PROJECT_DIR / "outputs/figures/ashp_cost_archetype_boxplots.png")
    elif variable == "capacity":
        plt.savefig(
            PROJECT_DIR / "outputs/figures/ashp_capacity_archetype_boxplots.png"
        )


def main():
    """Main function: Plots MCS data."""
    data = plottable_data()
    plot_median_costs(data)
    scop_trend_plot(data)
    ashp_capacity_cost_boxplot(data)
    ashp_archetypes_boxplot(data, "inflated_cost")
    ashp_archetypes_boxplot(data, "capacity")


if __name__ == "__main__":
    # Execute only if run as a script
    main()


# # numbers of installations each year
# def plot_installation_numbers(df):

#     numbers_data = df.groupby(["year", "tech_type"]).size().unstack().loc[2014:2021]

#     fig, ax = plt.subplots()

#     numbers_data.plot(kind="bar", ax=ax, zorder=3, color=["#1AC9E6", "#EB548C"])
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Installations")
#     ax.grid(axis="y", color="0.8", zorder=0)
#     ax.legend(title="Heat pump type")
#     ax.set_title("Numbers of MCS certified heat pumps installed")
#     plt.xticks(rotation=0)
#     plt.tight_layout()

#     plt.savefig(PROJECT_DIR / "outputs/figures/installation_numbers.png")


# def add_property_category(df):
#     df["property_category"] = np.nan
#     df.loc[
#         [string in ["Flat", "Maisonette"] for string in df.PROPERTY_TYPE],
#         "property_category",
#     ] = "Flat/Maisonette"
#     df.loc[
#         [
#             (string1 in ["House", "Bungalow"])
#             & (
#                 string2
#                 in [
#                     "Enclosed Mid-Terrace",
#                     "Mid-Terrace",
#                     "Enclosed End-Terrace",
#                     "End-Terrace",
#                 ]
#             )
#             for string1, string2 in zip(df.PROPERTY_TYPE, df.BUILT_FORM)
#         ],
#         "property_category",
#     ] = "Terraced House"
#     df.loc[
#         [
#             (string1 in ["House", "Bungalow"])
#             & (string2 in ["Detached", "Semi-Detached"])
#             for string1, string2 in zip(df.PROPERTY_TYPE, df.BUILT_FORM)
#         ],
#         "property_category",
#     ] = "Detached/Semi-Detached House"
#     return df


# c1 = "#142459"
# c2 = "#ee9a3a"
# c3 = "#820401"
# c4 = "#a9a9a9"


# def plot_property_medians(df, hp_type="Air Source Heat Pump", inflate=True):
#     if hp_type not in ["Air Source Heat Pump", "Ground/Water Source Heat Pump"]:
#         print("invalid hp type")
#     else:
#         df = df[df.tech_type == hp_type]
#         if inflate:
#             variable = "inflated_cost"
#             y_label = "Median inflation-adjusted cost of installation (£2021)"
#         else:
#             variable = "cost"
#             y_label = "Median cost of installation (£)"
#         property_category_medians = (
#             df.groupby(["year", "property_category"])["cost"]
#             .median()
#             .unstack()
#             .loc[2016:2021]
#         )

#         fig, ax = plt.subplots()

#         ax.plot(
#             property_category_medians.index,
#             property_category_medians["Detached/Semi-Detached House"].tolist(),
#             label="Detached/Semi-Detached House",
#             c=c1,
#             marker="o",
#             markersize=4,
#         )
#         ax.plot(
#             property_category_medians.index,
#             property_category_medians["Flat/Maisonette"].tolist(),
#             label="Flat/Maisonette",
#             c=c2,
#             marker="o",
#             markersize=4,
#         )
#         ax.plot(
#             property_category_medians.index,
#             property_category_medians["Terraced House"].tolist(),
#             label="Terraced House",
#             c=c3,
#             marker="o",
#             markersize=4,
#         )
#         ax.set_xticks(list(range(2016, 2022)))
#         ax.grid(color="0.8")
#         ax.set_ylim(bottom=0)
#         ax.set_xlabel("Year")
#         ax.set_ylabel(y_label)
#         ax.legend(title="Dwelling type", loc="lower right")
#         short_dict = {
#             "Air Source Heat Pump": "ASHP",
#             "Ground/Water Source Heat Pump": "G/WSHP",
#         }
#         if inflate:
#             ax.set_title(
#                 "Median inflation-adjusted cost of MCS certified\n"
#                 + short_dict[hp_type]
#                 + " installations by dwelling type"
#             )
#         else:
#             ax.set_title(
#                 "Median cost of MCS certified "
#                 + short_dict[hp_type]
#                 + " installations by dwelling type"
#             )

#         plt.savefig(
#             "outputs/figures/property_medians_"
#             + short_dict[hp_type].lower().replace("/", "").replace(" ", "")
#             + ".png"
#         )


# def plot_property_counts(df, hp_type="Air Source Heat Pump"):
#     if hp_type not in ["Air Source Heat Pump", "Ground/Water Source Heat Pump", "any"]:
#         print("invalid hp type")
#     else:
#         if hp_type in ["Air Source Heat Pump", "Ground/Water Source Heat Pump"]:
#             df = df[df.tech_type == hp_type]

#         df["property_category"] = df["property_category"].fillna("Other/Unknown")

#         property_category_counts = (
#             df.groupby(["year", "property_category"])["cost"]
#             .count()
#             .unstack()
#             .loc[2014:2021]
#         )

#         fig, ax = plt.subplots()

#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Detached/Semi-Detached House"].tolist(),
#             label="Detached/Semi-Detached House",
#             marker="o",
#             markersize=4,
#             c=c1,
#         )
#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Flat/Maisonette"].tolist(),
#             label="Flat/Maisonette",
#             marker="o",
#             markersize=4,
#             c=c2,
#         )
#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Terraced House"].tolist(),
#             label="Terraced House",
#             marker="o",
#             markersize=4,
#             c=c3,
#         )
#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Other/Unknown"].tolist(),
#             label="Other/Unknown",
#             marker="o",
#             markersize=4,
#             c=c4,
#         )
#         ax.set_xticks(list(range(2014, 2022)))
#         ax.set_ylim(bottom=0)
#         ax.grid(color="0.8")
#         ax.set_xlabel("Year")
#         ax.set_ylabel("Number of installations")
#         ax.legend(title="Dwelling type")
#         short_dict = {
#             "Air Source Heat Pump": "ASHP",
#             "Ground/Water Source Heat Pump": "G/WSHP",
#             "any": "heat pump",
#         }
#         ax.set_title(
#             "Number of MCS certified "
#             + short_dict[hp_type]
#             + " installations by dwelling type"
#         )
#         plt.tight_layout()

#         plt.savefig(
#             "outputs/figures/property_counts_"
#             + short_dict[hp_type].lower().replace("/", "").replace(" ", "")
#             + ".png"
#         )


# #### Geographic distribution - local authorities
# def regional_plot(df):
#     df.groupby("local_authority").agg(
#         number=("cost", lambda col: col.count()),
#         median_cost=("cost", lambda col: col.median()),
#     ).sort_values("median_cost", ascending=False)
#     # tbc...


# # overall plots


# df=merged
# col_list = ['red', 'green', 'blue']
# fig, ax = plt.subplots()
# ax.set_ylim(0, 100000)
# for year in range(2019, 2022):
#     df_filtered = df.loc[(df.year == year) & (df.capacity < 30) & (df.inflated_cost < 100000), ["capacity", "inflated_cost"]].dropna()
#     ax.scatter(df_filtered.capacity, df_filtered.inflated_cost, c=col_list[year-2019], s=2, alpha=0.01)
#     ax.plot(np.unique(df_filtered.capacity), np.poly1d(np.polyfit(df_filtered.capacity, df_filtered.inflated_cost, 1))(np.unique(df_filtered.capacity)), c=col_list[year-2019])

# fig, ax = plt.subplots()
# ax.set_ylim(0, 30000)
# for year in range(2019, 2022):
#     df_filtered = df.loc[(df.year == year) & (df.TOTAL_FLOOR_AREA < 400) & (df.inflated_cost < 30000), ["TOTAL_FLOOR_AREA", "inflated_cost"]].dropna()
#     ax.scatter(df_filtered.TOTAL_FLOOR_AREA, df_filtered.inflated_cost, c=col_list[year-2019], s=2, alpha=0.01)
#     ax.plot(np.unique(df_filtered.TOTAL_FLOOR_AREA), np.poly1d(np.polyfit(df_filtered.TOTAL_FLOOR_AREA, df_filtered.inflated_cost, 1))(np.unique(df_filtered.TOTAL_FLOOR_AREA)), c=col_list[year-2019])


# indiv_sums = dhps.groupby(['product_id', 'flow_temp', 'scop']).size().reset_index().rename(columns={0:'indiv_counts'})
# product_sums = dhps.groupby('product_id').size().reset_index().rename(columns={0:'total'})

# test = indiv_sums.merge(product_sums, on='product_id')
# test = test.set_index(['product_id', 'flow_temp', 'scop'])
# test.sort_values('total', ascending=False)


# some LAs with not many installations
# some non-standard local authority names -
# e.g. "Belfast City Council, Lisburn & Castlereagh..."
