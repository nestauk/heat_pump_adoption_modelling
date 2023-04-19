# File: getters/data_cleaning.py
"""Data auglomeration using H3

Created August 2021
@author: Julia Suter
Last updated on 05/08/2021
"""


import h3
import pandas as pd
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering


def add_hex_id(df, resolution=7.5):
    """Get H3 hex ID based on coordinates.

     Parameters
     ----------
    row : pandas.Series
        Dataframe row.

    resolution : int, default=7
        H3 resolution.

    Return
    ---------

        H3 index: int"""

    df["hex_id"] = df.apply(
        lambda row: h3.geo_to_h3(row["LATITUDE"], row["LONGITUDE"], resolution), axis=1
    )

    return df


def map_rating_to_cat(rating):

    if rating < 2.0:
        return "F-G"
    elif rating >= 2.0 and rating < 3.0:
        return "E-F"
    elif rating >= 3.0 and rating < 4.0:
        return "D-E"
    elif rating >= 4.0 and rating < 5.0:
        return "C-D"
    elif rating >= 5.0 and rating < 6.0:
        return "B-C"
    elif rating >= 6.0 and rating < 7.0:
        return "A-B"


def get_cat_distr_by_hex(df, feature, agglo_feature="hex_id"):

    # Group the data by HEX ID
    grouped_by_hex_id = df.groupby(agglo_feature)
    n_samples_per_hex = grouped_by_hex_id[feature].count()

    cats_by_hex = df.groupby([agglo_feature, feature]).size().unstack(fill_value=0)

    cat_percentages_by_hex = (cats_by_hex.T / n_samples_per_hex).T

    cat_percentages_by_hex[agglo_feature] = n_samples_per_hex

    cat_percentages_by_hex = cat_percentages_by_hex.reset_index()

    return cat_percentages_by_hex


def get_cat_distr_grouped_by_agglo_f(df, feature, agglo_feature="hex_id"):

    # Group the data by agglomeration feature (e.g. hex id)
    grouped_by_agglo_f = df.groupby(agglo_feature)

    # Count the samples per agglomeration feature category
    n_samples_agglo_cat = grouped_by_agglo_f[feature].count()

    # Get the feature categories by the agglomeration feature
    feature_cats_by_agglo_f = (
        df.groupby([agglo_feature, feature]).size().unstack(fill_value=0)
    )

    # Normalise by the total number of samples per category
    cat_percentages_by_agglo_f = (feature_cats_by_agglo_f.T / n_samples_agglo_cat).T

    # Get the most frequent feature cateogry
    cat_percentages_by_agglo_f[
        "MOST_FREQUENT_" + feature
    ] = cat_percentages_by_agglo_f.idxmax(axis=1)

    # Get the total
    cat_percentages_by_agglo_f[agglo_feature + "_TOTAL"] = n_samples_agglo_cat

    # Reset index for easier processing
    cat_percentages_by_agglo_f = cat_percentages_by_agglo_f.reset_index()

    return cat_percentages_by_agglo_f


def get_agglomerated_features(
    df, feature=None, agglo_feature="hex_id", year_stamp=None
):

    has_new_feature = False

    if feature not in [None, "CURRENT_ENERGY_RATING", "HP_INSTALLED"]:
        has_new_feature = True
        grouped_by_agglo_f = get_cat_distr_grouped_by_agglo_f(
            df, feature, agglo_feature=agglo_feature
        )

    epc_grouped_by_agglo_f = get_cat_distr_grouped_by_agglo_f(
        df, "CURRENT_ENERGY_RATING", agglo_feature=agglo_feature
    )

    epc_grouped_by_agglo_f["HEX_RATING"] = (
        1 * epc_grouped_by_agglo_f["G"]
        + 2 * epc_grouped_by_agglo_f["F"]
        + 3 * epc_grouped_by_agglo_f["E"]
        + 4 * epc_grouped_by_agglo_f["D"]
        + 5 * epc_grouped_by_agglo_f["C"]
        + 6 * epc_grouped_by_agglo_f["B"]
        + 7 * epc_grouped_by_agglo_f["A"]
    )

    epc_grouped_by_agglo_f["EPC_CAT"] = epc_grouped_by_agglo_f["HEX_RATING"].apply(
        feature_engineering.map_rating_to_cat
    )

    epc_grouped_by_agglo_f = epc_grouped_by_agglo_f[
        [
            agglo_feature,
            agglo_feature + "_TOTAL",
            "EPC_CAT",
            "HEX_RATING",
            "MOST_FREQUENT_CURRENT_ENERGY_RATING",
        ]
    ]

    epc_grouped_by_agglo_f.rename(
        columns={agglo_feature + "_TOTAL": "density"}, inplace=True
    )

    if has_new_feature:

        epc_grouped_by_agglo_f = pd.merge(
            grouped_by_agglo_f, epc_grouped_by_agglo_f, on=[agglo_feature]
        )

    hp_grouped_by_agglo_f = get_cat_distr_grouped_by_agglo_f(
        df, "HP_INSTALLED", agglo_feature=agglo_feature
    )

    print(hp_grouped_by_agglo_f.head())

    hp_grouped_by_agglo_f["HP_PERC"] = hp_grouped_by_agglo_f[True] * 100
    del hp_grouped_by_agglo_f[False]
    del hp_grouped_by_agglo_f[True]

    hp_grouped_by_agglo_f["HP_CAT"] = hp_grouped_by_agglo_f["HP_PERC"].apply(HP_cat)

    hp_grouped_by_agglo_f = hp_grouped_by_agglo_f[
        [agglo_feature, "HP_CAT", "HP_PERC", "MOST_FREQUENT_HP_INSTALLED"]
    ]

    grouped_by_agglo_f = pd.merge(
        epc_grouped_by_agglo_f, hp_grouped_by_agglo_f, on=[agglo_feature]
    )

    if year_stamp is not None:
        # print(year)

        # year_stamp = str(year) + "/01/01 00:00"
        grouped_by_agglo_f["YEAR_STAMP"] = year_stamp

    if "CO2_EMISSIONS_CURRENT" in df.columns:

        emissions_mean = df.groupby(agglo_feature)["CO2_EMISSIONS_CURRENT"].mean()
        grouped_by_agglo_f = pd.merge(
            grouped_by_agglo_f, emissions_mean, on=[agglo_feature]
        )

    if "HEATING_COST_CURRENT" in df.columns:
        costs_mean = df.groupby(agglo_feature)["HEATING_COST_CURRENT"].mean()
        grouped_by_agglo_f = pd.merge(
            grouped_by_agglo_f, costs_mean, on=[agglo_feature]
        )

    return grouped_by_agglo_f


def map_hex_to_feature(df, feature):

    hex_to_feature = get_cat_distr_grouped_by_agglo_f(
        df, feature, agglo_feature="hex_id"
    )[["hex_id", "MOST_FREQUENT_LOCAL_AUTHORITY_LABEL"]]

    hex_to_feature = hex_to_feature.rename(
        columns={"MOST_FREQUENT_" + feature: feature}
    )

    return hex_to_feature


# ------- Old STUFF ---------


def get_HP_uptake(df):

    # Any feature would do (instead of HAS_HP)
    counts_per_hex = df.groupby("hex_id")["HP_INSTALLED"].count()

    has_hp_per_hex = df.groupby(["hex_id", "HP_INSTALLED"]).size().unstack(fill_value=0)
    hp_df = (has_hp_per_hex.T / counts_per_hex).T
    hp_df["total"] = counts_per_hex

    hp_df = hp_df.reset_index()
    hp_df.columns = ["hex_id", "HP_INSTALLED_False", "heat_pump_fract", "density"]

    hp_df["heat_pump_perc"] = hp_df["heat_pump_fract"] * 100
    hp_df = hp_df[["hex_id", "heat_pump_perc", "density"]]

    return hp_df


def get_EPC_cat(epc_df):

    counts_per_hex = epc_df.groupby("hex_id")["CURRENT_ENERGY_RATING"].count()

    epc_per_hex = (
        epc_df.groupby(["hex_id", "CURRENT_ENERGY_RATING"]).size().unstack(fill_value=0)
    )
    epc_cats_df = (epc_per_hex.T / counts_per_hex).T
    epc_cats_df["total"] = counts_per_hex
    epc_cats_df = epc_cats_df.reset_index()

    epc_cats_df["HEX RATING"] = (
        1 * epc_cats_df["G"]
        + 2 * epc_cats_df["F"]
        + 3 * epc_cats_df["E"]
        + 4 * epc_cats_df["D"]
        + 5 * epc_cats_df["C"]
        + 6 * epc_cats_df["B"]
        + 7 * epc_cats_df["A"]
    )

    epc_cats_df["EPC_CAT"] = epc_cats_df["HEX RATING"].apply(
        feature_engineering.map_rating_to_cat
    )

    return epc_cats_df


def HP_cat(perc):

    if isinstance(perc, str):
        return "0.0 %"

    elif perc <= 0.0:
        return "0.0 %"

    elif perc > 0.0 and perc <= 0.5:
        return "0.0 - 0.5 %"

    elif perc > 0.5 and perc <= 2.0:
        return "0.5 - 2.0 %"

    elif perc > 2.0 and perc <= 5.0:
        return "2.0 - 5.0 %"

    elif perc > 5.0 and perc <= 7:
        return "5.0 % +"


# elif perc > 7.0 and perc <= 10:
#    return "7.0 - 10.0 %"

# elif perc > 10.0 and perc <= 20.0:
#     return "10.0 - 20.0 %"

# elif perc > 20.0 and perc <= 100.0:
#     return "20.0 - 100.0 %"


"""
sorter= ['0.0 %', '0.0 - 0.5 %', '0.5 - 2.0 %', '2.0 - 5.0 %', '5.0 - 10.0 %', '10.0 - 20.0 %',
       '20.0 - 100.0 %']

time_data["sort"] = pd.Categorical(time_data["HP_CAT"], categories = sorter)

time_data.sort_values(by = "sort", inplace=True)
time_data.reset_index(drop=True, inplace=True)
"""
