import re
import pandas as pd
import numpy as np

from heat_pump_adoption_modelling.pipeline.encoding import feature_encoding


def get_postcode_area(postcode):
    try:
        part_1, _ = re.findall(r"([A-Z]{1,2}[0-9]{1,2})\s*([0-9][A-Z]{1,2})", postcode)[
            0
        ]
    except IndexError:
        return np.nan
    postcode_area = re.findall(r"([A-Z]+)", part_1)[0]

    return postcode_area


def get_postcode_district(postcode):

    try:
        part_1, _ = re.findall(r"([A-Z]{1,2}[0-9]{1,2})\s*([0-9][A-Z]{1,2})", postcode)[
            0
        ]
    except IndexError:
        return np.nan
    return part_1


def get_postcode_sector(postcode):

    try:
        part_1, part_2 = re.findall(
            r"([A-Z]{1,2}[0-9]{1,2})\s*([0-9][A-Z]{1,2})", postcode
        )[0]
    except IndexError:
        return np.nan
    postcode_sector = part_1 + re.findall(r"[0-9]", part_2)[0]
    return postcode_sector


def get_postcode_levels(df, only_keep=None):

    df["POSTCODE_AREA"] = df["POSTCODE"].apply(get_postcode_area)
    df["POSTCODE_DISTRICT"] = df["POSTCODE"].apply(get_postcode_district)
    df["POSTCODE_SECTOR"] = df["POSTCODE"].apply(get_postcode_sector)
    df["POSTCODE_UNIT"] = df["POSTCODE"]

    if only_keep is not None:
        postcode_columns = [
            "POSTCODE_AREA",
            "POSTCODE_DISTRICT",
            "POSTCODE_SECTOR",
            "POSTCODE_UNIT",
            "POSTCODE",
        ]
        postcode_columns.remove(only_keep)
        df = df.drop(columns=postcode_columns)

    return df


def aggreate_categorical_features(df, features, agglo_feature="hex_id"):

    aggregated_features = []
    features = [f for f in features if f != agglo_feature]

    for feature in features:

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

        # Totals
        cat_percentages_by_agglo_f[agglo_feature + "_TOTAL"] = n_samples_agglo_cat

        # Reset index
        cat_percentages_by_agglo_f = cat_percentages_by_agglo_f.reset_index()

        # Rename columns
        col_rename_dict = {
            col: feature + ": " + str(col)
            for col in cat_percentages_by_agglo_f.columns[1:-2]
        }
        cat_percentages_by_agglo_f.rename(columns=col_rename_dict, inplace=True)

        # Get the total
        aggregated_features.append(cat_percentages_by_agglo_f)

    # Get the total
    aggregated_features = pd.concat(aggregated_features, axis=1)
    aggregated_features = aggregated_features.loc[
        :, ~aggregated_features.columns.duplicated()
    ]

    col = aggregated_features.pop(agglo_feature + "_TOTAL")
    aggregated_features.insert(1, col.name, col)

    return aggregated_features


def get_feature_count_grouped(
    df, feature, groupby_f, name=None
):  # Get the feature categories by the agglomeration feature

    if name is None:
        name = feature + ": True"

    feature_cats_by_agglo_f = (
        df.groupby([groupby_f, feature]).size().unstack(fill_value=0)
    ).reset_index()

    feature_cats_by_agglo_f.rename(columns={True: name}, inplace=True)

    return feature_cats_by_agglo_f[[groupby_f, name]]


def encode_agglomerated_features(
    df, postcode_level, ordinal_features, unaltered_features
):

    numeric_features = df.select_dtypes(include=np.number).columns.tolist()

    num_agglomerated = (
        df[numeric_features + [postcode_level]].groupby([postcode_level]).median()
    )
    num_agglomerated = num_agglomerated.reset_index()

    categorical_features = [
        feature
        for feature in df.columns
        if (feature not in ordinal_features)
        and (feature not in numeric_features)
        and (feature not in unaltered_features)
    ]

    cat_agglomerated = aggreate_categorical_features(
        df[categorical_features], categorical_features, agglo_feature=postcode_level
    )

    most_frequent = [col for col in cat_agglomerated.columns if "MOST_FREQUENT" in col]

    cat_agglomerated = feature_encoding.one_hot_encoding(
        cat_agglomerated, most_frequent, verbose=True
    )

    agglomerated_df = pd.concat(
        [
            cat_agglomerated,
            num_agglomerated.drop(columns=[postcode_level]),
            df[unaltered_features],
        ],
        axis=1,
    )

    return agglomerated_df


def get_target_variables(
    df, source_year, target_year, postcode_level, normalise="total"
):

    n_hp_installed_source = get_feature_count_grouped(
        source_year, "HP_INSTALLED", postcode_level, name="# HP at Time t"
    )
    n_hp_installed_target = get_feature_count_grouped(
        target_year, "HP_INSTALLED", postcode_level, name="# HP at Time t+1"
    )
    n_hp_installed_total = get_feature_count_grouped(
        df, "HP_INSTALLED", postcode_level, name="Total # HP"
    )

    if normalise == "total":
        total_basis = df
    else:
        total_basis = target_year

    n_prop_total = (
        total_basis.groupby([postcode_level]).size().reset_index(name="# Properties")
    )

    target_var_df = pd.merge(
        n_hp_installed_source, n_hp_installed_target, on=postcode_level
    )
    target_var_df = pd.merge(target_var_df, n_prop_total, on=postcode_level)

    target_var_df["HP_COVERAGE_CURRENT"] = (
        target_var_df["# HP at Time t"] / target_var_df["# Properties"]
    )
    target_var_df["HP_COVERAGE_FUTURE"] = (
        target_var_df["# HP at Time t+1"] / target_var_df["# Properties"]
    )
    target_var_df["GROWTH"] = (
        target_var_df["HP_COVERAGE_FUTURE"] - target_var_df["HP_COVERAGE_CURRENT"]
    )

    return target_var_df
