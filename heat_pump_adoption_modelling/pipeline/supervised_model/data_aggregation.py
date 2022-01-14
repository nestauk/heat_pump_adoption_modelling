# File: heat_pump_adoption_modelling/pipeline/supervised_model/data_aggregation.py
"""
Aggregating data on postcode level.
"""

# ----------------------------------------------------------------------------------

# Import

import re
import pandas as pd
import numpy as np

from heat_pump_adoption_modelling.pipeline.encoding import feature_encoding

# ----------------------------------------------------------------------------------


def split_postcode(postcode):
    """Split postcode into two sections.
    For example: 'YO30' and '5QW' for YO30 5QW

    Parameters
    ----------
    postcode: str
        Original full postcode.

    Return
    ---------
    part_1: str
        First part of postcode.

    part_2: str
        Second part of postcode."""

    # Split postcode into two sections
    try:
        part_1, part_2 = re.findall(
            r"([A-Z]{1,2}[0-9]{1,2})\s*([0-9][A-Z]{1,2})", postcode
        )[0]

    # If not found
    except IndexError:
        return None

    return part_1, part_2


def get_postcode_area(postcode):
    """Get postcode area (highest postcode level).
    For example: YO for YO30 5QW

    Parameters
    ----------
    postcode: str
        Original full postcode.

    Return
    ---------
    postcode area : str"""

    # Split postcode into two sections
    split_pc = split_postcode(postcode)

    if split_pc is None:
        return np.nan

    # Get first letters
    part_1, _ = split_pc
    postcode_area = re.findall(r"([A-Z]+)", part_1)[0]

    return postcode_area


def get_postcode_district(postcode):
    """Get postcode area (2nd highest postcode level).
    For example: YO30 for YO30 5QW

    Parameters
    ----------
    postcode: str
        Original full postcode.

    Return
    ---------
    postcode district : str"""

    # Split postcode into two sections
    split_pc = split_postcode(postcode)

    if split_postcode is None:
        return np.nan

    # First section is equal two district
    part_1, _ = split_pc
    return part_1


def get_postcode_sector(postcode):
    """Get postcode sector (2nd lowest postcode level).
    For example: YO30 5 for YO30 5QW

    Parameters
    ----------
    postcode: str
        Original full postcode.

    Return
    ---------
    postcode sector : str"""

    # Split postcode into two sections
    split_pc = split_postcode(postcode)

    if split_postcode is None:
        return np.nan

    # Get first part + numbers of 2nd part
    part_1, part_2 = split_pc
    postcode_sector = part_1 + re.findall(r"[0-9]", part_2)[0]
    return postcode_sector


def get_postcode_levels(df, only_keep=None):
    """Get 4 different postcode levels: area, district, sector and unit.
    Optionally, only keep required postcode level.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with column "POSTCODE"

    only_keep: str, efault=None
        If postcode level is given, the other postcode levels will be removed.
        For instance, if "POSTCODE_DISTRICT" is given, remove
        columns "POSTCODE_ADEA/SECTOR/UNIT" from dataframe.

    Return
    ---------
    df : pandas.DataFrame with added postcode levels
    """

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


def aggreate_categorical_features(df, features, agglo_feature="POSTCODE_UNIT"):
    """Aggregate categorical features.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe including features of interest.

    features: list of strings
        Features of interest.

    agglo_feature: str, default=POSTCODE_UNIT
        Feature by which to agglomerate, for instance postcode level or hex id.


    Return
    ---------
    df: pandas. Dataframe
        Dataframe with agglomerated features."""

    aggregated_features = []

    # Remove feature by which to agglomerate from features of interest
    features = [f for f in features if f != agglo_feature]

    # For each feature of interest, get agglomerated data
    for feature in features:

        # Group e.g. by postcode level
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

        # Rename columns with feature name + value
        col_rename_dict = {
            col: feature + ": " + str(col)
            for col in cat_percentages_by_agglo_f.columns[1:-2]
        }
        cat_percentages_by_agglo_f.rename(columns=col_rename_dict, inplace=True)

        # Get the total
        aggregated_features.append(cat_percentages_by_agglo_f)

    # Concatenate and remove duplicates
    aggregated_features = pd.concat(aggregated_features, axis=1)
    aggregated_features = aggregated_features.loc[
        :, ~aggregated_features.columns.duplicated()
    ]

    # Save number of samples per group (e.g. postcode level)
    col = aggregated_features.pop(agglo_feature + "_TOTAL")
    aggregated_features.insert(1, col.name, col)

    return aggregated_features


def get_feature_count_grouped(df, feature, groupby_f, value=True, name=None):
    """Get the number of specific value of feature when grouped,
    for instance the number of installed heat pumps (value=True) per postcode level.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe including features of interest.

    feature: str
        Feature of interest.

    groupby_f: str
        Feature to group by, e.g. postcode level.

    value: str, default=True
        Specific value in given feature to count.
        By default, set to True since often used with boolean features.

    name: str, default=None
        How to rename the feature.
        If None, name is "feature name: True"

    Return
    ---------
    feature_cats_by_agglo_f: pandas.Dataframe
        Count of given feature value per group."""

    # Set default name
    if name is None:
        name = feature + ": True"

    # Get the feature categories by the agglomeration feature
    feature_cats_by_agglo_f = (
        df.groupby([groupby_f, feature]).size().unstack(fill_value=0)
    ).reset_index()

    # Rename the column since "True" is not a meaningful name
    feature_cats_by_agglo_f.rename(columns={value: name}, inplace=True)

    return feature_cats_by_agglo_f[[groupby_f, name]]


def encode_agglomerated_features(
    df, postcode_level, ordinal_features, unaltered_features
):
    """Encode the agglomerated features.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe with features to encode.

    postcode_level: str
        Postcode level on which to agglomerate.

    ordinal_features: list
        List of ordinal features.

    unaltered_features: list
        Features that should not be altered or encoded."

    Return
    ---------
    agglomerated_df : pandas.Dataframe
        Encoded agglomerated features.
    """

    # Get numeric features
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()

    # Agglomerate numeric features using median
    num_agglomerated = (
        df[numeric_features + [postcode_level]].groupby([postcode_level]).median()
    )

    # Reset index
    num_agglomerated = num_agglomerated.reset_index()

    # Get categorical features (not ordinal, numeric or unaltered)
    categorical_features = [
        feature
        for feature in df.columns
        if (feature not in ordinal_features)
        and (feature not in numeric_features)
        and (feature not in unaltered_features)
    ]

    # Aggreate categorical features
    cat_agglomerated = aggreate_categorical_features(
        df[categorical_features], categorical_features, agglo_feature=postcode_level
    )

    # Get the features indicating the most frequent values
    most_frequent = [col for col in cat_agglomerated.columns if "MOST_FREQUENT" in col]

    # One-hot encode categorical features
    cat_agglomerated = feature_encoding.one_hot_encoding(
        cat_agglomerated, most_frequent, verbose=True
    )

    # Concatenate the processed categorical, numeric and unaltered features
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
    """Get the target variables (current and future heat pump coverage and growth)
    based on source and target year data.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe with all year's data, used for normalising.

    source_year: pandas.Dataframe
        Dataframe with data for source year.

    target_year: pandas.Dataframe
        Dataframe with data for target year.

    postcode_level: str
        Postcode level to group by.

    normalise: str, default='total'
        How to normalise the number of properties per postcode.
        'total' : normalise by total number of properties
        'target' : normalise by number of properties at target year

    Return
    ---------
    target_var_df : pandas.DataFrame
        Target variable dataframe with heat pump coverage and growth information."""

    # Get the number of installed heat pumps per group (e.g. postcode level)
    n_hp_installed_source = get_feature_count_grouped(
        source_year, "HP_INSTALLED", postcode_level, name="# HP at Time t"
    )
    n_hp_installed_target = get_feature_count_grouped(
        target_year, "HP_INSTALLED", postcode_level, name="# HP at Time t+1"
    )
    n_hp_installed_total = get_feature_count_grouped(
        df, "HP_INSTALLED", postcode_level, name="Total # HP"
    )

    # Select basis for normalisation
    if normalise == "total":
        total_basis = df
    elif normalise == "target":
        total_basis = target_year
    else:
        raise IOError(
            "Normalisation type '{}' is not defined. Please use 'total' or 'target' instead.".format(
                normalise
            )
        )

    # Get the number of properties for each postcode
    n_prop_total = (
        total_basis.groupby([postcode_level]).size().reset_index(name="# Properties")
    )

    # Merge source and target years
    target_var_df = pd.merge(
        n_hp_installed_source, n_hp_installed_target, on=postcode_level
    )

    # Merge with number of properties
    target_var_df = pd.merge(target_var_df, n_prop_total, on=postcode_level)

    # Compute heat pump coverage and growth
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
