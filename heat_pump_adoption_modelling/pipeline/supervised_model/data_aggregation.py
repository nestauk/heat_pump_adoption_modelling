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
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering
from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

# ----------------------------------------------------------------------------------

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)


IDENTIFIER = config["IDENTIFIER"]


def split_postcode_by_level(postcode, level, with_space=True):

    postcode = postcode.strip()

    seperation = " " if with_space else ""

    if level == "area":
        return re.findall(r"([A-Z]+)", postcode)[0]

    else:
        part_1 = postcode[:-3].strip()
        part_2 = postcode[-3:].strip()

        if level == "district":
            return part_1
        elif level == "sector":
            return part_1 + seperation + part_2[0]
        elif level == "unit":
            return part_1 + seperation + part_2
        else:
            raise IOError(
                "Postcode level '{}' unknown. Please select 'area', 'district', 'sector' or 'unit'.".format(
                    level
                )
            )


def get_postcode_levels(df, only_keep=None, with_space=True):
    """Get 4 different postcode levels: area, district, sector and unit.
    For example, given YO30 5QW:
    - area: YO
    - district: YO30
    - sector: YO30 5
    - unit: YO30 5QW

    Optionally, only keep required postcode level.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with column "POSTCODE"

    only_keep: str, default=None
        If postcode level is given, the other postcode levels will be removed.
        For instance, if "POSTCODE_DISTRICT" is given, remove
        columns "POSTCODE_ADEA/SECTOR/UNIT" from dataframe.

    Return
    ---------
    df : pandas.DataFrame with added postcode levels
    """

    levels = ["area", "sector", "district", "unit"]

    if only_keep is not None:
        levels = [only_keep]

    for level in levels:
        df["POSTCODE_" + level.upper()] = df["POSTCODE"].apply(
            split_postcode_by_level, level=level, with_space=with_space
        )

    return df


def aggregate_categorical_features(df, features, group_by_feature="POSTCODE_UNIT"):
    """Aggregate categorical features.

    First split the data into groups, e.g. based on postcode level, and get the number of samples.
    For each categorial feature, get the percentage of each category/value for every group.
    Also retrieve the most frequent category for every feature.

    For example: Postcode AB10 1QS has 8 samples. Feature TENURE is split in the following categories:
    12.5% private rental, 37.5% social rental and 50% owner-occupied properties.
    Thus, the most freuquent category for TENURE is owner-occupied.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe including features of interest.

    features: list of strings
        Features of interest.

    group_by_feature: str, default=POSTCODE_UNIT
        Feature by which to aggregate, for instance postcode level or hex id.


    Return
    ---------
    df: pandas. Dataframe
        Dataframe with aggregated features."""

    aggregated_features = []

    # Remove feature by which to aggregate from features of interest
    features = [f for f in features if f != group_by_feature]

    # For each feature of interest, get aggregated data
    for feature in features:

        # Group e.g. by postcode level
        grouped_by_aggr_f = df.groupby(group_by_feature)

        # Count the samples per aggregation feature category
        n_samples_agglo_cat = grouped_by_aggr_f[feature].count()

        # Get the feature categories by the aggregation feature
        feature_cats_by_aggr_f = (
            df.groupby([group_by_feature, feature]).size().unstack(fill_value=0)
        )

        # Normalise by the total number of samples per category
        cat_percentages_by_aggr_f = (feature_cats_by_aggr_f.T / n_samples_agglo_cat).T

        # Get the most frequent feature cateogry
        cat_percentages_by_aggr_f[
            "MOST_FREQUENT_" + feature
        ] = cat_percentages_by_aggr_f.idxmax(axis=1)

        # Totals
        cat_percentages_by_aggr_f[group_by_feature + "_TOTAL"] = n_samples_agglo_cat

        # Reset index
        cat_percentages_by_aggr_f = cat_percentages_by_aggr_f.reset_index()

        # Rename columns with feature name + value
        col_rename_dict = {
            col: feature + ": " + str(col)
            for col in cat_percentages_by_aggr_f.columns[1:-2]
        }
        cat_percentages_by_aggr_f.rename(columns=col_rename_dict, inplace=True)

        # Get the total
        aggregated_features.append(cat_percentages_by_aggr_f)

    # Concatenate and remove duplicates
    aggregated_features = pd.concat(aggregated_features, axis=1)
    aggregated_features = aggregated_features.loc[
        :, ~aggregated_features.columns.duplicated()
    ]

    # Save number of samples per group (e.g. postcode level)
    col = aggregated_features.pop(group_by_feature + "_TOTAL")
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
    feature_cats_by_aggr_f: pandas.Dataframe
        Count of given feature value per group."""

    # Set default name
    if name is None:
        name = feature + ": " + value

    # Get the feature categories by the aggregation feature
    feature_cats_by_aggr_f = (
        df.groupby([groupby_f, feature]).size().unstack(fill_value=0)
    ).reset_index()

    # Rename the column since "True" is not a meaningful name
    feature_cats_by_aggr_f.rename(columns={value: name}, inplace=True)

    return feature_cats_by_aggr_f[[groupby_f, name]]


def aggregate_and_encode_features(
    df, postcode_level, ordinal_features, drop_features=[]
):
    """Aggregated features on postcode level and encode them.
    For numeric features, the median of all postcode level properties is computed.
    For categorical feature, the percentage of category/value per postcode level is computed.
    In addition, the most frequent category/value is detected and one-hot encoded.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe with features to encode.

    postcode_level: str
        Postcode level on which to aggregate.

    ordinal_features: list
        List of ordinal features.

    drop_features: list, default=[]
        Features that should not be aggregated and are dropped."

    Return
    ---------
    aggregated_df : pandas.Dataframe
        Encoded aggregated features."""

    # Get numeric features
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()

    # Agglomerate numeric features using median
    num_aggregated = (
        df[numeric_features + [postcode_level]].groupby(postcode_level).median()
    )

    # Reset index
    num_aggregated = num_aggregated.reset_index()

    # Get categorical features (not ordinal, numeric or unaltered)
    categorical_features = [
        feature
        for feature in df.columns
        if (feature not in ordinal_features)
        and (feature not in numeric_features)
        and (feature not in drop_features)
    ]

    # Aggreate categorical features
    cat_aggregated = aggregate_categorical_features(
        df[categorical_features], categorical_features, group_by_feature=postcode_level
    )

    # Get the features indicating the most frequent values
    most_frequent = [col for col in cat_aggregated.columns if "MOST_FREQUENT" in col]

    # One-hot encode categorical "most frequent" features
    cat_aggregated = feature_encoding.one_hot_encoding(
        cat_aggregated, most_frequent, verbose=True
    )

    # Concatenate the processed categorical, numeric and unaltered features
    aggregated_df = pd.concat(
        [cat_aggregated, num_aggregated.drop(columns=[postcode_level])],
        axis=1,
    )
    return aggregated_df


def get_target_variables(
    df, source_year, target_year, postcode_level, normalize_by="total"
):
    """Get the target variables (current and future heat pump coverage and growth)
    based on source and target year data.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe with all year's data, used for normalising.
        Duplicates will be filtered out.

    source_year: pandas.Dataframe
        Dataframe with data up to source year.

    target_year: pandas.Dataframe
        Dataframe with data up to target year.

    postcode_level: str
        Postcode level to group by.

    normalize_by: str, default='total'
        How to normalize the number of properties per postcode.
        'total' : normalize by total number of properties
        'target' : normalize by number of properties by target year
        'total HPs' : normalize by total number of HPs in postcode properties
                      at latest entry time

    Return
    ---------
    target_var_df : pandas.DataFrame
        Target variable dataframe with heat pump coverage and growth information."""

    # Get the number of installed heat pumps per group (e.g. postcode level)
    n_hp_installed_source = get_feature_count_grouped(
        source_year, "HP_INSTALLED", postcode_level, name="# HP by source year"
    )
    n_hp_installed_target = get_feature_count_grouped(
        target_year, "HP_INSTALLED", postcode_level, name="# HP by target year"
    )

    dedupl_df = feature_engineering.filter_by_year(
        df, IDENTIFIER, None, selection="latest entry"
    )

    n_hp_installed_total = get_feature_count_grouped(
        dedupl_df, "HP_INSTALLED", postcode_level, name="Total # HP"
    )

    # Select basis for normalisation
    if normalize_by in ["total", "total HPs"]:
        total_basis = dedupl_df
        norm_feature = "# Properties" if normalize_by == "total" else "Total # HP"

    elif normalize_by == "target":
        total_basis = target_year
        norm_feature = "# Properties"
    else:
        raise IOError(
            "Normalisation type '{}' is not defined. Please use 'total' or 'target' instead.".format(
                normalize_by
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

    if normalize_by == "total HPs":
        print("get here")
        target_var_df = pd.merge(target_var_df, n_hp_installed_total, on=postcode_level)

    # Compute heat pump coverage and growth
    target_var_df["HP_COVERAGE_CURRENT"] = (
        target_var_df["# HP by source year"] / target_var_df[norm_feature]
    )
    target_var_df["HP_COVERAGE_FUTURE"] = (
        target_var_df["# HP by target year"] / target_var_df[norm_feature]
    )
    target_var_df["GROWTH"] = (
        target_var_df["HP_COVERAGE_FUTURE"] - target_var_df["HP_COVERAGE_CURRENT"]
    )
    return target_var_df[
        ["HP_COVERAGE_CURRENT", "HP_COVERAGE_FUTURE", "GROWTH", postcode_level]
    ]
