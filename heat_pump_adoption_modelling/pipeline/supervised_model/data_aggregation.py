import re
import pandas as pd
import numpy as np


def get_postcode_area(postcode):

    # print(postcode)
    try:
        part_1, _ = re.findall(r"([A-Z]{1,2}[0-9]{1,2})\s*([0-9][A-Z]{1,2})", postcode)[
            0
        ]
    except IndexError:
        return np.nan
    postcode_area = re.findall(r"([A-Z]+)", part_1)[0]

    return postcode_area


def get_postcode_district(postcode):

    # print(postcode)
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

    if only_keep is None:
        del df["POSTCODE"]
    else:
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

        if feature == "HP_INSTALLED":
            print(feature_cats_by_agglo_f.head())

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
