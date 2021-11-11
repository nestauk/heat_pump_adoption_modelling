# File: heat_pump_adoption_modelling/pipeline/encoding/feature_encoding.py
"""
Encoding categorical features with ordinal and one-hot encoding.
"""

# ----------------------------------------------------------------------------------

# Import
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------


order_dict = {
    "CURRENT_ENERGY_RATING": ["unknown", "G", "F", "E", "D", "C", "B", "A"],
    "POTENTIAL_ENERGY_RATING": ["unknown", "G", "F", "E", "D", "C", "B", "A"],
    "NUMBER_HABITABLE_ROOMS": [
        "unknown",
        "0.0",
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10+",
    ],
    "MAINS_GAS_FLAG": ["N", "unknown", "Y"],
    "CONSTRUCTION_AGE_BAND_ORIGINAL": [
        "England and Wales: before 1900",
        "Scotland: before 1919",
        "England and Wales: 1900-1929",
        "Scotland: 1919-1929",
        "England and Wales: 1930-1949",
        "Scotland: 1930-1949",
        "Scotland: 1950-1964",
        "England and Wales: 1950-1966",
        "England and Wales: 1967-1975",
        "Scotland: 1965-1975",
        "England and Wales: 1976-1982",
        "Scotland: 1976-1983",
        "England and Wales: 1983-1990",
        "Scotland: 1984-1991",
        "England and Wales: 1991-1995",
        "Scotland: 1992-1998",
        "England and Wales: 1996-2002",
        "Scotland: 1999-2002",
        "England and Wales: 2003-2006",
        "Scotland: 2003-2007",
        "England and Wales: 2007 onwards",
        "Scotland: 2008 onwards",
        "England and Wales: 2012 onwards",
        "unknown",
    ],
    "CONSTRUCTION_AGE_BAND": [
        "England and Wales: before 1900",
        "Scotland: before 1919",
        "1900-1929",
        "1930-1949",
        "1950-1966",
        "1965-1975",
        "1976-1983",
        "1983-1991",
        "1991-1998",
        "1996-2002",
        "2003-2007",
        "2007 onwards",
        "unknown",
    ],
    "N_ENTRIES_BUILD_ID": ["1", "2", "3", "4", "5.0+"],
    "N_ENTRIES": ["1", "2", "3", "4", "5.0+"],
    "FLOOR_LEVEL": [
        "unknown",
        "-1",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10+",
    ],
    "ENERGY_RATING_CAT": ["unknown", "E-G", "C-D", "A-B"],
    "GLAZED_TYPE": ["unknown", "single glazing", "double glazing", "triple glazing"],
}

eff_value_dict = {
    "Very Poor": 1,
    "Poor": 2,
    "Average": 3,
    "Good": 4,
    "Very Good": 5,
}


def create_efficiency_mapping(efficiency_set):
    """Create dict to map efficiency label(s) to numeric value.

    Parameters
    ----------
    efficiency_set : list
        List of efficiencies as encoded as strings.

    Return
    ---------
    efficiency_map : dict
        Dict to map efficiency labels to numeric values."""

    efficiency_map = {}

    for eff in efficiency_set:

        # If efficiency is float (incl. NaN)
        if isinstance(eff, float):
            efficiency_map[eff] = 0.0
            continue

        # Split parts of label (especially for Scotland data)
        eff_parts = [
            part.strip()
            for part in eff.split("|")
            if part.strip() not in ["N/A", "unknown", ""]
        ]

        if not eff_parts:
            efficiency_map[eff] = 0
            continue

        # Map labels to numeric value and take mean
        eff_value = sum([eff_value_dict[part] for part in eff_parts]) / float(
            len(eff_parts)
        )

        efficiency_map[eff] = round(eff_value, 1)

    return efficiency_map


def ordinal_encode_cat_features(df, features):
    """Ordinal encode given categorical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe including features to encode.

    features: list
        Features to ordinal encode.

    Return
    ---------
    df : pandas.DataFrame
        Dataframe with ordinal encoded features."""

    for feat in features:

        # If efficiency feature, get respective mapping
        if feat.endswith("_EFF"):
            map_dict = create_efficiency_mapping(list(df[feat].unique()))
        else:
            # General mapping given ordered list of categories
            map_dict = dict(zip(order_dict[feat], range(1, len(order_dict[feat]) + 1)))

        # Encode features
        df[feat] = df[feat].map(map_dict)

    return df


def one_hot_encoding(df, features, verbose=True):
    """One-hot encode given categorical features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe including features to encode.

    features: list
        Features to ordinal encode.

    Return
    ---------
    df : pandas.DataFrame
        Dataframe with one-hot encoded features."""

    if verbose:
        print("Before one hot encoding:", df.shape)

    for feat in features:
        one_hot = pd.get_dummies(df[feat])

        # Create new column names
        one_hot.columns = [feat + ": " + cat for cat in one_hot.columns]

        # Join enocoded features with original df
        df = df.join(one_hot)

        # Drop the original feature
        df = df.drop(feat, axis=1)

    if verbose:
        print("After one hot encoding:", df.shape)

    return df
