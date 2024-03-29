# File: heat_pump_adoption_modelling/pipeline/encoding/feature_encoding.py
"""
Encoding categorical features with ordinal and one-hot encoding.
"""

# ----------------------------------------------------------------------------------

# Import
import numpy as np
import pandas as pd

from heat_pump_adoption_modelling.pipeline.encoding import category_reduction

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
    "N_SAME_UPRN_ENTRIES": ["1", "2", "3", "4", "5.0+"],
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

    df["N_SAME_UPRN_ENTRIES"] = df["N_SAME_UPRN_ENTRIES"].replace(["5.0+"], 5)
    df["N_SAME_UPRN_ENTRIES"] = df["N_SAME_UPRN_ENTRIES"].astype("int")

    for feat in features:

        if feat not in df.columns:
            continue

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
        one_hot.columns = [feat + ": " + str(cat) for cat in one_hot.columns]
        false_columns = [col for col in one_hot.columns if col.endswith("False")]
        one_hot.drop(columns=false_columns, inplace=True)

        # Join enocoded features with original df
        df = df.join(one_hot)

        # Drop the original feature
        df = df.drop(feat, axis=1)

    if verbose:
        print("After one hot encoding:", df.shape)

    return df


def feature_encoding_pipeline(
    df,
    ordinal_features,
    reduce_categories=True,
    onehot_features="auto",
    unaltered_features=None,
    drop_features=None,
):
    """Pipeline for encoding ordinal and one-hot encoding of features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe including features to encode.

    ordinal_features : list
        Features to ordinal encode.

    reduce_categories : bool, default=True
        Whether or not merge categories of cateogrical features.

    onehot_features : list, default="auto"
        Features to one-hot encode.
        If set to "auto", suitable features will be identified automatically.
        To avoid one-hot encoding, use empty list or None.

    unaltered_features : list, str, default=None
        These variables will not be encoded.

    drop_features : list, default=None
        Features to discard.

    Return
    ---------
    df : pandas.DataFrame
        Updated and encoded dataframe."""

    # Drop featuress
    if drop_features is not None:
        df = df.drop(columns=drop_features)

    # Reduce/merge categories
    if reduce_categories:
        df = category_reduction.reduce_number_of_categories(df)

    # Get all only numeric features
    num_features = df.select_dtypes(include=np.number).columns.tolist()
    num_features = [f for f in num_features if f not in ["BUILDING_ID", "UPRN"]]

    print("numeric", num_features)

    # Ordinal encoding
    df = ordinal_encode_cat_features(df, ordinal_features)

    print("ordinal", ordinal_features)

    # Optional one-hot encoding
    if (onehot_features is not None) or onehot_features:

        # If automatically identifying one-hot features
        if onehot_features == "auto":

            # Get categorical features (exclude ordinally encoded ones)
            categorical_features = [
                feature
                for feature in df.columns
                if (feature not in ordinal_features) and (feature not in num_features)
            ]

            print("cat", categorical_features)

            # Convert target variables into list
            keep_features = [] if unaltered_features is None else unaltered_features
            keep_features = (
                list(keep_features) if isinstance(keep_features, str) else keep_features
            )

            # Select features to be one-hot encoded, exclude target variables
            one_hot_features = [
                f for f in categorical_features if f not in keep_features
            ]

        print("keep features", keep_features)
        print("one hot", one_hot_features)
        print("unaltered features", unaltered_features)

        # One-hot encoding
        df = one_hot_encoding(df, one_hot_features)

    return df


# numeric ['IMD Rank', 'IMD Decile', 'Income Score', 'Employment Score', 'ENERGY_CONSUMPTION_CURRENT', 'TOTAL_FLOOR_AREA', 'CURRENT_ENERGY_EFFICIENCY', 'CO2_EMISSIONS_CURRENT', 'HEATING_COST_CURRENT', 'HOT_WATER_COST_CURRENT', 'LIGHTING_COST_CURRENT', 'FLOOR_HEIGHT', 'EXTENSION_COUNT', 'FLOOR_LEVEL', 'GLAZED_AREA', 'NUMBER_HABITABLE_ROOMS', 'MAIN_HEATING_CONTROLS', 'MULTI_GLAZE_PROPORTION', 'PHOTO_SUPPLY', 'WIND_TURBINE_COUNT', 'UPRN', 'BUILDING_ID', 'DIFF_POT_ENERGY_RATING', 'version', 'n_certificates', '# records']
# ordinal ['MAINHEAT_ENERGY_EFF', 'CURRENT_ENERGY_RATING', 'POTENTIAL_ENERGY_RATING', 'FLOOR_ENERGY_EFF', 'WINDOWS_ENERGY_EFF', 'HOT_WATER_ENERGY_EFF', 'LIGHTING_ENERGY_EFF', 'GLAZED_TYPE', 'MAINHEATC_ENERGY_EFF', 'WALLS_ENERGY_EFF', 'ROOF_ENERGY_EFF', 'MAINS_GAS_FLAG', 'CONSTRUCTION_AGE_BAND_ORIGINAL', 'CONSTRUCTION_AGE_BAND', 'N_ENTRIES', 'N_ENTRIES_BUILD_ID', 'ENERGY_RATING_CAT']
# one hot ['INSPECTION_DATE', 'SECONDHEAT_DESCRIPTION', 'MECHANICAL_VENTILATION', 'ENERGY_TARIFF', 'SOLAR_WATER_HEATING_FLAG', 'TENURE', 'TRANSACTION_TYPE', 'BUILT_FORM', 'PROPERTY_TYPE', 'COUNTRY', 'N_SAME_UPRN_ENTRIES', 'HEATING_SYSTEM', 'HEATING_FUEL', 'MCS address', 'new', 'alt_type', 'installation_type', 'ANY_HP', 'HP_AT_FIRST', 'HP_AT_LAST', 'HP_LOST', 'HP_ADDED', 'HP_IN_THE_MIDDLE', 'ARTIFICIALLY_DUPL', 'EPC HP entry before MCS', 'No EPC HP entry after MCS']
# Before one hot encoding: (4945597, 73)
##unaltered features ['POSTCODE', 'POSTCODE_DISTRICT', 'POSTCODE_SECTOR', 'POSTCODE_UNIT', 'HP_INSTALLED', 'N_ENTRIES_BUILD_ID', 'POSTCODE_AREA', 'HP_INSTALL_DATE', 'UPRN']


# remove = [ 'version', 'n_certificates', '# records', 'INSPECTION_DATE',  'MCS address', 'new', 'alt_type', 'installation_type', 'ANY_HP', 'HP_AT_FIRST', 'HP_AT_LAST', 'HP_LOST', 'HP_ADDED', 'HP_IN_THE_MIDDLE', 'ARTIFICIALLY_DUPL', 'EPC HP entry before MCS', 'No EPC HP entry after MCS']
