# File: heat_pump_adoption_modelling/pipeline/preprocessing/feature_engineering.py
"""
Adding new features to EPC dataset.
"""

# ----------------------------------------------------------------------------------

# Import
import pandas as pd
import numpy as np
import re
from hashlib import md5

from heat_pump_adoption_modelling.getters import location_data
from heat_pump_adoption_modelling.pipeline.preprocessing import data_cleaning

# ----------------------------------------------------------------------------------


def get_coordinates(df):
    """Add coordinates (longitude and latitude) to the dataframe
    based on the postcode.

    df : pandas.DataFrame
        EPC dataframe.

    Return
    ---------
    df : pandas.DataFrame
        Same dataframe with longitude and latitude columns added."""

    # Get postcode/coordinates
    location_df = location_data.get_location_data()

    # Reformat POSTCODE
    df = data_cleaning.reformat_postcode(df)
    location_df = data_cleaning.reformat_postcode(location_df)

    # Merge with location data
    df = pd.merge(df, location_df, on=["POSTCODE"])

    return df


def short_hash(text):
    """Generate a unique short hash for given string.

    Parameters
    ----------
    text: str
        String for which to get ID.

    Return
    ---------

    short_code: int
        Unique ID."""

    hx_code = md5(text.encode()).hexdigest()
    int_code = int(hx_code, 16)
    short_code = str(int_code)[:16]
    return int(short_code)


def get_unique_building_id(df):
    """Add unique building ID column to dataframe.

    Parameters
    ----------
    text: str
        String for which to get ID.

    Return
    ---------

    short_code: int
        Unique ID."""

    if ("ADDRESS1" not in df.columns) or ("POSTCODE" not in df.columns):
        return df

    # Remove samples with no address
    df.dropna(subset=["ADDRESS1"], inplace=True)

    # Create unique address and building ID
    df["UNIQUE_ADDRESS"] = df["ADDRESS1"].str.upper() + df["POSTCODE"].str.upper()
    df["BUILDING_ID"] = df["UNIQUE_ADDRESS"].apply(short_hash)

    return df


def get_new_epc_rating_features(df):
    """Get new EPC rating features related to EPC ratings.

        CURR_ENERGY_RATING_NUM: EPC rating representeed as number
        high number = high rating.

        ENERGY_RATING_CAT: EPC range category.
        A-B, C-D or E-G

        DIFF_POT_ENERGY_RATING: Difference potential and current
        energy rating.


    Parameters
    ----------
    df : pandas.Dataframe
        EPC dataframe.

    Return
    ---------
    df : pandas.DateFrame
        Updated EPC dataframe with new EPC rating features."""

    # EPC rating dict
    rating_dict = {
        "A": 7,
        "B": 6,
        "C": 5,
        "D": 4,
        "E": 3,
        "F": 2,
        "G": 1,
        "unknown": float("NaN"),
    }

    # EPC range cat dict
    EPC_cat_dict = {
        "A": "A-B",
        "B": "A-B",
        "C": "C-D",
        "D": "C-D",
        "E": "E-G",
        "F": "E-G",
        "G": "E-G",
        "unknown": "unknown",
    }

    if "CURRENT_ENERGY_RATING" not in df.columns:
        return df

    # EPC rating in number instead of letter
    df["CURR_ENERGY_RATING_NUM"] = df.CURRENT_ENERGY_RATING.map(rating_dict)

    # EPC rating in category (A-B, C-D or E-G)
    df["ENERGY_RATING_CAT"] = df.CURRENT_ENERGY_RATING.map(EPC_cat_dict)

    if "POTENTIAL_ENERGY_RATING" in df.columns:

        # Numerical difference between current and potential energy rating (A-G)
        df["DIFF_POT_ENERGY_RATING"] = (
            df.POTENTIAL_ENERGY_RATING.map(rating_dict) - df["CURR_ENERGY_RATING_NUM"]
        )
        # Filter out unreasonable values (below 0 and above 7)
        df.loc[
            ((df.DIFF_POT_ENERGY_RATING < 0.0) | (df.DIFF_POT_ENERGY_RATING > 7)),
            "DIFF_POT_ENERGY_RATING",
        ] = np.nan

    return df


def map_quality_to_score(df, list_of_features):
    """Map quality string tag (e.g. 'very good') to score and add it as a new feature.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to update.

    list_of_features: list
        A list of dataframe features to update.

    Return
    ---------
    df : pandas.DataFrame:
        Updated dataframe with new score features."""

    quality_to_score_dict = {
        "Very Good": 5.0,
        "Good": 4.0,
        "Average": 3.0,
        "Poor": 2.0,
        "Very Poor": 1.0,
    }

    for feature in list_of_features:
        df[feature + "_AS_SCORE"] = df[feature].map(quality_to_score_dict)

    return df


def map_rating_to_cat(rating):
    """Map EPC rating in numbers (between 1.0 and 7.0) to EPC category range.

    Parameters
    ----------
    rating : float
        EPC rating - usually average scores.

    Return
    ---------
    EPC category range, e.g. A-B."""

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


def get_heating_features(df, fine_grained_HP_types=False):
    """Get heating type category based on HEATING_TYPE category.
    heating_system: heat pump, boiler, community scheme etc.
    heating_source: oil, gas, LPC, electric.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that is updated with heating features.

    fine_grained_HP_types : bool, default=False
        If True, get different heat pump types (air sourced, ground sourced etc.).
        If False, return "heat pump" as heating type category.

    Return
    ---------
    df : pandas.DataFrame
        Updated dataframe with heating system and source."""

    if "MAINHEAT_DESCRIPTION" not in df.columns:
        return df

    # Collections
    heating_system_types = []
    heating_source_types = []
    has_hp_tags = []
    hp_types = []

    # Get heating types
    heating_types = df["MAINHEAT_DESCRIPTION"]

    # Get specific and general heating category for each entry
    for heating in heating_types:

        # Set default value
        system_type = "unknown"
        source_type = "unknown"
        has_hp = False
        hp_type = "No HP"

        # If heating value exists
        if not (pd.isnull(heating) and isinstance(heating, float)):

            # Lowercase
            heating = heating.lower()

            other_heating_system = [
                ("boiler and radiator" in heating),
                ("boiler & radiator" in heating),
                ("boiler and underfloor" in heating),
                ("boiler & underfloor" in heating),
                ("community scheme" in heating),
                ("heater" in heating),  # not specified heater
            ]

            # Different heat pump types
            # --------------------------

            if "ground source heat pump" in heating:
                system_type = "ground source heat pump"
                source_type = "electric"
                has_hp = True

            elif "air source heat pump" in heating:
                system_type = "air source heat pump"
                source_type = "electric"
                has_hp = True

            elif "water source heat pump" in heating:
                system_type = "water source heat pump"
                source_type = "electric"
                has_hp = True

            elif "community heat pump" in heating:
                system_type = "community heat pump"
                source_type = "electric"
                has_hp = True

            elif "heat pump" in heating:
                system_type = "heat pump"
                source_type = "electric"
                has_hp = True

            # Electric heaters
            # --------------------------

            elif "electric storage heaters" in heating:
                system_type = "storage heater"
                source_type = "electric"

            elif "electric underfloor heating" in heating:
                system_type = "underfloor heating"
                source_type = "electric"

            # Warm air
            # --------------------------

            elif "warm air" in heating:
                system_type = "warm air"
                source_type = "electric"

            # Boiler and radiator / Boiler and underfloor / Community scheme / Heater (unspecified)
            # --------------------------

            elif any(other_heating_system):

                # Set heating system dict
                heating_system_dict = {
                    "boiler and radiator": "boiler and radiator",
                    "boiler & radiator": "boiler and radiator",
                    "boiler and underfloor": "boiler and underfloor",
                    "boiler & underfloor": "boiler and underfloor",
                    "community scheme": "community scheme",
                    "heater": "heater",  # not specified heater (otherwise handeld above)
                }

                # Set heating source dict
                heating_source_dict = {
                    "gas": "gas",
                    ", oil": "oil",  # with preceeding comma (!= "boiler")
                    "lpg": "LPG",
                    "electric": "electric",
                }

                # If heating system word is found, save respective system type
                for word, system in heating_system_dict.items():
                    if word in heating:
                        system_type = system

                # If heating source word is found, save respective source type
                for word, source in heating_source_dict.items():
                    if word in heating:
                        source_type = source

        # Set HP type
        if has_hp:
            hp_type = system_type

            # Don't differentiate between heat pump types
            if not fine_grained_HP_types:
                system_type = "heat pump"

        # Save heating system type and source type
        heating_system_types.append(system_type)
        heating_source_types.append(source_type)
        has_hp_tags.append(has_hp)
        hp_types.append(hp_type)

    # Add heating system and source to df
    df["HEATING_SYSTEM"] = heating_system_types
    df["HEATING_FUEL"] = heating_source_types
    df["HP_INSTALLED"] = has_hp_tags
    df["HP_TYPE"] = hp_types

    # Also consider secondheat description and other languages
    df["HP_INSTALLED"] = np.where(
        (df["HP_INSTALLED"])
        | (df["SECONDHEAT_DESCRIPTION"].str.lower().str.contains("heat pump"))
        | (df["MAINHEAT_DESCRIPTION"].str.lower().str.contains("pumpa teas"))
        | (df["MAINHEAT_DESCRIPTION"].str.lower().str.contains("pwmp gwres")),
        True,
        False,
    )

    return df


def get_year(date):
    """Year for given date.

    Parameters
    ----------
    date : str
        Given date in format year-month-day.

    Return
    ---------
    year : int
        Year derived from date."""

    if date is np.nan or date == "unknown":
        return np.nan

    year = date.split("/")[0]

    # If year format doesn't match
    if len(year) != 4:
        return np.nan

    return int(year)


def get_date_as_int(date):
    """Transform date into integer to compute earliest/latest date.
    Ideally used after reformatting date to year-month-date.

    Parameters
    ----------
    date : str
        Given date in format year-month-day or yearmonthday.

    Return
    ---------
    date : int
        Date as integer."""

    # If already numeric, return that
    if isinstance(date, float):
        return date

    # Handle unknown/NaN
    if date is np.nan or date == "unknown":
        return -1

    # Remove delimiter
    date = re.sub("/", "", date)

    return int(date)


def get_date_features(df):
    """Get year of inspection and entry date as integer features.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to which new features are added.

    Return
    ---------
    df : pandas.DataFrame
        Dataframe with new features."""

    if "INSPECTION_DATE" not in df.columns:
        return df

    df["ENTRY_YEAR"] = df["INSPECTION_DATE"].apply(get_year)
    df["ENTRY_YEAR_INT"] = df["ENTRY_YEAR"].apply(get_date_as_int)
    df["INSPECTION_DATE_AS_NUM"] = df["INSPECTION_DATE"].apply(get_date_as_int)

    return df


def filter_by_year(df, building_reference, year, up_to=True, selection=None):
    """Filter EPC dataset by year of inspection/entry.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe to which new features are added.

    building_reference : str
        Which building reference to use,
        e.g. "BUILDING_REFERENCE_NUMBER" or "BUILDING_ID".

    year : int, None, "all"
        Year by which to filter data.
        If None or "all", use all data.

    up_to : bool, default=True
        If True, get all samples up to given year.
        If False, only get sample from given year.

    selection : {"first entry", "latest entry"} or None, default=None
        For duplicates, get only first or latest entry.
        If None, do not remove any duplicates.

    Return
    ---------
    df : pandas.DataFrame
        Dataframe with new features."""

    # If year is given for filtering
    if year != "all" and year is not None:

        if up_to:
            df = df.loc[df["INSPECTION_DATE"].dt.year <= year]
        else:
            df = df.loc[df["INSPECTION_DATE"].dt.year == year]

    # Filter by selection
    selection_dict = {"first entry": "first", "latest entry": "last"}

    if selection in ["first entry", "latest entry"]:

        df = (
            df.sort_values("INSPECTION_DATE", ascending=True)
            .drop_duplicates(
                subset=[building_reference], keep=selection_dict[selection]
            )
            .sort_index()
        )

    elif selection is None:
        df = df

    else:
        raise IOError("{} not implemented.".format(selection))

    return df


def count_number_of_entries(row, feature, ref_counts):
    """Count the number entries for given building based on
    building reference number.

    row : pandas.Series
        EPC dataset row.

    feature: str
        Feature by which to count building entries.
        e.g. "BUILDING_REFERNCE_NUMBER" or "BUILDING_ID"

    ref_counts : pandas.Series
        Value counts for building reference number.

    Return
    ---------
    counts : int
        How many entries are there for given building."""

    building_ref = row[feature]
    try:
        counts = ref_counts[building_ref]
    except KeyError:
        return building_ref

    return counts


def get_postcode_coordinates(df):
    """Add coordinates (longitude and latitude) to the dataframe
    based on the postcode.

    df : pandas.DataFrame
        EPC dataframe.

    Return
    ---------
    df : pandas.DataFrame
        Same dataframe with longitude and latitude columns added."""

    # Get postcode/coordinates
    postcode_coordinates_df = location_data.get_postcode_coordinates()

    # Reformat POSTCODE
    df = data_cleaning.reformat_postcode(df)
    postcode_coordinates_df = data_cleaning.reformat_postcode(postcode_coordinates_df)

    postcode_coordinates_df["POSTCODE"] = (
        postcode_coordinates_df["POSTCODE"].str.upper().str.replace(" ", "")
    )

    # Merge with location data
    df = pd.merge(df, postcode_coordinates_df, on=["POSTCODE"])

    print(df.shape)

    return df


def get_building_entry_feature(df, feature):
    """Create feature that shows number of entries for any given building
    based on BUILDING_REFERENCE_NUMBER or BUILDING_ID.

    df : pandas.DataFrame
        EPC dataframe.

    feature: str
        Feature by which to count building entries.
        Has to be "BUILDING_REFERNCE_NUMBER" or "BUILDING_ID".

    Return
    ---------
    df : pandas.DataFrame
        EPC dataframe with # entry feature."""

    # Catch invalid inputs
    if feature not in ["BUILDING_REFERENCE_NUMBER", "BUILDING_ID", "UPRN"]:
        raise IOError("Feature '{}' is not a valid feature.".format(feature))

    feature_name_dict = {
        "BUILDING_REFERENCE_NUMBER": "N_ENTRIES",
        "BUILDING_ID": "N_ENTRIES_BUILD_ID",
        "UPRN": "N_SAME_UPRN_ENTRIES",
    }

    # Get name of new feature
    new_feature_name = feature_name_dict[feature]

    # Count IDs
    counts = df[feature].value_counts()

    # Create new feature representing how many entries there are for building
    df[new_feature_name] = df.apply(
        lambda row: count_number_of_entries(row, feature, counts), axis=1
    )

    df.loc[(df[new_feature_name] >= 5), new_feature_name] = "5.0+"

    return df


def get_building_entries(df):

    if "BUILDING_REFERENCE_NUMBER" in df.columns:
        df = get_building_entry_feature(df, "BUILDING_REFERENCE_NUMBER")

    if "BUILDING_ID" in df.columns:
        df = get_building_entry_feature(df, "BUILDING_ID")

    if "UPRN" in df.columns:
        df = get_building_entry_feature(df, "UPRN")

    return df


def get_additional_features(df):
    """Add new features to the EPC dataset.
    The new features include information about the inspection and entry date,
    building references, fine-grained heating system features and differences in EPC ratings.

    Parameters
    ---------
    df : pandas.DataFrame
        EPC dataframe.

    Return
    ---------
    df : pandas.DataFrame
        Updated dataframe with new features."""

    # df = get_date_features(df)

    df = get_unique_building_id(df)
    df = get_building_entries(df)

    df = get_heating_features(df)
    df = get_new_epc_rating_features(df)

    return df
