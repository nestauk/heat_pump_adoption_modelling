# File: heat_pump_adoption_modelling/pipeline/preprocessing/data_cleaning.py
"""Cleaning and standardising the EPC dataset."""

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)


def reformat_postcode(df):
    """Change the POSTCODE feature in uniform format (without spaces).

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe to format.

    Return
    ----------
    df : pandas.Dataframe
        Dataframe with reformatted POSTCODE."""

    df["POSTCODE"] = df["POSTCODE"].str.replace(r" ", "")

    return df


def date_formatter(date):
    """Reformat the date to the format year-month-day.

    Parameters
    ----------
    date: str/float
        Date to reformat (usually from INSPECTION_DATE).

    Return
    ----------
    formatted_date : str
        Formatted data in the required format."""

    if isinstance(date, float):
        return "unknown"

    # Get year, month and day from different formats
    if "-" in date:
        year, month, day = str(date).split("-")
    elif "/" in date:
        day, month, year = str(date).split("/")
    else:
        return "unknown"

    # Assume that years starting with 00xx mean 20xx
    if year.startswith("00"):
        year = "20" + year[-2:]

    # Discard entries past current year + plus (future projects) and before 2008
    if len(year) != 4 or int(year) > config["CURRENT_YEAR"] + 1 or int(year) < 2008:
        return "unknown"

    formatted_date = year + "-" + month + "-" + day

    return formatted_date


def standardise_tenure(tenure):
    """Standardise tenure types; one of the four categories:
    rental (social), rental (private), owner-occupied, unknown

    Parameters
    ----------
    tenure : str
        Raw tenure type.

    Return
    ----------
    standardised tenure : str
        Standardised tenure type."""

    # Catch NaN
    if isinstance(tenure, float):
        return "unknown"

    tenure = tenure.lower()
    tenure_mapping = {
        "owner-occupied": "owner-occupied",
        "rental (social)": "rental (social)",
        "rented (social)": "rental (social)",
        "rental (private)": "rental (private)",
        "rented (private)": "rental (private)",
        "unknown": "unknown",
        "no data!": "unknown",
        "not defined - use in the case of a new dwelling for which the intended tenure in not known. it is no": "unknown",
    }

    return tenure_mapping[tenure]


def standardise_constr_age(age, adjust_age_bands=True):

    """Standardise construction age bands and if necessary adjust
    the age bands to combine the Scotland and England/Wales data.

    Parameters
    ----------
    age : str
        Raw construction age.

    merge_country_data : bool, default=True
        Whether to merge Scotland and England/Wales age bands.

    Return
    ----------
    Standardised age construction band : str
        Standardised age construction band."""

    # Handle NaN
    if isinstance(age, float):
        return "unknown"

    age = age.strip()

    age_mapping = {
        "England and Wales: before 1900": "England and Wales: before 1900",
        "England and Wales: 1900-1929": "England and Wales: 1900-1929",
        "England and Wales: 1930-1949": "England and Wales: 1930-1949",
        "England and Wales: 1950-1966": "England and Wales: 1950-1966",
        "England and Wales: 1967-1975": "England and Wales: 1967-1975",
        "England and Wales: 1976-1982": "England and Wales: 1976-1982",
        "England and Wales: 1983-1990": "England and Wales: 1983-1990",
        "England and Wales: 1991-1995": "England and Wales: 1991-1995",
        "England and Wales: 1996-2002": "England and Wales: 1996-2002",
        "England and Wales: 2003-2006": "England and Wales: 2003-2006",
        "England and Wales: 2007-2011": "England and Wales: 2007-2011",
        "England and Wales: 2007 onwards": "England and Wales: 2007 onwards",
        "England and Wales: 2012 onwards": "England and Wales: 2012 onwards",
        "2021": "England and Wales: 2012 onwards",
        "2020": "England and Wales: 2012 onwards",
        "2019": "England and Wales: 2012 onwards",
        "2018": "England and Wales: 2012 onwards",
        "2017": "England and Wales: 2012 onwards",
        "2016": "England and Wales: 2012 onwards",
        "2015": "England and Wales: 2012 onwards",
        "2014": "England and Wales: 2012 onwards",
        "2013": "England and Wales: 2012 onwards",
        "2012": "England and Wales: 2012 onwards",
        "2011": "England and Wales: 2007 onwards",
        "2010": "England and Wales: 2007 onwards",
        "2009": "England and Wales: 2007 onwards",
        "2008": "England and Wales: 2007 onwards",
        "2007": "England and Wales: 2007 onwards",
        "before 1919": "Scotland: before 1919",
        "1919-1929": "Scotland: 1919-1929",
        "1930-1949": "Scotland: 1930-1949",
        "1950-1964": "Scotland: 1950-1964",
        "1965-1975": "Scotland: 1965-1975",
        "1976-1983": "Scotland: 1976-1983",
        "1984-1991": "Scotland: 1984-1991",
        "1992-1998": "Scotland: 1992-1998",
        "1999-2002": "Scotland: 1999-2002",
        "2003-2007": "Scotland: 2003-2007",
        "2008 onwards": "Scotland: 2008 onwards",
        "unknown": "unknown",
        "NO DATA!": "unknown",
        "INVALID!": "unknown",
        "Not applicable": "unknown",
    }

    age_mapping_adjust_age_bands = {
        "England and Wales: before 1900": "England and Wales: before 1900",
        "England and Wales: 1900-1929": "1900-1929",
        "England and Wales: 1930-1949": "1930-1949",
        "England and Wales: 1950-1966": "1950-1966",
        "England and Wales: 1967-1975": "1965-1975",
        "England and Wales: 1976-1982": "1976-1983",
        "England and Wales: 1983-1990": "1983-1991",
        "England and Wales: 1991-1995": "1991-1998",
        "England and Wales: 1996-2002": "1996-2002",
        "England and Wales: 2003-2006": "2003-2007",
        "England and Wales: 2007-2011": "2007 onwards",
        "England and Wales: 2007 onwards": "2007 onwards",
        "England and Wales: 2012 onwards": "2007 onwards",
        "2021": "2007 onwards",
        "2020": "2007 onwards",
        "2019": "2007 onwards",
        "2018": "2007 onwards",
        "2017": "2007 onwards",
        "2016": "2007 onwards",
        "2015": "2007 onwards",
        "2015": "2007 onwards",
        "2014": "2007 onwards",
        "2013": "2007 onwards",
        "2013": "2007 onwards",
        "2012": "2007 onwards",
        "2011": "2007 onwards",
        "2010": "2007 onwards",
        "2009": "2007 onwards",
        "2008": "2007 onwards",
        "2007": "2007 onwards",
        "1919-1929": "1900-1929",
        "1930-1949": "1930-1949",
        "1950-1964": "1950-1966",
        "1965-1975": "1965-1975",
        "1976-1983": "1976-1983",
        "1984-1991": "1983-1991",
        "1992-1998": "1991-1998",
        "1999-2002": "1996-2002",
        "2003-2007": "2003-2007",
        "2008 onwards": "2007 onwards",
        "unknown": "unknown",
        "NO DATA!": "unknown",
        "INVALID!": "unknown",
        "Not applicable": "unknown",
    }

    if adjust_age_bands:

        if age not in age_mapping_adjust_age_bands:
            return "unknown"
        else:
            return age_mapping_adjust_age_bands[age]
    else:
        if age not in age_mapping:
            return "unknown"
        else:
            return age_mapping[age]


def standardise_efficiency(efficiency):
    """Standardise efficiency types; one of the five categories:
    poor, very poor, average, good, very good

    Parameters
    ----------
    tenure : str
        Raw efficiency type.

    Return
    ----------
    standardised tenure : str
        Standardised efficiency type."""

    # Handle NaN
    if isinstance(efficiency, float):
        return "unknown"

    efficiency = efficiency.lower().strip()
    efficiency = efficiency.strip('"')
    efficiency = efficiency.strip()
    efficiency = efficiency.strip("|")
    efficiency = efficiency.strip()

    efficiency_mapping = {
        "poor |": "Poor",
        "very poor |": "Very Poor",
        "average |": "Average",
        "good |": "Good",
        "very good |": "Very Good",
        "poor": "Poor",
        "very poor": "Very Poor",
        "average": "Average",
        "good": "Good",
        "very good": "Very Good",
        "n/a": "unknown",
        "n/a |": "unknown",
        "n/a": "unknown",
        "n/a | n/a": "unknown",
        "n/a | n/a | n/a": "unknown",
        "n/a | n/a | n/a | n/a": "unknown",
        "no data!": "unknown",
        "unknown": "unknown",
    }

    return efficiency_mapping[efficiency]


def clean_local_authority(local_authority):
    """Clean local authority label.

    Paramters
    ----------
    local_authority : str
        Local authority label.

    Return
    ----------
    local_authority : str
        Cleaned local authority."""

    if local_authority in ["00EM", "16UD"]:
        return "unknown"
    else:
        return local_authority


def clean_epc_data(df):
    """Standardise and clean EPC data.
    For example, reformat dates and standardise cateogories.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw/original EPC dataframe.

    Return
    ----------
    df : pandas.DataFrame
        Standarised and cleaned EPC dataframe."""

    for column in df.columns:
        df[column] = df[column].fillna("unknown")

    if "POSTCODE" in df.columns:
        df["POSTCODE"] = df["POSTCODE"].str.upper()

    if "LOGEMENT_DATE" in df.columns:
        # Reformat dates
        df["LODGEMENT_DATE"] = df["LODGEMENT_DATE"].apply(date_formatter)

    if "INSPECTION_DATE" in df.columns:
        df.dropna(subset=["INSPECTION_DATE"], inplace=True)
        df["INSPECTION_DATE"] = df["INSPECTION_DATE"].apply(date_formatter)

    if "TENURE" in df.columns:
        # Standarise tenure type
        df["TENURE"] = df["TENURE"].apply(standardise_tenure)

    if "CONSTRUCTION_AGE_BAND" in df.columns:
        df["CONSTRUCTION_AGE_BAND"] = df["CONSTRUCTION_AGE_BAND"].apply(
            standardise_constr_age
        )

    if "NUMBER_HABITABLE_ROOMS" in df.columns:

        # Change 'unknown' temporarily to integer for capping at 10+ rooms
        df["NUMBER_HABITABLE_ROOMS"].replace({"unknown": -1}, inplace=True)

        df.loc[
            (df.NUMBER_HABITABLE_ROOMS >= 10),
            "NUMBER_HABITABLE_ROOMS",
        ] = "10+"

        # Change back to 'unknown'
        df["NUMBER_HABITABLE_ROOMS"].replace({-1: "unknown"}, inplace=True)

    if "WINDOWS_ENERGY_EFF" in df.columns:
        df["WINDOWS_ENERGY_EFF"] = df["WINDOWS_ENERGY_EFF"].apply(
            standardise_efficiency
        )

    if "FLOOR_ENERGY_EFF" in df.columns:
        df["FLOOR_ENERGY_EFF"] = df["FLOOR_ENERGY_EFF"].apply(standardise_efficiency)

    if "HOT_WATER_ENERGY_EFF" in df.columns:
        df["HOT_WATER_ENERGY_EFF"] = df["HOT_WATER_ENERGY_EFF"].apply(
            standardise_efficiency
        )

    if "LIGHTING_ENERGY_EFF" in df.columns:
        df["LIGHTING_ENERGY_EFF"] = df["LIGHTING_ENERGY_EFF"].apply(
            standardise_efficiency
        )

    if "LOCAL_AUTHORITY_LABEL" in df.columns:
        df["LOCAL_AUTHORITY_LABEL"] = df["LOCAL_AUTHORITY_LABEL"].apply(
            clean_local_authority
        )

    if "CURRENT_ENERGY_RATING" in df.columns:
        df["CURRENT_ENERGY_RATING"] = df["CURRENT_ENERGY_RATING"].replace(
            ["INVALID!"], "unknown"
        )

    if "POTENTIAL_ENERGY_RATING" in df.columns:
        df["POTENTIAL_ENERGY_RATING"] = df["POTENTIAL_ENERGY_RATING"].replace(
            ["INVALID!"], "unknown"
        )

    if "BUILT_FORM" in df.columns:
        df["BUILT_FORM"] = df["BUILT_FORM"].replace(["NO DATA!"], "unknown")

    return df
