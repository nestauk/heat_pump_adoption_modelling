# File: heat_pump_adoption_modelling/pipeline/supervised_model/data_preprocessing.py
"""
Preprocess EPC and MCS data before feeding it to supervised model.
"""

# ----------------------------------------------------------------------------------

# Import

from heat_pump_adoption_modelling.getters import epc_data, deprivation_data
from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

from heat_pump_adoption_modelling.pipeline.supervised_model import data_aggregation
from heat_pump_adoption_modelling.pipeline.encoding import feature_encoding
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------


ordinal_features = [
    "MAINHEAT_ENERGY_EFF",
    "CURRENT_ENERGY_RATING",
    "POTENTIAL_ENERGY_RATING",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
    "GLAZED_TYPE",
    "MAINHEATC_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "MAINS_GAS_FLAG",
    "CONSTRUCTION_AGE_BAND_ORIGINAL",
    "CONSTRUCTION_AGE_BAND",
    "N_ENTRIES",
    "N_ENTRIES_BUILD_ID",
    "ENERGY_RATING_CAT",
]

drop_features = [
    "BUILDING_REFERENCE_NUMBER",
    "ADDRESS1",
    "ADDRESS2",
    "POSTTOWN",
    "LODGEMENT_DATE",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "MAINHEAT_DESCRIPTION",
    "SHEATING_ENERGY_EFF",
    "HEATING_COST_POTENTIAL",
    "HOT_WATER_COST_POTENTIAL",
    "LIGHTING_COST_POTENTIAL",
    "CONSTRUCTION_AGE_BAND",
    "NUMBER_HEATED_ROOMS",
    "LOCAL_AUTHORITY_LABEL",
    "ENTRY_YEAR",
    "N_ENTRIES",
    "CURR_ENERGY_RATING_NUM",
    "ENERGY_RATING_CAT",
    "UNIQUE_ADDRESS",
    "INSPECTION_DATE",
    "MAIN_FUEL",
    "Country",
    "original_address",
    "FIRST_HP_MENTION",
    "INSPECTION_YEAR",
    "HP_INSTALL_YEAR",
    "FIRST_HP_MENTION_YEAR",
    "MCS_AVAILABLE",
    "HAS_HP_AT_SOME_POINT",
    "HP_TYPE",
]

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

# Get paths
MERGED_MCS_EPC = str(PROJECT_DIR) + config["MERGED_MCS_EPC"]
SUPERVISED_MODEL_OUTPUT = str(PROJECT_DIR) + config["SUPERVISED_MODEL_OUTPUT"]
SUPERVISED_MODEL_FIG_PATH = str(PROJECT_DIR) + config["SUPERVISED_MODEL_FIG_PATH"]
dtypes = config["dtypes"]


def select_samples_by_postcode_completeness(
    df, min_samples=5000000, start=250000, interval=1000
):
    """Select a subset of samples with a minimal number of samples,
    including all samples for each unique postcode.
    This is done by increasing the number of postcodes and adding up the samples.

    Parameters
    ----------
    df: pandas.Dataframe
        Dataframe that includes the samples from which to select subet.

    min_samples: int, default=5000000
        Number of minimal samples, stop after reaching this number.

    start: int, default=25000
        Start with this number of postcodes.

    interval: int, default=1000
        The interval by which to increase number of postcodes every iteration.

    Return
    ---------
    df_reduced : pandas.Dataframe
        Subset of original dataframe
        with minimal number of samples and complete postcodes."""

    # Iterate over batches of postcodes and add up samples
    for i in range(start, min_samples, interval):
        grouped_by = (
            df.groupby("POSTCODE")
            .size()
            .reset_index(name="count")[:i]
            .sample(frac=1, random_state=42)
        )
        n_samples = grouped_by["count"].sum()

        # If minimal number of samples is reached, get all samples for
        # selected postcodes.
        if n_samples > min_samples:
            sample_ids = list(grouped_by["POSTCODE"])
            df_reduced = df.loc[df["POSTCODE"].isin(sample_ids)]
            return df_reduced


def get_mcs_install_dates(epc_df):
    """Get MCS install dates and them to the EPC data.

    Parameters
    ----------
    epc_df : pandas.DataFrame
        EPC dataset.


    Return
    ---------
    epc_df : pandas.DataFrame:
        EPC dataset with added MCS install dates."""

    # Get original address from EPC
    epc_df["original_address"] = (
        epc_df["ADDRESS1"] + epc_df["ADDRESS2"] + epc_df["POSTCODE"]
    )
    epc_df["original_address"] = (
        epc_df["original_address"]
        .str.strip()
        .str.lower()
        .replace(r"\s+", "", regex=True)
    )

    # Load MCS data
    mcs_data = pd.read_csv(
        MERGED_MCS_EPC,
        usecols=["date", "tech_type", "original_address"],
    )

    # Get original EPC address from MCS/EPC match
    mcs_data = mcs_data.loc[~mcs_data["original_address"].isna()]
    mcs_data["original_address"] = (
        mcs_data["original_address"]
        .str.strip()
        .str.lower()
        .replace(r"\s+", "", regex=True)
    )

    # Rename columns
    mcs_data.columns = ["HP_INSTALL_DATE", "Type of HP", "original_address"]

    # Get the MCS install dates
    mcs_data["HP_INSTALL_DATE"] = (
        mcs_data["HP_INSTALL_DATE"]
        .str.strip()
        .str.lower()
        .replace(r"-", "", regex=True)
        .astype("float")
    )

    # Create a date dict from MCS data and apply to EPC data
    # If no install date is found for address, it assigns NaN
    date_dict = mcs_data.set_index("original_address").to_dict()["HP_INSTALL_DATE"]
    epc_df["HP_INSTALL_DATE"] = epc_df["original_address"].map(date_dict)

    return epc_df


def manage_hp_install_dates(df):
    """Manage heat pump install dates given by EPC and MCS.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with EPC data and MCS install dates.

    Return
    ---------
    df : pandas.DataFrame:
        Dataframe with updated install dates."""

    # Get the MCS install dates for EPC properties
    df = get_mcs_install_dates(df)

    # Get the first heat pump mention for each property
    first_hp_mention = (
        df.loc[df["HP_INSTALLED"] == True]
        .groupby("BUILDING_ID")
        .min(["INSPECTION_DATE_AS_NUM"])
        .reset_index()[["INSPECTION_DATE_AS_NUM", "BUILDING_ID"]]
    )
    first_hp_mention.columns = ["FIRST_HP_MENTION", "BUILDING_ID"]
    df = pd.merge(df, first_hp_mention, on="BUILDING_ID", how="outer")

    # Get inspection, install and fist mention years
    df["INSPECTION_YEAR"] = round(df["INSPECTION_DATE_AS_NUM"] / 10000.0)
    df["HP_INSTALL_YEAR"] = round(df["HP_INSTALL_DATE"] / 10000.0)
    df["FIRST_HP_MENTION_YEAR"] = round(df["FIRST_HP_MENTION"] / 10000.0)

    # If no HP Install date, set MCS availabibility to False
    df["MCS_AVAILABLE"] = np.where(df["HP_INSTALL_DATE"].isna(), False, True)

    # If no first mention of HP, then set has
    df["HAS_HP_AT_SOME_POINT"] = np.where(df["FIRST_HP_MENTION"].isna(), False, True)

    # HP entry conditions
    no_mcs_or_epc = (~df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == False)
    no_mcs_but_epc_hp = (~df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == True)
    mcs_and_epc_hp = (df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == True)
    no_epc_but_mcs_hp = (df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == False)
    either_hp = (df["MCS_AVAILABLE"]) | (df["HP_INSTALLED"] == True)

    print("Total", df.shape)
    print("MCS and EPC", df[mcs_and_epc_hp].shape)
    print("no MCS or EPC", df[no_mcs_or_epc].shape)
    print("either", df[either_hp].shape)
    print("no MCS but EPC", df[no_mcs_but_epc_hp].shape)
    print("no epc but mcs", df[no_epc_but_mcs_hp].shape)

    # Was inspection year before/after/same year as install year?
    # epc_entry_before_mcs = df["INSPECTION_YEAR"] < df["HP_INSTALL_YEAR"]
    # mcs_before_epc_entry = df["INSPECTION_YEAR"] > df["HP_INSTALL_YEAR"]
    # epc_entry_same_as_mcs = df["INSPECTION_YEAR"] == df["HP_INSTALL_YEAR"]
    epc_entry_before_mcs = df["INSPECTION_YEAR"] < df["HP_INSTALL_YEAR"]

    print(
        "no HP mention: epc_entry_before mcs",
        df[no_epc_but_mcs_hp & epc_entry_before_mcs].shape,
    )

    print(
        "no HP mention: epc_entry_after/same mcs",
        df[no_epc_but_mcs_hp & ~epc_entry_before_mcs].shape,
    )

    print(
        "EPC HP mention: epc_entry_before mcs",
        df[mcs_and_epc_hp & epc_entry_before_mcs].shape,
    )

    print(
        "EPC HP mention: epc_entry_after/same mcs",
        df[mcs_and_epc_hp & ~epc_entry_before_mcs].shape,
    )
    # -----
    # NO MCS/EPC HP entry
    # No heat pump, no install date
    df["HP_INSTALLED"] = np.where((no_mcs_or_epc), False, df["HP_INSTALLED"])
    df["HP_INSTALL_DATE"] = np.where((no_mcs_or_epc), np.nan, df["HP_INSTALL_DATE"])

    # -----
    # No MCS entry but EPC HP entry:
    # Heat pump: yes
    # Install date: first HP mention
    df["HP_INSTALLED"] = np.where((no_mcs_but_epc_hp), True, df["HP_INSTALLED"])
    df["HP_INSTALL_DATE"] = np.where(
        (no_mcs_but_epc_hp), df["FIRST_HP_MENTION"], df["HP_INSTALL_DATE"]
    )

    # -----
    # MCS and EPC HP entry, with same year EPC entry or later
    # HP: yes, MCS install date
    df["HP_INSTALLED"] = np.where(
        (mcs_and_epc_hp & ~epc_entry_before_mcs), True, df["HP_INSTALLED"]
    )

    # -----
    # MCS and EPC HP entry, EPC entry before MCS install
    # # We want to discard that option as it should not happen!
    df["HP_INSTALLED"] = np.where(
        (mcs_and_epc_hp & epc_entry_before_mcs), False, df["HP_INSTALLED"]
    )
    df["HP_INSTALL_DATE"] = np.where(
        (mcs_and_epc_hp & epc_entry_before_mcs), np.nan, df["HP_INSTALL_DATE"]
    )

    # # Heat pump: yes
    # # Install date: first HP mention or MCS install date, earlier of the two
    # df["HP_INSTALLED"] = np.where((mcs_and_epc_hp), True, df["HP_INSTALLED"])
    # df["HP_INSTALL_DATE"] = np.where(
    #     (mcs_and_epc_hp),
    #     df[["FIRST_HP_MENTION", "HP_INSTALL_DATE"]].min(axis=1),
    #     df["HP_INSTALL_DATE"],
    # )
    # -----

    # MCS but no EPC HP, with same year EPC entry or later
    # Heat pump: yes
    # Install date: no need to change HP Install Date

    df["HP_INSTALLED"] = np.where(
        (no_epc_but_mcs_hp & ~epc_entry_before_mcs), True, df["HP_INSTALLED"]
    )

    # # ---
    # # MCS but no EPC HP, with MCS before EPC entry
    # # We want to discard that option as it should not happen!

    # df["HP_INSTALLED"] = np.where(
    #     (no_epc_but_mcs_hp & mcs_before_epc_entry), False, df["HP_INSTALLED"]
    # )
    # df["HP_INSTALL_DATE"] = np.where(
    #     (no_epc_but_mcs_hp & mcs_before_epc_entry), np.nan, df["HP_INSTALL_DATE"]
    # )

    # ---
    # MCS but no EPC HP with EPC entry before MCS
    # Set current instance to no heat pump
    # but create duplicate with MCS install date if no future EPC HP mention

    # Get samples for which there is no future EPC HP mention
    # and create duplicate with MCS install data
    no_future_hp_entry = df[
        no_epc_but_mcs_hp & epc_entry_before_mcs & (df["HAS_HP_AT_SOME_POINT"] == False)
    ].copy()

    # Update heat pump data and add newly created entries to other data
    no_future_hp_entry["HP_INSTALLED"] = True
    no_future_hp_entry["HAS_HP_AT_SOME_POINT"] == True

    df["HP_INSTALLED"] = np.where(
        (no_epc_but_mcs_hp & epc_entry_before_mcs), False, df["HP_INSTALLED"]
    )
    df["HP_INSTALL_DATE"] = np.where(
        (no_epc_but_mcs_hp & epc_entry_before_mcs), np.nan, df["HP_INSTALL_DATE"]
    )

    df = pd.concat([df, no_future_hp_entry])

    df["INSPECTION_YEAR"] = round(df["INSPECTION_DATE_AS_NUM"] / 10000.0)
    df["HP_INSTALL_YEAR"] = round(df["HP_INSTALL_DATE"] / 10000.0)
    df["FIRST_HP_MENTION_YEAR"] = round(df["FIRST_HP_MENTION"] / 10000.0)

    print("Total", df.shape)
    print("MCS and EPC", df[mcs_and_epc_hp].shape)
    print("no MCS or EPC", df[no_mcs_or_epc].shape)
    print("either", df[either_hp].shape)
    print("no MCS but EPC", df[no_mcs_but_epc_hp].shape)
    print("no epc but mcs", df[no_epc_but_mcs_hp].shape)

    print(
        "no HP mention: epc_entry_before mcs",
        df[no_epc_but_mcs_hp & epc_entry_before_mcs].shape,
    )

    print(
        "no HP mention: epc_entry_after/same mcs",
        df[no_epc_but_mcs_hp & ~epc_entry_before_mcs].shape,
    )

    print(
        "EPC HP mention: epc_entry_before mcs",
        df[mcs_and_epc_hp & epc_entry_before_mcs].shape,
    )

    print(
        "EPC HP mention: epc_entry_after/same mcs",
        df[mcs_and_epc_hp & ~epc_entry_before_mcs].shape,
    )

    return df


def get_aggregated_temp_data(
    df,
    source_year,
    target_year,
    postcode_level="POSTCODE_UNIT",
    drop_features=[],
):
    """Get aggregated data for source and target year data
    and derive target variables HP coverage and growth.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with EPC data and MCS install dates.

    source_year : int
        Use data up to source year for training.

    target_year : int
        Use target variables derived
        from data up to target year as ground truths.

    postcode_level : str, default="POSTCODE_UNIT"
        Postcode level on which to aggregate features.

    drop_features : list, default=[]
        Features to discard.


    Return
    ---------
    source_year : pandas.DataFrame:
        Source year with added target variables from target year."""

    # Set the postcode levels
    postcode_levels = [
        "POSTCODE_AREA",
        "POSTCODE_DISTRICT",
        "POSTCODE_SECTOR",
        "POSTCODE_UNIT",
        "POSTCODE",
    ]

    # Add unnecessary postcode levels to drop features
    drop_features += [
        postcode for postcode in postcode_levels if postcode != postcode_level
    ]

    # Get data for data up to t (source year) and t+1 (target year)
    source_year = feature_engineering.filter_by_year(
        df, "BUILDING_ID", source_year, selection="latest entry", up_to=True
    )
    target_year = feature_engineering.filter_by_year(
        df, "BUILDING_ID", target_year, selection="latest entry", up_to=True
    )

    # Drop unnecessary features
    source_year = source_year.drop(columns=drop_features + ["BUILDING_ID"])

    # Encode aggregated features
    source_year = data_aggregation.encode_aggregated_features(
        source_year, postcode_level, ordinal_features, ["HP_INSTALLED"]
    )

    # Get target variables (growth and coverage)
    target_variables = data_aggregation.get_target_variables(
        df, source_year, target_year, postcode_level, normalise="total"
    )

    # Drop the variable HP_INSTALLED as it is no longer relevant
    source_year.drop(columns=["HP_INSTALLED"], inplace=True)

    # Merge with target variables
    source_year = pd.merge(
        source_year,
        target_variables[
            ["HP_COVERAGE_CURRENT", "HP_COVERAGE_FUTURE", "GROWTH", postcode_level]
        ],
        on=postcode_level,
    )

    # Remove samples with unreasonable growth
    samples_to_discard = list(
        target_variables.loc[target_variables["GROWTH"] < 0.0][postcode_level]
    )
    source_year = source_year[~source_year[postcode_level].isin(samples_to_discard)]

    return source_year


def epc_sample_loading(subset="5m", usecols=None, preload=True):
    """Load respective subset of EPC samples.

    Parameters
    ----------
    subset : str, default='5m'
        Subset name of EPC dataset.
        '5m' : 5 million samples
        'complete' : complete set of samples of ~21 million samples

    preload : boolean, default=True
        Whether or not to load the 5m samples from file.

    Return
    ---------
    epc_df : pandas.DataFrame:
        Respective subset of EPC dataset."""

    # Load the complete EPC set
    if subset == "complete":

        epc_df = epc_data.load_preprocessed_epc_data(
            version="preprocessed", usecols=usecols
        )

    # Load only 5 million samples
    elif subset == "5m":

        if preload:
            epc_df = pd.read_csv(
                SUPERVISED_MODEL_OUTPUT + "epc_df_{}.csv".format(subset), dtype=dtypes
            )

        # Select a subset of at least 5m samples keeping postcodes complete
        else:

            # Load full set
            epc_df = epc_data.load_preprocessed_epc_data(
                version="preprocessed", usecols=usecols
            )

            # Reduce to fewer samples based on postcode completeness
            epc_df = select_samples_by_postcode_completeness(
                epc_df, min_samples=5000000
            )

            # Save output
            epc_df.to_csv(
                SUPERVISED_MODEL_OUTPUT + "epc_df_{}.csv".format(subset), index=False
            )
    else:
        raise IOError("Subset '{}' is not defined.".format(subset))

    return epc_df


def feature_encoding_for_hp_status(epc_df, subset="5m"):
    """Feature encode the EPC dataset for the HP status predictions.

    Parameters
    ----------
    epc_df : pandas.Dataframe
        EPC dataset with unencoded features.

    subset : str, default='5m'
        Subset name of EPC dataset.
        '5m' : 5 million samples
        'complete' : complete set of samples of ~21 million samples

    Return
    ---------
    epc_df : pandas.DataFrame:
        Encoded EPC dataset."""

    # Encode EPC features
    epc_df = feature_encoding.feature_encoding_pipeline(
        epc_df,
        ordinal_features,
        reduce_categories=True,
        onehot_features="auto",
        unaltered_features=[
            "POSTCODE",
            "POSTCODE_DISTRICT",
            "POSTCODE_SECTOR",
            "POSTCODE_UNIT",
            "HP_INSTALLED",
            "N_ENTRIES_BUILD_ID",
            "POSTCODE_AREA",
        ],
        drop_features=drop_features,
    )

    # Save encoded features
    epc_df.to_csv(
        SUPERVISED_MODEL_OUTPUT + "epc_df_{}_encoded.csv".format(subset), index=False
    )

    return epc_df


def data_preprocessing(epc_df, encode_features=False, subset="5m"):
    """Load EPC and MCS data, fix heat pump install dates, encode features.

    Parameters
    ----------
    epc_df : pandas.Dataframe
        Unprocessed EPC dataset.

    encode_features : bool, default=False
        Encode the features in the end (e.g. for HP status).

    subset : str, default='5m'
        Subset name of EPC dataset.
        '5m' : 5 million samples
        'complete' : complete set of samples of ~21 million samples

    Return
    ---------
    epc_df : pandas.DataFrame:
        Accordingly preprocessed EPC dataset."""

    FIGPATH = SUPERVISED_MODEL_FIG_PATH

    # Add the IMD info
    imd_df = deprivation_data.get_gb_imd_data()
    epc_df = deprivation_data.merge_imd_with_other_set(
        imd_df, epc_df, postcode_label="POSTCODE"
    )

    # Fix the HP install dates and add different postcode levels
    epc_df = manage_hp_install_dates(epc_df)
    epc_df = data_aggregation.get_postcode_levels(epc_df)
    epc_df.to_csv(
        SUPERVISED_MODEL_OUTPUT + "epc_df_{}_preprocessed.csv".format(subset),
        index=False,
    )

    # Feature encoding
    if encode_features:

        print("encoding")

        epc_df = feature_encoding.feature_encoding_pipeline(
            epc_df,
            ordinal_features,
            reduce_categories=True,
            onehot_features="auto",
            unaltered_features=[
                "POSTCODE",
                "POSTCODE_DISTRICT",
                "POSTCODE_SECTOR",
                "POSTCODE_UNIT",
                "HP_INSTALLED",
                "N_ENTRIES_BUILD_ID",
                "POSTCODE_AREA",
            ],
            drop_features=drop_features,
        )

        epc_df.to_csv(
            SUPERVISED_MODEL_OUTPUT + "epc_df_{}_encoded.csv".format(subset),
            index=False,
        )

    return epc_df


def main():

    subset = "5m"

    # Loading the data and preprocessing (5m subset)
    # =================================================

    # epc_df = epc_sample_loading(subset="5m", preload=True)
    # epc_df = data_preprocessing(epc_df, encode_features=False)

    # Equivalent to:
    epc_df = pd.read_csv(
        SUPERVISED_MODEL_OUTPUT + "epc_df_{}_preprocessed.csv".format(subset)
    )

    # Encoding features for Household HP Status (5m subset)
    # =================================================

    # epc_df = pd.read_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_preprocessed.csv")
    # epc_df = data_preprocessing.feature_encoding_for_hp_status(epc_df)

    # Equivalent to:
    epc_df = pd.read_csv(
        SUPERVISED_MODEL_OUTPUT + "epc_df_{}_encoded.csv".format(subset)
    )

    # Aggregating and encoding features for Area HP Growth (5m subset)
    # =================================================

    epc_df = pd.read_csv(
        SUPERVISED_MODEL_OUTPUT + "epc_df_{}_preprocessed.csv".format(subset)
    )

    drop_features = ["HP_INSTALL_DATE"]
    postcode_level = "POSTCODE_UNIT"

    aggr_temp = get_aggregated_temp_data(
        epc_df, 2015, 2018, postcode_level, drop_features=drop_features
    )


if __name__ == "__main__":
    # Execute only if run as a script
    main()
