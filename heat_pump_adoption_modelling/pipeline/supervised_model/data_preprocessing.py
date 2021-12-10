from heat_pump_adoption_modelling.getters import epc_data, deprivation_data
from heat_pump_adoption_modelling import PROJECT_DIR

from heat_pump_adoption_modelling.pipeline.supervised_model import data_aggregation
from heat_pump_adoption_modelling.pipeline.encoding import (
    feature_encoding,
    category_reduction,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path


import numpy as np
import pandas as pd

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
    "Unnamed: 0",
    "original_address",
    "FIRST_HP_MENTION",
    "INSPECTION_YEAR",
    "HP_INSTALL_YEAR",
    "FIRST_HP_MENTION_YEAR",
    "MCS_AVAILABLE",
    "HAS_HP_AT_SOME_POINT",
    "HP_TYPE",
]

dtypes = {
    "BUILDING_REFERENCE_NUMBER": int,
    "ADDRESS1": str,
    "ADDRESS2": str,
    "POSTTOWN": str,
    "POSTCODE": str,
    "INSPECTION_DATE": str,
    "LODGEMENT_DATE": str,
    "ENERGY_CONSUMPTION_CURRENT": float,
    "TOTAL_FLOOR_AREA": float,
    "CURRENT_ENERGY_EFFICIENCY": int,
    "CURRENT_ENERGY_RATING": str,
    "POTENTIAL_ENERGY_RATING": str,
    "CO2_EMISS_CURR_PER_FLOOR_AREA": float,
    "WALLS_ENERGY_EFF": str,
    "ROOF_ENERGY_EFF": str,
    "FLOOR_ENERGY_EFF": str,
    "WINDOWS_ENERGY_EFF": str,
    "MAINHEAT_DESCRIPTION": str,
    "MAINHEAT_ENERGY_EFF": str,
    "MAINHEATC_ENERGY_EFF": str,
    "SHEATING_ENERGY_EFF": str,
    "HOT_WATER_ENERGY_EFF": str,
    "LIGHTING_ENERGY_EFF": str,
    "CO2_EMISSIONS_CURRENT": float,
    "HEATING_COST_CURRENT": float,
    "HEATING_COST_POTENTIAL": float,
    "HOT_WATER_COST_CURRENT": float,
    "HOT_WATER_COST_POTENTIAL": float,
    "LIGHTING_COST_CURRENT": float,
    "LIGHTING_COST_POTENTIAL": float,
    "CONSTRUCTION_AGE_BAND": str,
    "FLOOR_HEIGHT": float,
    "EXTENSION_COUNT": float,
    "FLOOR_LEVEL": float,
    "GLAZED_AREA": float,
    "NUMBER_HABITABLE_ROOMS": str,
    "NUMBER_HEATED_ROOMS": str,
    "LOCAL_AUTHORITY_LABEL": str,
    "MAINS_GAS_FLAG": str,
    "MAIN_FUEL": str,
    "MAIN_HEATING_CONTROLS": float,
    "MECHANICAL_VENTILATION": str,
    "ENERGY_TARIFF": str,
    "MULTI_GLAZE_PROPORTION": float,
    "GLAZED_TYPE": str,
    "PHOTO_SUPPLY": float,
    "SOLAR_WATER_HEATING_FLAG": str,
    "TENURE": str,
    "TRANSACTION_TYPE": str,
    "WIND_TURBINE_COUNT": float,
    "BUILT_FORM": str,
    "PROPERTY_TYPE": str,
    "COUNTRY": str,
    "CONSTRUCTION_AGE_BAND_ORIGINAL": str,
    "ENTRY_YEAR": str,
    "ENTRY_YEAR_INT": float,
    "INSPECTION_DATE_AS_NUM": int,
    "UNIQUE_ADDRESS": str,
    "BUILDING_ID": str,
    "N_ENTRIES": str,
    "N_ENTRIES_BUILD_ID": str,
    "HEATING_SYSTEM": str,
    "HEATING_FUEL": str,
    "HP_INSTALLED": bool,
    "HP_TYPE": str,
    "CURR_ENERGY_RATING_NUM": float,
    "ENERGY_RATING_CAT": str,
    "DIFF_POT_ENERGY_RATING": float,
}

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

# Get paths
MERGED_MCS_EPC = str(PROJECT_DIR) + config["MERGED_MCS_EPC"]
SUPERVISED_MODEL_OUTPUT = str(PROJECT_DIR) + config["SUPERVISED_MODEL_OUTPUT"]
SUPERVISED_MODEL_FIG_PATH = str(PROJECT_DIR) + config["SUPERVISED_MODEL_FIG_PATH"]


def select_samples_by_postcode_completeness(
    df, min_samples=5000000, start=250000, interval=1000
):

    for i in range(start, min_samples, interval):
        grouped_by = (
            df.groupby("POSTCODE")
            .size()
            .reset_index(name="count")[:i]
            .sample(frac=1, random_state=42)
        )
        n_samples = grouped_by["count"].sum()

        if n_samples > min_samples:
            sample_ids = list(grouped_by["POSTCODE"])
            df_reduced = df.loc[df["POSTCODE"].isin(sample_ids)]
            return df_reduced


def merge_epc_with_mcs(epc_df):

    epc_df["original_address"] = (
        epc_df["ADDRESS1"] + epc_df["ADDRESS2"] + epc_df["POSTCODE"]
    )
    epc_df["original_address"] = (
        epc_df["original_address"]
        .str.strip()
        .str.lower()
        .replace(r"\s+", "", regex=True)
    )

    mcs_data = pd.read_csv(
        MERGED_MCS_EPC,
        usecols=["date", "tech_type", "original_address"],
    )

    mcs_data = mcs_data.loc[~mcs_data["original_address"].isna()]
    mcs_data["original_address"] = (
        mcs_data["original_address"]
        .str.strip()
        .str.lower()
        .replace(r"\s+", "", regex=True)
    )
    mcs_data.columns = ["HP_INSTALL_DATE", "Type of HP", "original_address"]
    mcs_data["HP_INSTALL_DATE"] = (
        mcs_data["HP_INSTALL_DATE"]
        .str.strip()
        .str.lower()
        .replace(r"-", "", regex=True)
        .astype("float")
    )

    date_dict = mcs_data.set_index("original_address").to_dict()["HP_INSTALL_DATE"]
    epc_df["HP_INSTALL_DATE"] = epc_df["original_address"].map(date_dict)

    return epc_df


def add_mcs_install_dates(df):

    df = merge_epc_with_mcs(df)

    first_hp_mention = (
        df.loc[df["HP_INSTALLED"] == True]
        .groupby("BUILDING_ID")
        .min(["INSPECTION_DATE_AS_NUM"])
        .reset_index()[["INSPECTION_DATE_AS_NUM", "BUILDING_ID"]]
    )
    first_hp_mention.columns = ["FIRST_HP_MENTION", "BUILDING_ID"]
    df = pd.merge(df, first_hp_mention, on="BUILDING_ID", how="outer")

    df["INSPECTION_YEAR"] = round(df["INSPECTION_DATE_AS_NUM"] / 10000.0)
    df["HP_INSTALL_YEAR"] = round(df["HP_INSTALL_DATE"] / 10000.0)
    df["FIRST_HP_MENTION_YEAR"] = round(df["FIRST_HP_MENTION"] / 10000.0)

    # If no HP Install date, set MCS availabibility to False
    df["MCS_AVAILABLE"] = np.where(df["HP_INSTALL_DATE"].isna(), False, True)

    # If no first mention of HP, then set has
    df["HAS_HP_AT_SOME_POINT"] = np.where(df["FIRST_HP_MENTION"].isna(), False, True)

    mcs_available = df["MCS_AVAILABLE"] == True
    no_mcs_or_epc = (~df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == False)
    no_mcs_but_epc_hp = (~df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == True)
    mcs_and_epc_hp = (df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == True)
    no_epc_but_mcs_hp = (df["MCS_AVAILABLE"]) & (df["HP_INSTALLED"] == False)
    either_hp = (df["MCS_AVAILABLE"]) | (df["HP_INSTALLED"] == True)

    epc_entry_before_mcs = df["INSPECTION_YEAR"] < df["HP_INSTALL_YEAR"]
    mcs_before_epc_entry = df["INSPECTION_YEAR"] > df["HP_INSTALL_YEAR"]
    epc_entry_same_as_mcs = df["INSPECTION_YEAR"] == df["HP_INSTALL_YEAR"]

    # -----
    # NO MCS/EPC HP entry
    df["HP_INSTALLED"] = np.where((no_mcs_or_epc), False, df["HP_INSTALLED"])
    df["HP_INSTALL_DATE"] = np.where((no_mcs_or_epc), np.nan, df["HP_INSTALL_DATE"])

    # -----
    # No MCS entry but EPC HP
    df["HP_INSTALLED"] = np.where((no_mcs_but_epc_hp), True, df["HP_INSTALLED"])
    df["HP_INSTALL_DATE"] = np.where(
        (no_mcs_but_epc_hp), df["FIRST_HP_MENTION"], df["HP_INSTALL_DATE"]
    )

    # -----
    # MCS and EPC HP entry
    df["HP_INSTALLED"] = np.where((mcs_and_epc_hp), True, df["HP_INSTALLED"])
    df["HP_INSTALL_DATE"] = np.where(
        (mcs_and_epc_hp),
        df[["FIRST_HP_MENTION", "HP_INSTALL_DATE"]].min(axis=1),
        df["HP_INSTALL_DATE"],
    )
    # -----
    # MCS but no EPC HP, with same year EPC entry
    # No need to chnage HP Install Date

    df["HP_INSTALLED"] = np.where(
        (no_epc_but_mcs_hp & epc_entry_same_as_mcs), True, df["HP_INSTALLED"]
    )

    # ---
    # MCS but no EPC HP, with MCS before EPC entry
    # We want to discard that option as it should not happen!

    df["HP_INSTALLED"] = np.where(
        (no_epc_but_mcs_hp & mcs_before_epc_entry), False, df["HP_INSTALLED"]
    )
    df["HP_INSTALL_DATE"] = np.where(
        (no_epc_but_mcs_hp & mcs_before_epc_entry), np.nan, df["HP_INSTALL_DATE"]
    )

    # ---
    # MCS but EPC HP with MCS after EPC entry
    # Set current instance to no heat pump but duplicate with MCS install date if no future EPC HP mention

    df["HP_INSTALLED"] = np.where(
        (no_epc_but_mcs_hp & epc_entry_before_mcs), False, df["HP_INSTALLED"]
    )
    df["HP_INSTALL_DATE"] = np.where(
        (no_epc_but_mcs_hp & epc_entry_before_mcs), np.nan, df["HP_INSTALL_DATE"]
    )

    # Get samples for which there is no future EPC HP mention and duplicate with MCS install data
    no_future_hp_entry = df[
        no_epc_but_mcs_hp & epc_entry_before_mcs & (df["HAS_HP_AT_SOME_POINT"] == False)
    ].copy()

    no_future_hp_entry["HP_INSTALLED"] = True
    no_future_hp_entry["HAS_HP_AT_SOME_POINT"] == True

    df = pd.concat([df, no_future_hp_entry])

    return df


def get_aggregated_temp_data(
    df,
    source_year,
    target_year,
    postcode_level="POSTCODE_UNIT",
    drop_features=[],
):

    postcode_levels = [
        "POSTCODE_AREA",
        "POSTCODE_DISTRICT",
        "POSTCODE_SECTOR",
        "POSTCODE_UNIT",
        "POSTCODE",
    ]

    drop_features += [
        postcode for postcode in postcode_levels if postcode != postcode_level
    ]

    # Get data for t and t+1
    source_year = feature_engineering.filter_by_year(
        df, "BUILDING_ID", source_year, selection="latest entry", up_to=True
    )
    target_year = feature_engineering.filter_by_year(
        df, "BUILDING_ID", target_year, selection="latest entry", up_to=True
    )

    # Drop unnecessary features
    source_year = source_year.drop(columns=drop_features + ["BUILDING_ID"])

    # Encode agglomerated features
    source_year = data_aggregation.encode_agglomerated_features(
        source_year, postcode_level, ordinal_features, ["HP_INSTALLED"]
    )

    # Get target variables (growth and coverage)
    target_variables = data_aggregation.get_target_variables(
        df, source_year, target_year, postcode_level, normalise="total"
    )

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


def epc_sample_loading(subset="5m", preload=True):

    if subset == "complete":

        epc_df = epc_data.load_preprocessed_epc_data(
            version="preprocessed", usecols=None
        )
    elif subset == "5m":

        if preload:
            epc_df = pd.read_csv(
                SUPERVISED_MODEL_OUTPUT + "epc_df_5m.csv", dtype=dtypes
            )
        else:

            epc_df = epc_data.load_preprocessed_epc_data(
                version="preprocessed", usecols=None
            )
            epc_df = select_samples_by_postcode_completeness(
                epc_df, min_sampels=5000000
            )

            epc_df.to_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_5m.csv")
    else:
        raise IOError("Subset '{}' is not defined.".format(subset))

    return epc_df


def feature_encoding_for_hp_status(epc_df):

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

    epc_df.to_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_encoded.csv")

    return epc_df


def data_preprocessing(epc_df, encode_features=False, verbose=True):

    FIGPATH = SUPERVISED_MODEL_FIG_PATH

    # Add the IMD info
    imd_df = deprivation_data.get_gb_imd_data()
    epc_df = deprivation_data.merge_imd_with_other_set(
        imd_df, epc_df, postcode_label="POSTCODE"
    )

    epc_df = add_mcs_install_dates(epc_df)
    epc_df = data_aggregation.get_postcode_levels(epc_df)
    epc_df.to_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_preprocessed.csv")

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

        epc_df.to_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_encoded.csv")

    return epc_df


def main():

    # epc_df = epc_sample_loading(subset="5m", preload=True)
    # epc_df = data_preprocessing(epc_df, encode_features=True)

    epc_df = pd.read_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_preprocessed.csv")

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

    epc_df.to_csv(SUPERVISED_MODEL_OUTPUT + "epc_df_encoded.csv")

    # aggr_temp = get_aggregated_temp_data(2015, 2018, "POSTCODE_UNIT")


if __name__ == "__main__":
    # Execute only if run as a script
    main()
