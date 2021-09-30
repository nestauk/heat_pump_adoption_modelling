# File: heat_pump_adoption_modelling/pipeline/preprocessing/preprocess_epc_data.py
"""Loading and preprocessing the raw EPC data for England, Wales and Scotland."""

# ---------------------------------------------------------------------------------
import os
import re

import pandas as pd
from zipfile import ZipFile

from heat_pump_adoption_modelling import get_yaml_config, Path, PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    feature_engineering,
    data_cleaning,
)

# ---------------------------------------------------------------------------------

# Load config file
epc_data_config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

# Get paths
RAW_ENG_WALES_DATA_PATH = str(PROJECT_DIR) + epc_data_config["RAW_ENG_WALES_DATA_PATH"]
RAW_SCOTLAND_DATA_PATH = str(PROJECT_DIR) + epc_data_config["RAW_SCOTLAND_DATA_PATH"]

RAW_ENG_WALES_DATA_ZIP = str(PROJECT_DIR) + epc_data_config["RAW_ENG_WALES_DATA_ZIP"]
RAW_SCOTLAND_DATA_ZIP = str(PROJECT_DIR) + epc_data_config["RAW_SCOTLAND_DATA_ZIP"]

RAW_EPC_DATA_PATH = str(PROJECT_DIR) + epc_data_config["RAW_EPC_DATA_PATH"]
PREPROC_EPC_DATA_PATH = str(PROJECT_DIR) + epc_data_config["PREPROC_EPC_DATA_PATH"]
PREPROC_EPC_DATA_DEDUPL_PATH = (
    str(PROJECT_DIR) + epc_data_config["PREPROC_EPC_DATA_DEDUPL_PATH"]
)

EPC_FEAT_SELECTION = epc_data_config["EPC_FEAT_SELECTION"]


def extract_data(file_path):
    """Extract data from zip file.

    Parameters
    ----------
    file_path : str
        Path to the file to unzip.

    Return: None"""

    # Check whether file exists
    if not Path(file_path).is_file():
        raise IOError("The file '{}' does not exist.".format(file_path))

    # Get directory
    zip_dir = os.path.dirname(file_path) + "/"

    # Unzip the data
    with ZipFile(file_path, "r") as zip:

        print("Extracting...\n{}".format(zip.filename))
        zip.extractall(zip_dir)
        print("Done!")


def load_Scotland_data(usecols=None, low_memory=False):
    """Load the Scotland EPC data.

    Parameters
    ----------
    usecols : list, default=None
        List of features/columns to load from EPC dataset.
        If None, then all features will be loaded.

    low_memory : bool, default=False
        Internally process the file in chunks, resulting in lower memory use while parsing,
        but possibly mixed type inference.
        To ensure no mixed types either set False, or specify the type with the dtype parameter.

    Return
    ---------
    EPC_certs : pandas.DateFrame
        Scotland EPC certificate data for given features."""

    # If sample file does not exist (probably just not unzipped), unzip the data
    if not [
        file
        for file in os.listdir(RAW_SCOTLAND_DATA_PATH)
        if file.startswith("D_EPC_data_2012_Q4_extract")
    ]:
        extract_data(RAW_SCOTLAND_DATA_ZIP)

    if usecols is not None:
        # Fix columns ("WALLS" features are labeled differently here)
        usecols = [re.sub("WALLS_", "WALL_", col) for col in usecols]
        usecols = [re.sub("POSTTOWN", "POST_TOWN", col) for col in usecols]

    # Get all directories
    all_directories = os.listdir(RAW_SCOTLAND_DATA_PATH)
    directories = [file for file in all_directories if file.endswith(".csv")]

    EPC_certs = [
        pd.read_csv(
            RAW_SCOTLAND_DATA_PATH + file,
            low_memory=low_memory,
            usecols=usecols,
            skiprows=1,  # don't load first row (more ellaborate feature names),
            encoding="ISO-8859-1",
        )
        for file in directories
    ]

    # Concatenate single dataframes into dataframe
    EPC_certs = pd.concat(EPC_certs, axis=0)
    EPC_certs["COUNTRY"] = "Scotland"

    EPC_certs = EPC_certs.rename(
        columns={
            "WALL_ENV_EFF": "WALLS_ENV_EFF",
            "WALL_ENERGY_EFF": "WALLS_ENERGY_EFF",
            "POST_TOWN": "POSTTOWN",
        }
    )

    return EPC_certs


def load_Wales_England_data(subset=None, usecols=None, low_memory=False):
    """Load the England and/or Wales EPC data.

    Parameters
    ----------
    subset : {'England', 'Wales', None}, default=None
        EPC certificate area subset.
        If None, then the data for both England and Wales will be loaded.

    usecols : list, default=None
        List of features/columns to load from EPC dataset.
        If None, then all features will be loaded.

    low_memory : bool, default=False
        Internally process the file in chunks, resulting in lower memory use while parsing,
        but possibly mixed type inference.
        To ensure no mixed types either set False, or specify the type with the dtype parameter.

    Return
    ---------
    EPC_certs : pandas.DateFrame
        England/Wales EPC certificate data for given features."""

    # If sample file does not exist (probably just not unzipped), unzip the data
    if not Path(
        RAW_ENG_WALES_DATA_PATH + "domestic-W06000015-Cardiff/certificates.csv"
    ).is_file():
        extract_data(RAW_ENG_WALES_DATA_ZIP)

    # Get all directories
    directories = [
        dir
        for dir in os.listdir(RAW_ENG_WALES_DATA_PATH)
        if not (dir.startswith(".") or dir.endswith(".txt") or dir.endswith(".zip"))
    ]

    # Set subset dict to select respective subset directories
    start_with_dict = {"Wales": "domestic-W", "England": "domestic-E"}

    # Get directories for given subset
    if subset in start_with_dict:
        directories = [
            dir for dir in directories if dir.startswith(start_with_dict[subset])
        ]

    # Load EPC certificates for given subset
    # Only load columns of interest (if given)
    EPC_certs = [
        pd.read_csv(
            RAW_ENG_WALES_DATA_PATH + directory + "/certificates.csv",
            low_memory=low_memory,
            usecols=usecols,
        )
        for directory in directories
    ]

    # Concatenate single dataframes into dataframe
    EPC_certs = pd.concat(EPC_certs, axis=0)
    EPC_certs["COUNTRY"] = subset

    return EPC_certs


def load_epc_data(subset="GB", usecols=None, low_memory=False):
    """Load and return EPC dataset, or specific subset, as pandas dataframe.

    Parameters
    ----------
    subset : {'GB', 'Wales', 'England', 'Scotland', None}, default='GB'
        EPC certificate area subset.

    usecols : list, default=None
        List of features/columns to load from EPC dataset.
        If None, then all features will be loaded.

    low_memory : bool, default=False
        Internally process the file in chunks, resulting in lower memory use while parsing,
        but possibly mixed type inference.
        To ensure no mixed types either set False, or specify the type with the dtype parameter.

    Return
    ---------
    EPC_certs : pandas.DateFrame
        EPC certificate data for given area and features."""

    all_epc_df = []

    # Get Scotland data
    if subset in ["Scotland", "GB"]:
        epc_Scotland_df = load_Scotland_data(usecols=usecols)
        all_epc_df.append(epc_Scotland_df)

        if subset == "Scotland":
            return epc_Scotland_df

    # Get the Wales/England data
    if subset in ["Wales", "England"]:
        epc_df = load_Wales_England_data(subset, usecols=usecols, low_memory=low_memory)
        return epc_df

    # Merge the two datasets for GB
    elif subset == "GB":

        for country in ["Wales", "England"]:

            epc_df = load_Wales_England_data(
                country, usecols=usecols, low_memory=low_memory
            )
            all_epc_df.append(epc_df)

        epc_df = pd.concat(all_epc_df, axis=0)

        return epc_df

    else:
        raise IOError("'{}' is not a valid subset of the EPC dataset.".format(subset))


def preprocess_data(df, remove_duplicates=True, save_data=True, verbose=True):
    """Preprocess the raw EPC data by cleaning it and removing duplications.
    The data at the different processing steps can be saved.

    The processing steps:

    - raw:
    Merged but otherwise not altered EPC data

    - preprocessed:
    Partially cleaned and with additional features

    - preprocessed_dedupl:
    Same as 'preprocessed' but without duplicates

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe holding EPC data to process.

    remove_duplicates : bool, default=True
        Whether or not to remove duplicates.

    save_data : bool, default=True
        Whether or not to save preprocessed data at different stages (original, cleaned, deduplicated).

    verbose : bool, default=True
        Print number of features and samples after each processing step.

    Return
    ---------
    df : pandas.DataFrame
        Preprocessed EPC dataset."""

    # --------------------------------
    # Raw data
    # --------------------------------

    if save_data:
        # Save unaltered_version
        df.to_csv(RAW_EPC_DATA_PATH, index=False)

    processing_steps = []
    processing_steps.append(("Original data", df.shape[0], df.shape[1]))

    # --------------------------------
    # Preprocessing data
    # --------------------------------

    df = data_cleaning.clean_epc_data(df)
    processing_steps.append(("After cleaning", df.shape[0], df.shape[1]))

    df = feature_engineering.get_additional_features(df)
    processing_steps.append(("After adding features", df.shape[0], df.shape[1]))

    if save_data:
        # Save unaltered_version
        df.to_csv(PREPROC_EPC_DATA_PATH, index=False)

    # --------------------------------
    # Deduplicated data
    # --------------------------------

    if remove_duplicates:

        df = feature_engineering.filter_by_year(
            df, "BUILDING_ID", None, selection="latest entry"
        )
        processing_steps.append(("After removing duplicates", df.shape[0], df.shape[1]))

        if save_data:
            # Save unaltered_version
            df.to_csv(PREPROC_EPC_DATA_DEDUPL_PATH, index=False)

    # --------------------------------
    # Print stats
    # --------------------------------

    if verbose:

        for step in processing_steps:
            print("{}:\t{} samples, {} features".format(step[0], step[1], step[2]))

    return df


def load_and_preprocess_epc_data(
    subset="GB", usecols=EPC_FEAT_SELECTION, remove_duplicates=True, save_data=True
):
    """Load and preprocess the EPC data.

    Parameters
    ----------
    subset : {'GB', 'Wales', 'England', 'Scotland', None}, default='GB'
        EPC certificate area subset.

    usecols : list, default=EPC_FEAT_SELECTION
        List of features/columns to load from EPC dataset.
        By default, a pre-selected list of features (specified in the config file) is used.
        If None, then all features will be loaded.

    remove_duplicates : bool, default=True
        Whether or not to remove duplicates.

    save_data : bool, default=True
        Whether or not to save preprocessed data at different stages (original, cleaned, deduplicated).


    Return
    ---------
    epc_df : pandas.DataFrame
        Preprocessed EPC dataset."""

    # Do not save/overwrite the preprocessed data when not loading entire GB dataset
    # in order to prevent confusion.
    if subset != "GB":
        print(
            "The precessed data will be returned but not be written to file. Change subset to 'GB' or save processed data manually."
        )
        save_data = False

    epc_df = load_epc_data(subset=subset, usecols=usecols)
    epc_df = preprocess_data(
        epc_df, remove_duplicates=remove_duplicates, save_data=save_data
    )
    return epc_df


# ---------------------------------------------------------------------------------


def main():
    """Main function: Loads and preprocessed EPC data with default settings."""

    epc_df = load_and_preprocess_epc_data()
    print(epc_df.head())


if __name__ == "__main__":
    # Execute only if run as a script
    main()
