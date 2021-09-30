# File: getters/epc_data.py
"""Extracting and loading the EPC data."""

# ---------------------------------------------------------------------------------

import os
import pandas as pd
import numpy as np
from zipfile import ZipFile

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

# ---------------------------------------------------------------------------------

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)


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


def load_cleansed_EPC(remove_duplicates=True):
    """Load the cleansed EPC dataset (provided by EST)
    with the option of excluding/including duplicates.

    Parameters
    ----------
    remove_duplicates : bool, default=True.
        Whether or not to remove duplicates.

    Return
    ----------
    cleansed_epc : pandas.DataFrame
        Cleansed EPC datast as dataframe."""

    if remove_duplicates:
        file_path = str(PROJECT_DIR) + config["EST_CLEANSED_EPC_DATA_DEDUPL_PATH"]
    else:
        file_path = str(PROJECT_DIR) + config["EST_CLEANSED_EPC_DATA_PATH"]

    # If file does not exist (probably just not unzipped), unzip the data
    if not Path(file_path).is_file():
        extract_data(file_path + ".zip")

    print("Loading cleansed EPC data... This will take a moment.")
    cleansed_epc = pd.read_csv(file_path, low_memory=False)

    # Drop first column
    cleansed_epc = cleansed_epc.drop(columns="Unnamed: 0")

    # Add HP feature
    cleansed_epc["HEAT_PUMP"] = cleansed_epc.FINAL_HEATING_SYSTEM == "Heat pump"
    print("Done!")

    return cleansed_epc


def load_preprocessed_epc_data(
    version="preprocessed_dedupl", usecols=None, low_memory=False
):
    """Load the EPC dataset including England, Wales and Scotland.
    Select one of the following versions:

        - raw:
        EPC data merged for all countries but otherwise not altered

        - preprocessed:
        Partially cleaned and with additional features

        - preprocessed_dedupl:
        Same as 'preprocessed' but without duplicates

    Parameters
    ----------
    version : str, {'raw', 'preprocessed', 'preprocessed_dedupl'}, default='preprocessed_dedupl'
        The version of the EPC data to load.

    usecols : list, default=None
        List of features/columns to load from EPC dataset.
        If None, then all features will be loaded.

    low_memory : bool, default=False
        Internally process the file in chunks, resulting in lower memory use while parsing,
        but possibly mixed type inference.
        To ensure no mixed types either set False, or specify the type with the dtype parameter.

    Return
    ----------
    epc_df : pandas.DataFrame
        EPC data in the given version."""

    version_path_dict = {
        "raw": "RAW_EPC_DATA_PATH",
        "preprocessed_dedupl": "PREPROC_EPC_DATA_PATH",
        "preprocessed": "PREPROC_EPC_DATA_DEDUPL_PATH",
    }

    # Get the respective file path for version
    file_path = str(PROJECT_DIR) + config[version_path_dict[version]]

    # If file does not exist (likely just not unzipped), unzip the data
    if not Path(file_path).is_file():
        extract_data(file_path + ".zip")

    # Load  data
    epc_df = pd.read_csv(file_path, usecols=usecols, low_memory=low_memory)

    return epc_df


def get_epc_sample(full_df, sample_size):
    """Randomly sample a subset of the full data.

    Parameters
    ----------
    full_df : pandas.DataFrame
        Full dataframe from which to extract a subset.

    sample_size: int
        Size of subset / number of samples.

    Return
    ----------
    sample_df : pandas.DataFrame
        Randomly sampled subset of full dataframe."""

    rand_ints = np.random.choice(len(full_df), size=sample_size)
    sample_df = full_df.iloc[rand_ints]

    return sample_df


def main():
    """Main function for testing."""


if __name__ == "__main__":
    # Execute only if run as a script
    main()
