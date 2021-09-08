# File: getters/epc_data.py
"""
Extracting and loading the EPC data.
"""

# ---------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from zipfile import ZipFile

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

# ---------------------------------------------------------------------------------

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)


def extract_epc_data():
    """Extract EPC data from zip file."""

    # Filename
    ZIP_FILE_PATH = str(PROJECT_DIR) + config["CLEANSED_EPC_DATA_ZIP_PATH"]

    # Unzip EPC data
    with ZipFile(ZIP_FILE_PATH, "r") as zip:

        print("Extracting...\n{}".format(zip.filename))
        zip.extractall(str(PROJECT_DIR) + "/inputs/")
        print("Done!")


def load_cleansed_EPC():
    """Load the Cleansed EPC dataset.

    Return
    ----------
    cleansed_epc : pandas.DataFrame
        Cleansed EPC datast as dataframe."""

    print("Loading cleansed EPC data... This will take a moment.")
    cleansed_epc = pd.read_csv(
        str(PROJECT_DIR) + config["CLEANSED_EPC_DATA_PATH"],
        low_memory=False
        # CW: engine="python",  # slower, but errors if default engine (C) is used
        # on_bad_lines="warn",
        #    skiprows=5674127,  # JS: I don't think we need any of this...
    )

    # Drop first column
    cleansed_epc = cleansed_epc.drop(columns="Unnamed: 0")

    # Add HP feature
    cleansed_epc["HEAT_PUMP"] = cleansed_epc.FINAL_HEATING_SYSTEM == "Heat pump"
    print("Done!")

    return cleansed_epc


def get_epc_sample(full_df, sample_size):
    """Randomly sample a subset of the full data.
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


"""
def vc(variable):
    return sample[variable].value_counts()  # JS: Do we really need that?
"""


def main():
    """Main function for testing."""
    extract_epc_data()
    load_cleansed_EPC()


if __name__ == "__main__":
    # Execute only if run as a script
    main()
