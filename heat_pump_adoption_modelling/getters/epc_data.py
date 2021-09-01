import pandas as pd
import numpy as np
from zipfile import ZipFile

from heat_pump_adoption_modelling import PROJECT_DIR

# if this doesn't work, unzip in command line...
# it's too big to unzip through Finder
def extract_epc():
    with ZipFile(PROJECT_DIR / "inputs/data/epc.zip", "r") as zip:
        print("Files to be extracted:")
        zip.printdir()
        print("Extracting files...")
        zip.extractall(path=PROJECT_DIR / "inputs/data")


def read_epc():
    """Reads EPC data from CSV. Takes a while!
    Currently it seems to just hang...
    """
    epc = pd.read_csv(
        PROJECT_DIR / "inputs/data/epc.csv",
        engine="python",  # slower, but errors if default engine (C) is used
        on_bad_lines="warn",
        #    skiprows=5674127,
    )
    epc = epc.drop(columns="Unnamed: 0")
    print("CSV read! Adding heat pump column...")
    epc["HEAT_PUMP"] = epc.FINAL_HEATING_SYSTEM == "Heat pump"
    print("Done!")

    return epc


def get_epc_sample(full_data, sample_size):
    """Takes a subset of the full EPC dataset to make exploring easier."""
    rand_ints = np.random.choice(len(full_data), size=sample_size)
    sample = epc.iloc[rand_ints]

    return sample


def vc(variable):
    return sample[variable].value_counts()
