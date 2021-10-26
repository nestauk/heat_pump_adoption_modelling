# File: heat_pump_adoption_modelling/pipeline/MCS/load_mcs.py
"""Loading MCS HP records and relevant EPC records."""

import pandas as pd
import numpy as np
import re

from heat_pump_adoption_modelling import PROJECT_DIR

max_cost = 5000000
epc_address_fields = ["ADDRESS1", "POSTTOWN", "POSTCODE"]
epc_characteristic_fields = [
    "TOTAL_FLOOR_AREA",
    "CONSTRUCTION_AGE_BAND",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "HP_INSTALLED",
]
# TODO: put in config


def load_domestic_hps():
    """Loads domestic MCS HP installation data from file
    and performs some cleaning.

    Return
    ----------
    dhps: pandas.Dataframe
        All MCS heat pump installation records with 'domestic' installation type.
    """

    hps = (
        pd.read_excel(
            PROJECT_DIR
            / "inputs/NESTA data files/Heat pump installations 2010 to 31082021.xlsx",
            dtype={
                "Address Line 1": str,
                "Address Line 2": str,
                "Address Line 3": str,
                "Postcode": str,
            },
        )
        .rename(
            columns={
                "Commissioning Date": "date",
                "Address Line 1": "address_1",
                "Address Line 2": "address_2",
                "Address Line 3": "address_3",
                "Postcode": "postcode",
                "Local Authority": "local_authority",
                "Products": "products",
                "Technology Type": "tech_type",
                " Installation Type": "installation_type",
                "Installation New at Commissioning Date?": "new",
                "Annual Space Heating Demand": "heat_demand",
                "Annual Space Heating Supplied": "heat_supplied",
                "RHI?": "rhi",
                "Alternative Heating System Type": "alt_type",
                "Alternative Heating System Fuel Type": "alt_fuel",
                "Overall Cost": "cost",
            }
        )
        .convert_dtypes()
        .drop_duplicates()  # maybe we shouldn't do this? if e.g. two identical HPs installed on the same day
    )

    # Make RHI field values easier to use
    hps["rhi"] = hps["rhi"].replace(
        {
            "RHI Installation ": True,
            "Not Domestic RHI installation ": False,
            "Unspecified": np.nan,
        }
    )

    # Filter to domestic installations
    dhps = (
        hps[hps["installation_type"].isin(["Domestic", "Domestic "])]
        .drop(columns="installation_type")
        .reset_index(drop=True)
    )

    # Extract information from product column
    regex_dict = {
        "product_name": "Product Name: ([^\|]+)",
        "manufacturer": "License Holder: ([^\|]+)",
        "flow_temp": "Flow Temp: ([^\|]+)",
        "scop": "SCOP: ([^\)]+)",
    }

    for key, value in regex_dict.items():
        dhps[key] = [
            re.search(value, product).group(1).strip() for product in dhps.products
        ]

    # Replace unreasonable cost values with NA
    dhps["cost"] = dhps["cost"].mask((dhps["cost"] == 0) | (dhps["cost"] > max_cost))

    return dhps


def load_epcs():
    """Loads relevant columns of EPC records.

    Return
    ----------
    epcs: pandas.Dataframe
        EPC records, columns specified in config.
    """
    epcs = pd.read_csv(
        PROJECT_DIR
        / "outputs/EPC_data/preprocessed_data/Q2_2021/EPC_GB_preprocessed_and_deduplicated.csv",
        usecols=epc_address_fields + epc_characteristic_fields,
    )

    return epcs
