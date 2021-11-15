# File: heat_pump_adoption_modelling/pipeline/MCS/load_mcs.py
"""Loading MCS HP records and relevant EPC records."""

import pandas as pd
import re

from heat_pump_adoption_modelling import PROJECT_DIR

max_cost = 5000000
max_capacity = 100
epc_address_fields = ["ADDRESS1", "POSTTOWN", "POSTCODE"]
epc_characteristic_fields = [
    "TOTAL_FLOOR_AREA",
    "CONSTRUCTION_AGE_BAND",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "BUILDING_REFERENCE_NUMBER",
    "INSPECTION_DATE",
    "FLOOR_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "HP_INSTALLED",
]
mcs_path = "inputs/MCS_data/mcs_heat_pumps.xlsx"
epc_path = "outputs/EPC_data/preprocessed_data/Q2_2021/EPC_GB_preprocessed.csv"
inflation_path = "inputs/data/inflation.csv"
# TODO: put in config

mcs_colnames_dict = {
    "Version Number": "version",
    "Commissioning Date": "date",
    "Address Line 1": "address_1",
    "Address Line 2": "address_2",
    "Address Line 3": "address_3",
    "Postcode": "postcode",
    "Local Authority": "local_authority",
    "Total Installed Capacity": "capacity",
    "Green Deal Installation?": "green_deal",
    "Products": "products",
    "Flow temp/SCOP ": "flow_scop",
    "Technology Type": "tech_type",
    " Installation Type": "installation_type",
    "Installation New at Commissioning Date?": "new",
    "Renewable System Design": "design",
    "Annual Space Heating Demand": "heat_demand",
    "Annual Water Heating Demand": "water_demand",
    "Annual Space Heating Supplied": "heat_supplied",
    "Annual Water Heating Supplied": "water_supplied",
    "Installation Requires Metering?": "metering",
    "RHI Metering Status": "rhi_status",
    "RHI Metering Not Ready Reason": "rhi_not_ready",
    "Number of MCS Certificates": "n_certificates",
    "Alternative Heating System Type": "alt_type",
    "Alternative Heating System Fuel Type": "alt_fuel",
    "Overall Cost": "cost",
}


def load_domestic_hps():
    """Load domestic MCS HP installation data from file
    and performs some cleaning.

    Return
    ----------
    dhps: pandas.Dataframe
        All MCS heat pump installation records with 'domestic' installation type.
    """

    hps = (
        pd.read_excel(
            PROJECT_DIR / mcs_path,
            dtype={
                "Address Line 1": str,
                "Address Line 2": str,
                "Address Line 3": str,
                "Postcode": str,
            },
        )
        .rename(columns=mcs_colnames_dict)
        .convert_dtypes()
        .drop_duplicates()
    )

    # Filter to domestic installations
    dhps = (
        hps[hps["installation_type"].str.strip() == "Domestic"]
        .drop(columns="installation_type")
        .reset_index(drop=True)
    )

    # Drop records with a newer version
    # TODO: consider asking for the MCS certificate number field
    # instead of just using the address
    most_recent_indices = dhps.groupby(["address_1", "address_2", "address_3"])[
        "version"
    ].idxmax()

    dhps = dhps.iloc[most_recent_indices]

    # Extract information from product column
    product_regex_dict = {
        "product_id": "MCS Product Number: ([^\|]+)",
        "product_name": "Product Name: ([^\|]+)",
        "manufacturer": "License Holder: ([^\|]+)",
        "flow_temp": "Flow Temp: ([^\|]+)",
        "scop": "SCOP: ([^\)]+)",
    }

    for product_feat, regex in product_regex_dict.items():
        dhps[product_feat] = [
            re.search(regex, product).group(1).strip() for product in dhps.products
        ]

    # Add RHI field - any "Unspecified" values in rhi_status field signify
    # that the installation is not for DRHI, missing values are unknown
    dhps["rhi"] = True
    dhps.loc[(dhps["rhi_status"] == "Unspecified"), "rhi"] = False
    dhps["rhi"].mask(dhps["rhi_status"].isna())

    # Replace unreasonable cost and capacity values with NA
    dhps["cost"] = dhps["cost"].mask((dhps["cost"] == 0) | (dhps["cost"] > max_cost))
    dhps["capacity"] = dhps["capacity"].mask(dhps["capacity"] > max_capacity)
    dhps["flow_temp"] = pd.to_numeric(dhps["flow_temp"])
    dhps["flow_temp"] = dhps["flow_temp"].mask(dhps["flow_temp"] <= 0)
    dhps["scop"] = pd.to_numeric(dhps["scop"].mask(dhps["scop"] == "Unspecified"))
    dhps["scop"] = dhps["scop"].mask(dhps["scop"] == 0)
    dhps["year"] = dhps["date"].dt.year

    dhps = dhps.reset_index(drop=True)

    return dhps


def load_epcs():
    """Load relevant columns of EPC records.

    Return
    ----------
    epcs: pandas.Dataframe
        EPC records, columns specified in config.
    """
    epcs = pd.read_csv(
        PROJECT_DIR / epc_path,
        usecols=epc_address_fields + epc_characteristic_fields,
    )

    return epcs


def load_inflation():
    inflation = pd.read_csv(PROJECT_DIR / inflation_path).rename(
        columns={
            "Year": "year",
            "Multiplier to Use for 2021\n [Combined] Overall Index": "multiplier",
        }
    )

    return inflation
