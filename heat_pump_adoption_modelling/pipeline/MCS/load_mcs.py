# File: heat_pump_adoption_modelling/pipeline/MCS/load_mcs.py
"""Loading MCS heat pump records and inflation multipliers."""

import pandas as pd
import re

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

all_hp_filepath = config["MCS_HP_PATH"]
domestic_hp_filepath = config["MCS_DOMESTIC_HP_PATH"]
max_cost = config["MCS_MAX_COST"]
max_capacity = config["MCS_MAX_CAPACITY"]
inflation_path = config["INFLATION_PATH"]

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

    Parameters
    ----------
    filepath: str
        Path to domestic HP data.
        Function will load data from file if it exists;
        otherwise it will load all HP data, filter
        and save domestic HP data to filepath.

    Return
    ----------
    dhps: pandas.Dataframe
        All MCS heat pump installation records with 'domestic' installation type.
    """

    if Path(str(PROJECT_DIR) + domestic_hp_filepath).is_file():
        dhps = pd.read_csv(str(PROJECT_DIR) + domestic_hp_filepath)

    else:  # load all HP data and process
        hps = (
            pd.read_excel(
                str(PROJECT_DIR) + all_hp_filepath,
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
        dhps["cost"] = dhps["cost"].mask(
            (dhps["cost"] == 0) | (dhps["cost"] > max_cost)
        )
        dhps["capacity"] = dhps["capacity"].mask(dhps["capacity"] > max_capacity)
        dhps["flow_temp"] = pd.to_numeric(dhps["flow_temp"])
        dhps["flow_temp"] = dhps["flow_temp"].mask(dhps["flow_temp"] <= 0)
        dhps["scop"] = pd.to_numeric(dhps["scop"].mask(dhps["scop"] == "Unspecified"))
        dhps["scop"] = dhps["scop"].mask(dhps["scop"] == 0)
        dhps["year"] = dhps["date"].dt.year

        dhps = dhps.reset_index(drop=True)

        dhps.to_csv(str(PROJECT_DIR) + domestic_hp_filepath)

    return dhps


def load_inflation():
    """Load inflation multiplier data.

    Return
    ----------
    inflation: pandas.Dataframe
        Inflation multiplier for each calendar year.
    """

    inflation = pd.read_csv(str(PROJECT_DIR) + inflation_path).rename(
        columns={
            "Year": "year",
            "Multiplier to Use for 2021\n [Combined] Overall Index": "multiplier",
        }
    )

    return inflation
