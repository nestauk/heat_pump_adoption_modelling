# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
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
    "HP_INSTALLED",
]
mcs_path = "inputs/MCS_data/heat pumps 2008 to end of sept 2021 - Copy.xlsx"
epc_path = "outputs/EPC_data/preprocessed_data/Q2_2021/EPC_GB_preprocessed_and_deduplicated.csv"
# TODO: put in config


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
    .rename(
        columns={
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
            # "RHI?": "rhi",
            "Alternative Heating System Type": "alt_type",
            "Alternative Heating System Fuel Type": "alt_fuel",
            "Overall Cost": "cost",
        }
    )
    .convert_dtypes()
    .drop_duplicates()
)

# Make RHI field values easier to use
# Commented out as RHI field has disappeared from the most recent MCS data
# hps["rhi"] = hps["rhi"].replace(
#     {
#         "RHI Installation ": True,
#         "Not Domestic RHI installation ": False,
#         "Unspecified": np.nan,
#     }
# )

hps.head()


# %%
# Filter to domestic installations
dhps = (
    hps[hps["installation_type"].isin(["Domestic", "Domestic "])]  # strip()
    .drop(columns="installation_type")
    .reset_index(drop=True)
)

print(dhps.shape)
dhps.head()


# %%
most_recent_indices = dhps.groupby(["address_1", "address_2", "address_3"])[
    "version"
].idxmax()

dhps = dhps.iloc[most_recent_indices]
print(dhps.shape)
dhps.head()


# %%
dhps.products


# %%
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

print(dhps["scop"].unique())
# dhps['scop'] = pd.to_numeric(dhps['scop']) # includes unspecified
dhps["scop"].unique()


# %%
# Replace unreasonable cost and capacity values with NA
dhps["cost"] = dhps["cost"].mask((dhps["cost"] == 0) | (dhps["cost"] > max_cost))
dhps["capacity"] = dhps["capacity"].mask(dhps["cost"] > max_capacity)
# dhps["flow_temp"] = dhps["flow_temp"].mask(dhps["flow_temp"] <= 0)


dhps = dhps.reset_index(drop=True)

# return dhps
dhps.head()


# %%
def load_epcs():
    """Loads relevant columns of EPC records.

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


epcs = load_epcs()
epcs.head()


# %%
# File: heat_pump_adoption_modelling/pipeline/MCS/mcs_epc_joining.py
"""Joining the MCS and EPC datasets."""

import pandas as pd
import numpy as np
import string
import re
import recordlinkage as rl
import time

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.MCS.load_mcs import (
    load_domestic_hps,
    load_epcs,
)


matching_parameter = 0.7
merged_path = "outputs/mcs_epc.csv"
# TODO: put this in config


# Overall process:
# - standardise address and postcode fields
# - extract numeric tokens from address
# - group by postcode
# - exact match on numeric tokens
# - compare address using jaro-winkler
# - drop anything below a certain parameter
# - of remaining matches, take the best
# - join using this matching

#### UTILS


def rm_punct(address):
    """Remove all unwanted punctuation from an address.
    Underscores are kept and slashes/dashes are converted
    to underscores so that the numeric tokens in e.g.
    "Flat 3/2" and "Flat 3-2" are treated as a whole later.

    Parameters
    ----------
    address : str
        Address to format.

    Return
    ----------
    address : str
        Formatted address."""
    if (address is pd.NA) | (address is np.nan):
        return ""
    else:
        # replace / and - with _
        address = address.replace("/", "_").replace("-", "_")
        # remove all punctuation other than _
        # address = address.translate(
        #   str.maketrans("", "", string.punctuation.replace("_", ""))
        #
        import re

        punct_regex = r"[\!\"#\$%&\\\'(\)\*\+,-\./:;<=>\?@\[\]\^`\{|\}~”“]"
        address = re.sub(punct_regex, "", address)
        return address


# %%
def extract_numeric_tokens(df):
    """Extract character strings containing numbers
    e.g. "45", "3a", "4_1".

    Parameters
    ----------
    df : pandas.Dataframe
        Dataframe with standardised_address field.

    Return
    ----------
    token_list : list
        List of sets of numeric tokens within each standardised_address."""

    token_list = [
        set(re.findall("\w*\d+\w*", address))
        for address in df[
            "standardised_address"
        ]  # "\w*\d\w*|\W[A-Z]\W|^[A-Z]\W|\W[A-Z]$"
    ]

    return token_list


# this isn't perfect - e.g. "Flat 3, 3 Street Name" would
# just become {3} - could use a multiset instead

# wonder if single-letter tokens should be in here too
# for e.g. "Flat A" or whether this would give too many
# false positives

# ---------------------------------------------------------------------------------

#### PREPROCESSING


def prepare_dhps(dhps):
    """Prepares Dataframe of domestic HP installations by adding
    standardised_postcode, standardised_address and numeric_tokens fields.

    Parameters
    ----------
    dhps : pandas.Dataframe
        Dataframe with postcode, address_1 and address_2 fields.

    Return
    ----------
    dhps : pandas.Dataframe
        Dataframe containing domestic HP records with added fields."""

    dhps["standardised_postcode"] = [
        pc.upper().strip() for pc in dhps["postcode"].fillna("unknown")
    ]

    dhps["standardised_address"] = [
        rm_punct(add1).lower().strip() + " " + rm_punct(add2).lower().strip()  # comment
        for add1, add2 in zip(dhps.address_1.fillna(""), dhps.address_2.fillna(""))
    ]

    dhps["numeric_tokens"] = extract_numeric_tokens(dhps)

    return dhps


# %%
def prepare_epcs(epcs):
    """Prepares Dataframe of EPC records by adding
    standardised_postcode, standardised_address and numeric_tokens fields.

    Parameters
    ----------
    epcs : pandas.Dataframe
        Dataframe with POSTCODE and ADDRESS1 fields.

    Return
    ----------
    epcs : pandas.Dataframe
        Dataframe containing EPC records with added fields."""

    # Remove spaces, uppercase and strip whitespace from
    # postcodes in order to exact match on this field
    epcs["standardised_postcode"] = [
        postcode.upper().replace(" ", "")
        for postcode in epcs["POSTCODE"].fillna("UNKNOWN")
    ]

    # Remove punctuation, lowercase and concatenate address fields
    # for approximate matching
    epcs["standardised_address"] = [
        rm_punct(address).lower().strip()
        for address in epcs["ADDRESS1"]  # only Address 1?
    ]

    epcs["numeric_tokens"] = extract_numeric_tokens(epcs)

    return epcs


# ---------------------------------------------------------------------------------


# %%
#### JOINING


def form_matching(df1, df2):
    """Forms a matching between two Dataframes.
    Initially an index is formed between records with shared
    standardised_postcode, then the records are compared for
    exact matches on numeric tokens and fuzzy matches on
    standardised_address (using Jaro-Winkler method).

    Parameters
    ----------
    df1 : pandas.Dataframe
        Dataframe with standardised_postcode, numeric_tokens
        and standardised_address fields.
    df2 : pandas.Dataframe
        Dataframe with standardised_postcode, numeric_tokens
        and standardised_address fields.

    Return
    ----------
    matching : pandas.Dataframe
        Dataframe giving indices of df1 and matched indices in df2
        along with similarity scores for numeric tokens (0/1) and address (0-1)."""

    # Index
    print("- Forming an index...")
    indexer = rl.Index()
    indexer.block(on="standardised_postcode")
    candidate_pairs = indexer.index(df1, df2)

    # Compare
    print("- Forming a comparison...")
    comp = rl.Compare()
    comp.exact("numeric_tokens", "numeric_tokens", label="numerics")
    # Jaro-Winkler chosen here as it prioritises characters
    # near the start of the string - this feels suitable as
    # the key information (such as house name) is likely to
    # be near the start of the string
    # This feels better suited than e.g. Levenshtein as to not
    # punish extra information at the end of the address field
    # e.g. town, county
    comp.string(
        "standardised_address",
        "standardised_address",
        method="jarowinkler",
        label="address_score",
    )

    # Classify
    print("- Computing a matching...")
    matching = comp.compute(candidate_pairs, df1, df2)

    return matching


# %%
def join_mcs_epc_data(dhps=None, epcs=None, save=True, drop_epc_address=False):
    """Joins MCS and EPC data.

    Parameters
    ----------
    dhps : pandas.Dataframe
        Dataframe with standardised_postcode, numeric_tokens
        and standardised_address fields.
        If None,  HP data is loaded and augmented.
    epcs : pandas.Dataframe
        Dataframe with standardised_postcode, numeric_tokens
        and standardised_address fields.
        If None, EPC data is loaded and augmented.
    save : bool
        Whether or not to save the output.
    drop_epc_address : bool
        Whether or not to drop addresses from the EPC records.
        Useful to keep for determining whether matches are sensible.

    Return
    ----------
    merged : pandas.Dataframe
        Dataframe containing merged MCS and EPC records."""

    if dhps is None:
        print("Preparing domestic HP data...")
        dhps = load_domestic_hps()
        dhps = prepare_dhps(dhps)

    if epcs is None:
        print("Preparing EPC data...")
        epcs = load_epcs()
        epcs = prepare_epcs(epcs)

    print("Forming a matching...")
    matching = form_matching(df1=dhps, df2=epcs)

    # First ensure that all matches are above the matching parameter
    good_matches = matching[
        (matching["numerics"] == 1) & (matching["address_score"] >= matching_parameter)
    ].reset_index()

    # Take the indices of the rows with best address match for each MCS index
    top_match_indices = good_matches.groupby("level_0")["address_score"].idxmax()

    # Filter the matches df to just the top matches
    # Drop duplicates in case of ties
    top_matches = (
        good_matches.loc[top_match_indices]
        .drop_duplicates("level_0")
        .drop(columns=["numerics", "address_score"])
    )

    print("Joining the data...")
    merged = (
        dhps.reset_index()
        .merge(top_matches, how="left", left_on="index", right_on="level_0")
        .merge(epcs.reset_index(), how="left", left_on="level_1", right_on="index")
        .drop(
            columns=[
                "index_x",
                "standardised_postcode_x",
                "standardised_address_x",
                "numeric_tokens_x",
                "ADDRESS1",
                "POSTTOWN",
                "POSTCODE",
                "index_y",
                "standardised_postcode_y",
                "numeric_tokens_y",
            ]
        )
    )

    if drop_epc_address:
        merged = merged.drop(columns="standardised_address_y")
    else:
        merged = merged.rename(columns={"standardised_address_y": "epc_address"})

    if save:
        merged.to_csv(PROJECT_DIR / merged_path)

    return merged


# %%
# ---------------------------------------------------------------------------------


def main():
    """Main function: Loads and joins MCS data to EPC data."""

    start_time = time.time()

    merged = join_mcs_epc_data()

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)

    print(
        "Loading and joining MCS and EPC data took {} minutes.\n\nOutput saved in {}".format(
            runtime, merged_path
        )
    )

    merged.head()


if __name__ == "__main__":
    # Execute only if run as a script
    main()


# %%
# merged = join_mcs_epc_data()
merged = join_mcs_epc_data()
merged.head()


# %%
merged.columns


# %%
address_df = merged[
    [
        "address_1",
        "address_2",
        "address_3",
        "epc_address",
        "postcode",
        "level_0",
        "level_1",
        "BUILT_FORM",
    ]
]
address_df.head(50)


# %%
print(address_df.shape)
address_df["BUILT_FORM"].value_counts(dropna=False)
# address_df.loc[address_df['BUILT_FORM'] == np.nan].shape


# %%
address_df.loc[address_df["address_1"].str.startswith("(MPAN")]


# %%
dhps = load_domestic_hps()
dhps = prepare_dhps(dhps)

dhps.head()


# %%
dhps.tail(50)


# %%
print(dhps["standardised_address"].value_counts)


# %%
epcs = load_epcs()
epcs = prepare_epcs(epcs)


# %%
epcs.head(50)


# %%
