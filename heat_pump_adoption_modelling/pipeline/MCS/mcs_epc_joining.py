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
        address = address.translate(
            str.maketrans("", "", string.punctuation.replace("_", ""))
        )
        return address


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
        set(re.findall("\w*\d\w*", address)) for address in df["standardised_address"]
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
        rm_punct(add1).lower().strip() + " " + rm_punct(add2).lower().strip()
        for add1, add2 in zip(dhps.address_1.fillna(""), dhps.address_2.fillna(""))
    ]

    dhps["numeric_tokens"] = extract_numeric_tokens(dhps)

    return dhps


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
        postcode.replace(" ", "") for postcode in epcs["POSTCODE"].fillna("UNKNOWN")
    ]

    # Remove punctuation, lowercase and concatenate address fields
    # for approximate matching
    epcs["standardised_address"] = [
        rm_punct(address).lower().strip() for address in epcs["ADDRESS1"]
    ]

    epcs["numeric_tokens"] = extract_numeric_tokens(epcs)

    return epcs


# ---------------------------------------------------------------------------------


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


def join_mcs_epc_data(
    dhps=None, epcs=None, save=True, all_records=False, drop_epc_address=False
):
    """Joins MCS and EPC data.

    Parameters
    ----------
    dhps : pandas.Dataframe
        Dataframe with standardised_postcode, numeric_tokens
        and standardised_address fields.
        If None, domestic HP data is loaded and augmented.
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

    if all_records:

        def get_max_epc_indices(df):
            return df.loc[df.address_score == df.address_score.max(), "level_1"]

        top_matches = (
            good_matches.groupby("level_0")
            .apply(get_max_epc_indices)
            .droplevel(1)
            .reset_index()
            .rename(columns={"index": "level_0"})
        )

    else:
        # Take the indices of the rows with best address match for each MCS index
        top_match_indices = good_matches.groupby("level_0")["address_score"].idxmax()

        # Filter the matches df to just the top matches
        top_matches = good_matches.loc[top_match_indices].drop(
            columns=["numerics", "address_score"]
        )

    print("Joining the data...")
    merged = (
        dhps.reset_index()
        .merge(top_matches, how="left", left_on="index", right_on="level_0")
        .merge(epcs.reset_index(), how="left", left_on="level_1", right_on="index")
        .drop(
            columns=[
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
        merged = merged.rename({"standardised_address_y": "epc_address"})

    if save:
        merged.to_csv(PROJECT_DIR / merged_path)

    return merged


# ---------------------------------------------------------------------------------


def main():
    """Main function: Loads and joins MCS data to EPC data."""

    start_time = time.time()

    join_mcs_epc_data()

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)

    print(
        "Loading and joining MCS and EPC data took {} minutes.\n\nOutput saved in ".format(
            runtime
        )
        + merged_path
    )


if __name__ == "__main__":
    # Execute only if run as a script
    main()
