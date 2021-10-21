import pandas as pd
import numpy as np
import string
import re
import recordlinkage as rl
import time

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.load_mcs import load_domestic_hps

epc_address_fields = ['ADDRESS1', 'POSTTOWN', 'POSTCODE']
epc_characteristic_fields = ['TOTAL_FLOOR_AREA', 'CONSTRUCTION_AGE_BAND', 'BUILT_FORM', 'PROPERTY_TYPE', 'HP_INSTALLED']
matching_parameter = 0.7
# TODO: put this in config


#### UTILS

def rm_punct(address):
    if (address is pd.NA) | (address is np.nan):
        return ""
    else:
        # replace / and - with _
        address = address.replace("/", "_").replace("-", "_")
        # remove all punctuation other than _
        address = address.translate(
            str.maketrans('', '', string.punctuation.replace("_", ""))
        )
        
        return address

# \w below interprets _ as part of a word
# /s and -s are important to keep in things like "Flat 3/2" and "Flat 3-2"
# not sure if this breaks address similarity when it appears in words though

def extract_numeric_tokens(df):
    # Extracts character strings containing numbers
    # as these are likely to be house or building numbers
    token_list = [
        set(re.findall('\w*\d\w*', address))
        for address
        in df['standardised_address']
    ]
    return token_list

# this isn't perfect - e.g. "Flat 3, 3 Street Name" would
# just become {3} - could use a multiset

# wonder if single-letter tokens should be in here too
# for e.g. "Flat A" or whether this would give too many
# false positives

# ---------------------------------------------------------------------------------

#### PREPROCESSING

def prepare_dhps():
    dhps = load_domestic_hps()
    
    dhps['standardised_postcode'] = [
        pc.upper().strip()
        for pc
        in dhps['postcode'].fillna("unknown")
    ]
    
    dhps['standardised_address'] = [
        rm_punct(add1).lower() + " " + rm_punct(add2).lower()
        for add1, add2
        in zip(dhps.address_1.fillna(""), dhps.address_2.fillna(""))
    ]
    
    dhps['numeric_tokens'] = extract_numeric_tokens(dhps)
    
    return dhps


def prepare_epcs():
    epcs = (
        pd.read_csv(
            PROJECT_DIR / 'outputs/EPC_data/preprocessed_data/Q2_2021/EPC_GB_preprocessed_and_deduplicated.csv',
            usecols = epc_address_fields + epc_characteristic_fields
        )
    )
    
    # Remove spaces, uppercase and strip whitespace from
    # postcodes in order to exact match on this field
    epcs['standardised_postcode'] = [
        postcode.replace(" ", "")
        for postcode
        in epcs['POSTCODE'].fillna('UNKNOWN')
    ]
    
    # Remove punctuation, lowercase and concatenate address fields
    # for approximate matching
    epcs['standardised_address'] = [
        rm_punct(address).lower()
        for address
        in epcs['ADDRESS1']
    ]
    
    epcs['numeric_tokens'] = extract_numeric_tokens(epcs)
    
    return epcs


# ---------------------------------------------------------------------------------


#### JOINING

def form_matching(df1, df2):
    
    # Index
    print('- Forming an index...')
    indexer = rl.Index()
    indexer.block(on='standardised_postcode')
    candidate_pairs = indexer.index(df1, df2)
    
    # Compare
    print('- Forming a comparison...')
    comp = rl.Compare()
    comp.exact(
        'numeric_tokens',
        'numeric_tokens',
        label='numerics'
    )
    comp.string(
        'standardised_address',
        'standardised_address',
        method='jarowinkler',
        label='address'
    )
    
    # Classify
    print('- Computing a matching...')
    matching = comp.compute(candidate_pairs, df1, df2)
    
    return matching


def join_mcs_epc_data(dhps=None, epcs=None, save=True):
    
    print('Preparing domestic HP data...')
    if dhps is None:
        dhps = prepare_dhps()
    
    print('Preparing EPC data...')
    if epcs is None:
        epcs = prepare_epcs()
    
    print('Forming a matching...')
    matching = form_matching(df1=dhps, df2=epcs)
    
    good_matches = matching[
        (matching.numerics == 1) & (matching.address >= matching_parameter)
    ]
    
    def epc_lookup(i):
        try:
            matches = good_matches.loc[i]
            index = matches.idxmax()['address']
            return epcs.loc[index][
                epc_characteristic_fields + ['standardised_address']
            ]
        except KeyError:
            return np.nan
    
    dhps[epc_characteristic_fields + ['standardised_address']] = np.nan
    
    print('Joining the data...')
    # this bit is really slow - is there a quicker method?
    for i in dhps.index:
        dhps.loc[i, epc_characteristic_fields + ['standardised_address']] = epc_lookup(i)
    
    merged = dhps.drop(columns=['standardised_postcode', 'numeric_tokens'])
    
    if save:
        merged.to_csv(PROJECT_DIR / 'outputs/mcs_epc.csv')
    
    return merged


# ---------------------------------------------------------------------------------


def main():
    """Main function: Loads and joins MCS data to EPC data."""

    start_time = time.time()

    join_mcs_epc_data()

    end_time = time.time()
    runtime = round((end_time - start_time) / 60)

    print("\nLoading and joining MCS and EPC data took {} minutes.".format(runtime))


if __name__ == "__main__":
    # Execute only if run as a script
    main()



