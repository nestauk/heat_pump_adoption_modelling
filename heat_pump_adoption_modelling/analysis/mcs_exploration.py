# this should be turned into a jupyter notebook!
import pandas as pd
import numpy as np
import datetime as dt

from heat_pump_adoption_modelling.pipeline.load_mcs import load_domestic_hps
from heat_pump_adoption_modelling.pipeline.mcs_epc_joining import prepare_dhps, prepare_epcs, form_matching

dhps = load_domestic_hps()


#### Investigating cleanliness of data

# DUPLICATES

# Already dropped exact duplicates during data import, but still
# several properties with multiple records
dhps["concat_address"] = dhps["address_1"] + " " + dhps["postcode"]
dhps[dhps.concat_address.duplicated(keep=False)].sort_values('concat_address').head(20)

# Some have different products on different dates, but
# others seem to be errors

# Example where the records seem to be referring to the
# same installation but with slightly different info
dhps[dhps.concat_address=='1 DE127QZ']

# Example with duplicate records on different dates
dhps[dhps.concat_address=='4 YO434RY']

# Can/should we drop duplicate records from the same day
# or within some small time interval?
# Also records that are exactly the same other than certain fields (e.g. date)?
dupes = dhps[dhps.concat_address.duplicated(keep=False)]
time_diffs = dupes.groupby("concat_address").apply(
    lambda x: x.date.max() - x.date.min()
).sort_values()
time_diffs[time_diffs == dt.timedelta(0)]

# conclusion: wait until we have version number data


# COST

# extremely high cost
dhps.sort_values('cost', ascending=False).head(20)
dhps[dhps.cost > 100000].shape[0] # 94 records over £100,000

# zero cost
dhps[dhps.cost == 0].shape[0] # 2762 records

# zero heat supplied
dhps[dhps.heat_supplied == 0].shape[0] # 116 records

# zero cost and heat
dhps[(dhps.cost == 0) & (dhps.heat_supplied == 0)] # 16 records

# Some of the zero / very low cost installations seem to be
# a collection of installations in the same area -
# possibly new builds for which the cost is difficult
# to measure?
dhps[dhps.concat_address == "GFF LE20UZ"]

# conclusion: exclude 0 values


#### ADDRESS MATCHING

# What is a sensible parameter for address matching?

dhps = prepare_dhps()
epcs = prepare_epcs()
matching = form_matching(dhps, epcs)

adequate_matches = matching[(matching.numerics == 1) & (matching.address > 0)]

top_addresses = (
    adequate_matches
    .loc[
        adequate_matches
        .groupby(level=0)
        ['address']
        .idxmax()
    ]
    .drop(columns='numerics')
    .reset_index(level=1)
)

top_addresses[['dhps_address', 'epcs_address']] = np.nan

for i in top_addresses.index:
    j = top_addresses.loc[i, 'level_1']
    top_addresses.loc[i, ['dhps_address', 'epcs_address']] = [dhps.loc[i, 'standardised_address'], epcs.loc[j, 'standardised_address']]

top_addresses[top_addresses.address>0.7].sort_values('address').head(30)




# TODO: 
# - do houses with heat pumps match?
# - what proportion of EPCs not in MCS etc
# - how many heat pumps in each?