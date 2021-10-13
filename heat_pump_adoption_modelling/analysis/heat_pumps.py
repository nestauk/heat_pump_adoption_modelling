import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import string
import fuzzymatcher
import fuzzywuzzy


#### Loading in MCS data and filtering to domestic HPs

hps = (
    pd.read_excel("NESTA data files/Heat pump installations 2010 to 31082021.xlsx",
    dtype = {
        "Address Line 1": str,
        "Address Line 2": str,
        "Address Line 3": str,
        "Postcode": str
    })
    .rename(columns={
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
        "Overall Cost": "cost"
    })
    .convert_dtypes()
    .drop_duplicates()
)

# Make RHI field values easier to use
hps["rhi"] = (
    hps["rhi"]
    .replace({
        "RHI Installation ": True,
        "Not Domestic RHI installation ": False,
        "Unspecified": np.nan
    })
)

# Filter to domestic installations
dhps = (
    hps
    [hps['installation_type'] == 'Domestic']
    .drop(columns='installation_type')
    .reset_index(drop=True)
)

# 69262 domestic HP installations


#### Investigating cleanliness of data

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


#### Extracting insights

# median installation cost per year
dhps["year"] = dhps.date.dt.year

dhps_noexhaust = dhps[dhps.tech_type != "Exhaust Air Heat Pump"]

cost_data = (
    dhps_noexhaust
    .groupby(["year", "tech_type"])
    ["cost"]
    .median()
    .loc[2015:2021]
    .unstack()
)

fig, ax = plt.subplots()
ax.plot(
    cost_data.index,
    cost_data['Ground/Water Source Heat Pump'].tolist(),
    label = 'Ground/Water Source Heat Pump',
    c = 'blue'
)
ax.plot(
    cost_data.index,
    cost_data['Air Source Heat Pump'].tolist(),
    label = 'Air Source Heat Pump',
    c = 'orange'
)
ax.set_ylim(0, 21000)
ax.grid(axis='y', color='0.8')
ax.set_xlabel('Year')
ax.set_ylabel('Median cost of installation (£)')
ax.legend(title = "Heat pump type")
ax.set_title('Median cost of heat pump installations\n(years with >100 installations only)')

plt.show()


# numbers of installations each year
numbers_data = (
    dhps_noexhaust
    .groupby(["year", "tech_type"])
    ["cost"]
    .count()
    .unstack()
    .loc[2014:2021]
)

fig, ax = plt.subplots()
numbers_data.plot(
    kind="bar",
    ax=ax,
    zorder=3,
    color=['orange', 'blue']
)
ax.set_xlabel('Year')
ax.set_ylabel('Installations')
ax.grid(axis='y', color='0.8', zorder=0)
ax.legend(title = 'Heat pump type')
ax.set_title('Numbers of heat pumps installed')
plt.xticks(rotation=0)
plt.tight_layout()



#### Geographic distribution - local authorities

dhps.groupby('local_authority').agg(
        number = ('cost', lambda col: col.count()), 
        median_cost = ('cost', lambda col: col.median())
    ).sort_values('median_cost', ascending=False)

# some LAs with not many installations
# some non-standard local authority names -
# e.g. "Belfast City Council, Lisburn & Castlereagh..."


#### Joining to EPC data

# First experiment with a small area - Leicester

le_epc = pd.read_csv('leicester_epc.csv').drop(columns='Unnamed: 0')
le_hps = dhps[dhps.local_authority=='Leicester']

def rm_punct(address):
    if (address is pd.NA) | (address is np.nan):
        return ""
    else:
        return address.translate(str.maketrans('','',string.punctuation))

le_epc['concat_address'] = [
    rm_punct(add1).lower() + " " + pc.replace(" ", "")
    for add1, pc
    in zip(le_epc.ADDRESS1, le_epc.POSTCODE)
]

le_hps['concat_address'] = [
    rm_punct(add1).lower() + " " + rm_punct(add2).lower() + " " + pc.replace(" ", "")
    for add1, add2, pc
    in zip(le_hps.address_1, le_hps.address_2, le_hps.postcode)
]

test = fuzzymatcher.fuzzy_left_join(le_hps, le_epc, left_on='concat_address', right_on='concat_address')

# this works pretty well


# Now try with the full data

property_chars = pd.read_csv('property_characteristics.csv').drop(columns='Unnamed: 0')

property_chars['concat_address'] = [
    rm_punct(add1).lower() + " " + pc.replace(" ", "")
    for add1, pc
    in zip(property_chars.ADDRESS1, property_chars.POSTCODE)
]

# some fields are NA so we can't immediately use .replace
dhps[['address_1', 'address_2', 'postcode']] = dhps[['address_1', 'address_2', 'postcode']].fillna("")

dhps['concat_address'] = [
    rm_punct(add1).lower() + " " + rm_punct(add2).lower() + " " + pc.replace(" ", "")
    for add1, add2, pc
    in zip(dhps.address_1, dhps.address_2, dhps.postcode)
]

# This takes too long
merged = fuzzymatcher.fuzzy_left_join(dhps, property_chars, left_on='concat_address', right_on='concat_address')

# It will be more reliable (and probably more efficient)
# to first join on postcode, then fuzzy match the remaining address

# TODO: 
# - do houses with heat pumps match?
# - what proportion of EPCs not in MCS etc
# - how many heat pumps in each?