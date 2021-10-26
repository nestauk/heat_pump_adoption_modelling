# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import datetime as dt

from heat_pump_adoption_modelling.pipeline.load_mcs import load_domestic_hps
from heat_pump_adoption_modelling.pipeline.mcs_epc_joining import (
    prepare_dhps,
    prepare_epcs,
    form_matching,
)

# %%
dhps = load_domestic_hps()

# %% [markdown]
# ## Duplicates
#
# Already dropped exact duplicates during data import, but still several properties with multiple records:

# %%
dhps["concat_address"] = dhps["address_1"] + " " + dhps["postcode"]

dhps[dhps.concat_address.duplicated(keep=False)].sort_values("concat_address").head(20)

# %% [markdown]
# Some have different products on different dates, but others seem to be errors. Example where the records seem to be referring to the same installation but with slightly different info:

# %%
dhps[dhps.concat_address == "1 DE127QZ"]

# %% [markdown]
# Example with duplicate records on different dates:

# %%
dhps[dhps.concat_address == "4 YO434RY"]

# %% [markdown]
# Record version number is collected by MCS and they are sending it over, so holding off on this for now as we can just keep the latest version of each record and see what impact this has.

# %% [markdown]
# ## Cost
#
# Some records have extremely high cost (94 records over £100,000):

# %%
dhps.sort_values("cost", ascending=False).head(20)

dhps[dhps.cost > 100000].shape[0]

# %% [markdown]
# 2762 records have zero cost:

# %%
dhps[dhps.cost == 0].shape[0]

# %% [markdown]
# Some of the zero / very low cost installations seem to be collections of installations in the same area - possibly new builds for which the cost is difficult to measure? e.g.

# %%
dhps[dhps.concat_address == "GFF LE20UZ"]

# %% [markdown]
# Ofgem made similar considerations when forming [RHI monthly deployment data](https://www.gov.uk/government/statistics/rhi-monthly-deployment-data-august-2021):
#
# <em>The number of applications contributing to the averages after data cleaning processes were applied, after installations with a capacity or total cost of zero had been removed, as well as installations with a capacity of over 45kW. Previously, we also removed the top and bottom 5% by cost/kW however subsequent research has suggested that this may be cutting out valid costs.</em>

# %%
dhps.sort_values("heat_supplied", ascending=False).head(30)

# %% [markdown]
# # Address matching
#
# ## Identifying a sensible Jaro-Winkler threshold

# %%
dhps = prepare_dhps()
epcs = prepare_epcs()
matching = form_matching(dhps, epcs)

# %%
# filter to matches with the same postcode and at least some address similarity
adequate_matches = matching[(matching.numerics == 1) & (matching.address > 0)]

top_addresses = (
    adequate_matches.loc[adequate_matches.groupby(level=0)["address"].idxmax()]
    .drop(columns="numerics")
    .reset_index(level=1)
)

top_addresses[["dhps_address", "epcs_address"]] = np.nan

for i in top_addresses.index:
    j = top_addresses.loc[i, "level_1"]
    top_addresses.loc[i, ["dhps_address", "epcs_address"]] = [
        dhps.loc[i, "standardised_address"],
        epcs.loc[j, "standardised_address"],
    ]

# %% [markdown]
# Matches with distance 0.6 don't match very closely:

# %%
top_addresses[top_addresses.address > 0.6].sort_values("address").head(30)

# %% [markdown]
# 0.7 matches are a lot better:

# %%
top_addresses[top_addresses.address > 0.7].sort_values("address").head(30)

# %%
import matplotlib.pyplot as plt

# %%
dhps.heat_demand.plot(kind="hist")

# %%
dhps.heat_demand.sort_values(ascending=False).head(20)

# %%
dhps[dhps.heat_demand < 2000000].heat_demand.plot(kind="hist")

# %%
dhps[dhps.heat_demand < 50000].heat_demand.sort_values(ascending=False).head(20)
