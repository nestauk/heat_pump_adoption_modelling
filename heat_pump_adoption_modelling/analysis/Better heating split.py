# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     formats: ipynb,py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: heat_pump_adoption_modelling
#     language: python
#     name: heat_pump_adoption_modelling
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from heat_pump_adoption_modelling.getters import epc_data, deprivation_data
from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.encoding import (
    feature_encoding,
    category_reduction,
)
from heat_pump_adoption_modelling.pipeline.supervised_model import (
    utils,
    data_aggregation,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

import sklearn
from sklearn import svm

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

from ipywidgets import interact

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

# %%
FIGPATH = str(PROJECT_DIR) + "/outputs/figures/"

# Load preprocessed and deduplicated data
version = "preprocessed_dedupl"
version = "preprocessed"  # _dedupl"

# Load all available columns
epc_df = epc_data.load_raw_epc_data(
    # version=version,
    # nrows=500000,
    usecols=["MAINHEAT_DESCRIPTION"]
)


epc_df.head()

# %%
list(epc_df.columns)


# %%
def get_heating_system(heating):

    if "," in heating:
        parts = heating.split(",")
        system, fuel = parts[0], parts[-1]

    elif "with" in heating or "utilising" in heating:
        # print(first_heating)
        parts = re.findall(r"(.+)(with|utilising)(.+)", heating)[0]
        system, fuel = parts[0], parts[-1]

    else:
        system = heating
        fuel = "unknown"

    system = system.strip().lower()
    fuel = fuel.strip().lower()

    system = "unknown" if system == "" else system
    fuel = "unknown" if fuel == "" else fuel
    # print(system)
    # print(fuel)
    return system, fuel


def get_first_and_second_heating_system(row):

    heating = row["MAINHEAT_DESCRIPTION"]

    if isinstance(heating, float):
        return ("unknown", "unknown"), ("unknown", "unknown")

    heating = heating.replace(
        "????????????????????????????????????????????????????", ","
    )

    if "|" in heating:
        parts = heating.split("|")
        first_heating, second_heating = parts[0], parts[1]

    else:
        first_heating = heating
        second_heating = ""

    first_system, first_fuel = get_heating_system(first_heating)
    sec_system, sec_fuel = get_heating_system(second_heating)

    return (first_system, first_fuel, sec_system, sec_fuel)


# %%
# sap = (epc_df['SYSTEM'].str.contains('main-heating', na=False))
# epc_df[sap].head()

# %%
for x in zip(
    *epc_df["MAINHEAT_DESCRIPTION"].apply(get_first_and_second_heating_system)
):
    print(x)

# %%
# try:
(
    epc_df["FIRST_HEATING_SYSTEM"],
    epc_df["FIRST_HEATING_FUEL"],
    epc_df["SECOND_HEATING_SYSTEM"],
    epc_df["SECOND_HEATING_FUEL"],
) = epc_df.apply(get_first_and_second_heating_system)
# epc_df['FIRST_HEATING_SYSTEM'], epc_df['FIRST_HEATING_FUEL'], epc_df['SECOND_HEATING_SYSTEM'], epc_df['SECOND_HEATING_FUEL'] =  zip(*epc_df['MAINHEAT_DESCRIPTION'].apply(get_first_and_second_heating_system))

# %%
# try:
epc_df[
    [
        "FIRST_HEATING_SYSTEM",
        "FIRST_HEATING_FUEL",
        "SECOND_HEATING_SYSTEM",
        "SECOND_HEATING_FUEL",
    ]
] = epc_df["MAINHEAT_DESCRIPTION"].apply(
    get_first_and_second_heating_system, axis=1, result_type="expand"
)
# epc_df['FIRST_HEATING_SYSTEM'], epc_df['FIRST_HEATING_FUEL'], epc_df['SECOND_HEATING_SYSTEM'], epc_df['SECOND_HEATING_FUEL'] =  zip(*epc_df['MAINHEAT_DESCRIPTION'].apply(get_first_and_second_heating_system))

# %%
epc_df["FIRST_HEATING_SYSTEM"] = epc_df["FIRST_HEATING_SYSTEM"].apply(
    heating_system_map
)
epc_df["FIRST_HEATING_FUEL"] = epc_df["FIRST_HEATING_FUEL"].apply(heating_fuel_map)

epc_df["SECOND_HEATING_SYSTEM"] = epc_df["SECOND_HEATING_SYSTEM"].apply(
    heating_system_map
)
epc_df["SECOND_HEATING_FUEL"] = epc_df["SECOND_HEATING_FUEL"].apply(heating_fuel_map)

# %%
heating_system_map = {
    "boiler and radiators": "boiler and radiators",
    "boiler": "boiler and radiators",
    "boiler and": "boiler and radiators",
    "solid-fuel boiler": "boiler and radiators",
    "boiler & underfloor": "boiler and radiators",
    "boiler &amp; underfloor": "boiler and radiators",
    "bwyler a rheiddiaduron": "boiler and radiators",
    "radiator heating": "boiler and radiators",
    "electric storage": "electric storage heaters",
    "room heaters": "room heaters",
    "wresogyddion ystafell": "room heaters",
    "no system present: electric heaters assumed": "no system present: electric heaters assumed",
    "boiler and underfloor heating": "boiler and underfloor heating",
    "electric storage heaters": "electric storage heaters",
    "no system present": "electric heaters assumed",
    "portable electric heaters": "eletric heaters",
    "gas/lpg boiler pre-1998 with balanced or open-flue": "boiler and radiators",
    "electric boiler": "electric boiler",
    "stôr wresogyddion trydan": "electric storage heaters",
    "boiler and fan coil units": "unknown",
    "st?r wresogyddion trydan": "room heaters",
    "air source heat pump": "air source heat pump",
    "air source heat pump, systems": "air source heat pump",
    "air source heat pump fan coil units": "air source heat pump",
    "community scheme": "community scheme",
    "community": "community scheme",
    "community scheme utilising waste heat": "community",
    "community": "community",
    "water source heat pump": "water source heat pump",
    "cynllun cymunedol": "community",
    "hot-water-only systems": "hot-water-only systems",
    "air source heat pump fan coil units": "air source heat pump",
    "air sourceheat pump": "air source heat pump",
    "warm air": "warm air",
    "electric heat pumps": "heat pump",
    "community heat pump": "community heat pump",
    "ground sourceheat pump": "ground source heat pump",
    "mixed exhaust air source heat pump": "air source heat pump",
    "warm air heat pump": "warm air heat pump",
    "heat pump": "heat pump",
    "solar assisted source heat pump": "solar assisted source heat pump",
    "solar-assisted heat pump": "solar assisted source heat pump",
    "solar assisted heat pump": "solar assisted source heat pump",
    "exhaust source heat pump": "heat pump",
    "water source heat pump": "water source heat pump",
    "ground source heat pump": "ground source heat pump",
    "exhaust air mev source heat pump, , systems": "air source heat pump",
    "heat pumptrydan": "heat pump",
    "fully double glazed": "unknown",
    "sap05:main-heating": "unknown",
    "pwmp gwres sy’n tarddu yn y ddaear": "ground source heat pump",
    "electric underfloor heating": "electric underfloor heating",
    "electric underfloor heating (standard tariff)": "electric underfloor heating",
    "electric ceiling heating": "electric ceiling heating",
    "portable electric heaters assumed for most rooms": "no system present: electric heaters assumed",
}


# %%
pd.options.display.max_rows = 1000

epc_df["SYSTEM"].value_counts(dropna=False)

# %%
heating_fuel_map = {
    "mains gas": "mains gas",
    "nwy prif gyflenwad": "mains gas",
    "gas": "mains gas",
    "bottled gas": "bottled gas",
    "heat from boilers - gas": "mains gas",
    "balanced or open-flue, mains gas": "mains gas",
    "heat from boilers - mains gas": "mains gas",
    "mains gas and mains gas": "mains gas",
    "unknown": "unknown",
    "radiators": "unknown",
    "community": "unknown",
    "underfloor": "unknown",
    "auxiliary heater (electric) source heat pump, radiators, electric": "electric",
    "electric": "electric",
    "electricaire": "electric",
    "radiators, electric": "electric",
    "eletric": "electric",
    "electric storage heaters": "electric",
    "heat pump": "electric",
    "electricity": "electric",
    "electricity (24-hr heating tariff)": "electric",
    "trydan": "electric",
    "electric ceiling heating": "electric",
    "electric underfloor heating": "electric",
    "oil": "oil",
    "olew": "oil",
    "lpg": "LPG",
    "bottled lpg": "LPG",
    "bulk lpg": "LPG",
    "lpg (bottleed)": "LPG",
    "lpg subject to special condition 18": "LPG",
    "dual fuel (mineral and wood)": "dual fuel (mineral and wood)",
    "dual fuel appliance": "dual fuel (mineral and wood)",
    "dual fuel appliance (mineral and wood)": "dual fuel (mineral and wood)",
    "coal": "coal",
    "glo": "coal",
    "house coal": "coal",
    "wood logs": "wood logs",
    "logiau coed": "wood logs",
    "smokeless fuel": "smokeless fuel",
    "wood chips": "wood chips",
    "main wood pellets": "wood pellets",
    "wood pellets": "wood pellets",
    "bulk wood pellets": "wood pellets",
    "wood chips": "wood chips",
    "biomass": "biomass",
    "geothermal heat": "geothermal heat",
    "bioethanol": "bio fuel",
    "b30k": "bio fuel",
    "liquid biofuel": "bio fuel",
    "solid fuel": "solid fuel",
    "anthracite": "anthracite",
    "waste heat": "waste heat",
    "chp": "Combined Heat and Power (CHP)",
    "chp, mains gas": "CHP / gas",
    "biomass and mains gas": "biomass / mains gas",
    "mains gas and biomass": "biomass / mains gas",
    "biomass and oil": "biomass / oil",
    "chp, biomass": "CHP / biomass",
    "chp, waste combustion and waste combustion": "CHP / waste",
    "chp, wood chips and mains gas": "CHP / wood chips / mains gas",
    "chp, radiators, mains gas": "CHP / mains gas",
    "chp and electric": "CHP / electric",
    "chp and oil": "CHP / oil",
    "geothermal heat and biomass": "geothermal heat / biomass",
    "chp and geothermal": "CHP / geothermal",
    "wood pellets and mains gas": "mains gas / wood pellets",
    "mains gas and wood pellets": "mains gas / wood pellets",
    "waste heat, mains gas": "waste heat / mains gas",
    "wood chips and mains gas": "wood chips / mains gas",
    "oil and mains gas": "oil / mains gas",
    "chp, mains gas and oil": "CHP / mains gas / oil",
}

# %%
epc_df["FUEL"].value_counts()

# %%
epc_df["CLEAN_SYSTEM"] = epc_df["SYSTEM"].map(heating_map)
epc_df["CLEAN_FUEL"] = epc_df["FUEL"].map(fuel_map)

# %%
epc_df.head(100)

# %%
epc_df["CLEAN_SYSTEM"].value_counts(dropna=False)

# %%
pd.options.display.max_rows = 1000

epc_df["FUEL"].value_counts()

# %%
epc_df.loc[epc_df["CLEAN_SYSTEM"] == "boiler and radiators"].head()

# %%
list(epc_df["MAINHEAT_DESCRIPTION"].unique())

# %%
