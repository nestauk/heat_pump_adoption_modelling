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
#     display_name: heat_pump_adoption_modelling
#     language: python
#     name: heat_pump_adoption_modelling
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path


from heat_pump_adoption_modelling.getters import epc_data, deprivation_data
from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.encoding import (
    feature_encoding,
    category_reduction,
)
from heat_pump_adoption_modelling.pipeline.supervised_model import (
    data_aggregation,
    data_preprocessing,
    hp_growth_prediction,
    hp_status_prediction,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

from heat_pump_adoption_modelling.pipeline.supervised_model.utils import (
    error_analysis,
    plotting_utils,
)
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import kepler

import pandas as pd
from keplergl import KeplerGl
import matplotlib as mpl
import numpy as np

mpl.rcParams.update(mpl.rcParamsDefault)

from ipywidgets import interact

# %%
EPC_PREPROC_FEAT_SELECTION = [
    "BUILDING_REFERENCE_NUMBER",
    "ADDRESS1",
    "ADDRESS2",
    "POSTTOWN",
    "POSTCODE",
    "INSPECTION_DATE",
    "CURRENT_ENERGY_RATING",
    "MAINHEAT_DESCRIPTION",
    "LOCAL_AUTHORITY_LABEL",
    "TENURE",
    "TRANSACTION_TYPE",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "ENTRY_YEAR",
    "ENTRY_YEAR_INT",
    "INSPECTION_DATE",
    "INSPECTION_DATE_AS_NUM",
    "UNIQUE_ADDRESS",
    "BUILDING_ID",
    "N_ENTRIES",
    "N_ENTRIES_BUILD_ID",
    "HEATING_SYSTEM",
    "HP_INSTALLED",
    "HP_TYPE",
    "ENERGY_RATING_CAT",
    "CONSTRUCTION_AGE_BAND",
    "CONSTRUCTION_AGE_BAND_ORIGINAL",
    "SECONDHEAT_DESCRIPTION",
]

# %%
epc_df = pd.read_csv(
    str(PROJECT_DIR)
    + "/outputs/EPC_data/preprocessed_data/Q2_2021/EPC_GB_preprocessed.csv",
    usecols=EPC_PREPROC_FEAT_SELECTION,
)

# %%
# epc_df = data_preprocessing.load_epc_samples(
#    subset="complete", usecols=EPC_PREPROC_FEAT_SELECTION, preload=False
# )

# %%
epc_df.shape

# %%
epc_df["MAINHEAT_DESCRIPTION"] = epc_df["MAINHEAT_DESCRIPTION"].str.lower()
epc_df["SECONDHEAT_DESCRIPTION"] = epc_df["SECONDHEAT_DESCRIPTION"].str.lower()

# %%
epc_df["MAINHEAT_DESCRIPTION"] = epc_df["MAINHEAT_DESCRIPTION"].str.lower()
epc_df["SECONDHEAT_DESCRIPTION"] = epc_df["SECONDHEAT_DESCRIPTION"].str.lower()

epc_df["MAINHEAT_DESCRIPTION"] = epc_df["MAINHEAT_DESCRIPTION"].str.lower()
epc_df["SECONDHEAT_DESCRIPTION"] = epc_df["SECONDHEAT_DESCRIPTION"].str.lower()

print(
    epc_df.loc[epc_df["SECONDHEAT_DESCRIPTION"].str.contains("heat pump")][
        "SECONDHEAT_DESCRIPTION"
    ].shape
)
print(
    epc_df.loc[epc_df["SECONDHEAT_DESCRIPTION"].str.contains("pumpa teas")][
        "SECONDHEAT_DESCRIPTION"
    ].shape
)
print(
    epc_df.loc[epc_df["SECONDHEAT_DESCRIPTION"].str.contains("pwmp gwres")][
        "SECONDHEAT_DESCRIPTION"
    ].shape
)

# %%
epc_df.loc[epc_df["MAINHEAT_DESCRIPTION"].str.contains("pwmp gwres")][
    "MAINHEAT_DESCRIPTION"
].unique()

# %%
print(
    epc_df.loc[epc_df["MAINHEAT_DESCRIPTION"].str.contains("pumpa teas")][
        "MAINHEAT_DESCRIPTION"
    ].shape
)
print(
    epc_df.loc[epc_df["MAINHEAT_DESCRIPTION"].str.contains("pwmp gwres")][
        "MAINHEAT_DESCRIPTION"
    ].shape
)
print(
    epc_df.loc[epc_df["MAINHEAT_DESCRIPTION"].str.contains("heat pump")][
        "MAINHEAT_DESCRIPTION"
    ].shape
)

# %%
epc_df["HP_INSTALLED"].value_counts()

# %%
epc_df["HP_INSTALLED"] = np.where(
    (epc_df["HP_INSTALLED"])
    | (epc_df["MAINHEAT_DESCRIPTION"].str.contains("pumpa teas"))
    | (epc_df["MAINHEAT_DESCRIPTION"].str.contains("pwmp gwres")),
    True,
    False,
)

epc_df["HP_INSTALLED"].value_counts()

# %%
epc_df["HP_INSTALLED_2nd"] = np.where(
    epc_df["SECONDHEAT_DESCRIPTION"].str.contains("heat pump"), True, False
)
epc_df["HP_INSTALLED_2nd"].value_counts(dropna=False)

# %%
epc_df["HP_INSTALLED"].value_counts(dropna=False)

# %%
print(epc_df.loc[epc_df["HP_INSTALLED"] & (epc_df["HP_INSTALLED_2nd"])].shape)

# %%
epc_df["HP_INSTALLED"] = np.where(
    (epc_df["HP_INSTALLED"]) | (epc_df["HP_INSTALLED_2nd"]), True, False
)

epc_df["HP_INSTALLED"].value_counts(dropna=False)

# %%
prep_epc_df = data_preprocessing.preprocess_data(
    epc_df, encode_features=False, subset="complete"
)

# %%
print(list(prep_epc_df.columns))

# %%
no_epc_entry_after_mcs = prep_epc_df.loc[prep_epc_df["No EPC HP entry after MCS"]]

# %%
no_epc_entry_after_mcs.columns

# %%
no_epc_entry_after_mcs.shape

# %%
no_epc_entry_after_mcs[
    [
        "IMD Rank",
        "ADDRESS1",
        "ADDRESS2",
        "POSTCODE",
        "MCS address",
        "MAINHEAT_DESCRIPTION",
        "SECONDHEAT_DESCRIPTION",
        "TRANSACTION_TYPE",
        "INSPECTION_DATE",
        "HP_INSTALL_DATE",
    ]
].tail(50)

# %%
no_epc_entry_after_mcs["MAINHEAT_DESCRIPTION"].value_counts()

# %%
epc_entry_before_mcs = prep_epc_df.loc[prep_epc_df["EPC HP entry before MCS"]]

# %%
epc_entry_before_mcs[
    [
        "IMD Rank",
        "ADDRESS1",
        "ADDRESS2",
        "POSTCODE",
        "MCS address",
        "MAINHEAT_DESCRIPTION",
        "SECONDHEAT_DESCRIPTION",
        "TRANSACTION_TYPE",
        "INSPECTION_DATE",
        "HP_INSTALL_DATE",
    ]
].head(50)

# %%
epc_df.loc[~epc_df["MCS_AVAILABLE"] == True].shape

# %%
epc_df.loc[epc_df["MCS_AVAILABLE"] == False].shape

# %%
mcs_df = pd.read_csv(str(PROJECT_DIR) + "/inputs/MCS_data/mcs_epc.csv")

# %%
mcs_df.shape

# %%
mcs_df.loc[~mcs_df["compressed_epc_address"].isna()].shape

# %%
mcs_df.loc[mcs_df["compressed_epc_address"].isna()].shape

# %%
100 / 96271 * 76934

# %%
