# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
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
]

# %%
epc_df = pd.read_csv(
    str(PROJECT_DIR)
    + "/outputs/EPC_data/preprocessed_data/Q2_2021/EPC_GB_preprocessed.csv",
    usecols=EPC_PREPROC_FEAT_SELECTION,
)

# %%
epc_df.shape

# %%
epc_df = data_preprocessing.epc_sample_loading(
    subset="complete", usecols=EPC_PREPROC_FEAT_SELECTION, preload=False
)

# %%
epc_df.shape

# %%
epc_df = data_preprocessing.data_preprocessing(
    epc_df, encode_features=False, subset="complete"
)

# %%
print(list(epc_df.columns))

# %%
epc_df.loc[~epc_df["MCS_AVAILABLE"] == True].shape

# %%
epc_df.loc[epc_df["MCS_AVAILABLE"] == False].shape
