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
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: heat_pump_adoption_modelling
#     language: python
#     name: heat_pump_adoption_modelling
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from heat_pump_adoption_modelling.getters import epc_data
from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.encoding import (
    feature_encoding,
    category_reduction,
)
from heat_pump_adoption_modelling.pipeline.supervised_model import utils
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
epc_df = epc_data.load_preprocessed_epc_data(
    version=version, nrows=500000, usecols=None
)


epc_df.head()

# %%
ordinal_features = [
    "MAINHEAT_ENERGY_EFF",
    "CURRENT_ENERGY_RATING",
    "POTENTIAL_ENERGY_RATING",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
    "GLAZED_TYPE",
    "MAINHEATC_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "MAINS_GAS_FLAG",
    "CONSTRUCTION_AGE_BAND_ORIGINAL",
    "CONSTRUCTION_AGE_BAND",
    "N_ENTRIES",
    "N_ENTRIES_BUILD_ID",
    "ENERGY_RATING_CAT",
]

drop_features = [
    "BUILDING_REFERENCE_NUMBER",
    "ADDRESS1",
    "POSTTOWN",
    "LODGEMENT_DATE",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "MAINHEAT_DESCRIPTION",
    "SHEATING_ENERGY_EFF",
    "HEATING_COST_POTENTIAL",
    "HOT_WATER_COST_POTENTIAL",
    "LIGHTING_COST_POTENTIAL",
    "CONSTRUCTION_AGE_BAND",
    "NUMBER_HEATED_ROOMS",
    "LOCAL_AUTHORITY_LABEL",
    "ENTRY_YEAR",
    "N_ENTRIES",
    "CURR_ENERGY_RATING_NUM",
    "ENERGY_RATING_CAT",
    "UNIQUE_ADDRESS",
    "INSPECTION_DATE",
    "MAIN_FUEL",
    "HEATING_SYSTEM",
    "HP_TYPE",
]

# %%
epc_df_encoded = feature_encoding.feature_encoding_pipeline(
    epc_df,
    ordinal_features,
    reduce_categories=True,
    onehot_features="auto",
    target_variables=["HP_INSTALLED", "POSTCODE"],
    drop_features=drop_features,
)

# %%
epc_df_encoded.head()


# %%
def get_year_range_data(df, years):
    year_range_df = pd.concat(
        [
            feature_engineering.filter_by_year(
                df,
                "BUILDING_ID",
                year,
                selection="latest entry",
                up_to=False,
            )
            for year in years
        ],
        axis=0,
    )

    return year_range_df


training_years = get_year_range_data(epc_df_encoded, [2008, 2009, 2010])
prediction_years = get_year_range_data(epc_df_encoded, [2011])

# %%
training_years["POSTCODE"].value_counts()

# %%
for postcode in training_years["POSTCODE"].unique():
    print(postcode)
    postcode_df = training_years.loc[training_years["POSTCODE"] == postcode]
    print(postcode_df["POSTCODE"].value_counts())

# %%
training_years.groupby("POSTCODE").head()

# %%

# %%
