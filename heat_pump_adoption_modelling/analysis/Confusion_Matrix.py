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

# %% [markdown]
# ## Imports and Data Loading

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
    version=version, nrows=5000000, usecols=None
)


epc_df.head()


# %% [markdown]
# ## Categorical Feature Encoding

# %%
epc_df = category_reduction.reduce_number_of_categories(epc_df)


# %%
@interact(feature=epc_df.columns)
def value_counts(feature):
    print(feature)
    print(epc_df[feature].value_counts(dropna=False))

    print(epc_df[feature].unique())
    print(epc_df[feature].max())
    print(epc_df[feature].min())


# %% [markdown]
# ### Numerical features

# %%
encoded_features = epc_df.copy()
# Get all only numericfeatures
num_features = encoded_features.select_dtypes(include=np.number).columns.tolist()
print("Number of numeric features: {}".format(len(num_features)))
print(num_features)

# %% [markdown]
# ### Encode ordinal cat features

# %%
encoded_features = epc_df.copy()

ordinal_cat_features = [
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

encoded_features = feature_encoding.ordinal_encode_cat_features(
    encoded_features, ordinal_cat_features
)

print("Number of ordinal features: {}".format(len(ordinal_cat_features)))

encoded_features[ordinal_cat_features].head()

# %% [markdown]
# ### Drop Unnecessary Features

# %%
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
    "POSTCODE",
    "INSPECTION_DATE",
    "MAIN_FUEL",
    "HEATING_SYSTEM",
    "HP_TYPE",
]

encoded_features = encoded_features.drop(columns=drop_features)

print("Number of features to be dropped: {}".format(len(drop_features)))

# %% [markdown]
# ### One Hot Encode Categorical Features

# %%
other_cat_features = [
    feature
    for feature in encoded_features.columns
    if (feature not in ordinal_cat_features) and (feature not in num_features)
]
one_hot_features = [f for f in other_cat_features if f != "HP_INSTALLED"]

print(
    "Number of categorical features (one-hot encoding): {}".format(
        len(one_hot_features)
    )
)
print(one_hot_features)
print()

encoded_features = feature_encoding.one_hot_encoding(encoded_features, one_hot_features)

# %% [markdown]
# ## Correlation Matrix

# %%
# Pearson Correlation Matrix
plt.figure(figsize=(25, 15))
cor = encoded_features.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.bwr)
plt.title("Correlation Matrix (complete)")
plt.tight_layout()
plt.savefig(FIGPATH + "correlation_matrix_complete.png", dpi=200)
plt.show()

# %%
# Get upper diagonal triangle (non duplication of features)
cor_matrix = encoded_features.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Identify highly correlated features
to_drop = [
    column
    for column in upper_tri.columns
    if any(upper_tri[column] > 0.75)
    if column not in ["HP_INSTALLED"]
]

print(to_drop)

# Drop features
uncorrelated_features = encoded_features.drop(to_drop, axis=1)
print(uncorrelated_features.shape)

# %%

# %%
