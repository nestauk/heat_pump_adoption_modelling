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

# %% [markdown]
# # Predicting the Heat Pump Status for Households

# %% [markdown]
# ### Imports

# %%
# %load_ext autoreload
# %autoreload 2

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
from heat_pump_adoption_modelling.getters import epc_data, deprivation_data

from heat_pump_adoption_modelling.pipeline.supervised_model import (
    data_aggregation,
    data_preprocessing,
    hp_growth_prediction,
    hp_status_prediction,
    prediction_pipeline,
)

from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

from heat_pump_adoption_modelling.pipeline.encoding import (
    feature_encoding,
    category_reduction,
)

from heat_pump_adoption_modelling.pipeline.supervised_model.utils import plotting_utils

import pandas as pd

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

from ipywidgets import interact

# %% [markdown]
# ### Loading, preprocessing and encoding

# %%
# If preloaded file is not yet available:
# epc_df = data_preprocessing.load_epc_samples(subset="5m", preload=True)
# epc_df = data_preprocessing.preprocess_data(epc_df, encode_features=False)

# %%
# epc_df = pd.read_csv(data_preprocessing.SUPERVISED_MODEL_OUTPUT + "epc_df_5m_preprocessed.csv")
# epc_df = data_preprocessing.encode_features_for_hp_status(epc_df)
# epc_df.head()

# %%
epc_df = pd.read_csv(
    data_preprocessing.SUPERVISED_MODEL_OUTPUT + "epc_df_5m_encoded.csv"
)
epc_df.head()

# %% [markdown]
# ### Get training data and labels

# %%
version = "Future HP Status"

# %%
drop_features = [
    "POSTCODE_AREA",
    "POSTCODE_DISTRICT",
    "POSTCODE_SECTOR",
    "POSTCODE_UNIT",
]

X, y = prediction_pipeline.get_data_with_labels(
    epc_df, version=version, drop_features=drop_features, balanced_set=True
)
X.head()

# %% [markdown]
# ### Train the Predictive Model

# %%
prediction_pipeline.predict_heat_pump_adoption(X, y, version, save_predictions=True)

# %% [markdown]
# ### Dimensionality Reduction and Feature Coefficients

# %%
X_scaled = prediction_pipeline.model_settings.prepr_pipeline_no_pca.fit_transform(X)

# Reduce dimensionality to level of 90% explained variance ratio
X_dim_reduced = plotting_utils.dimensionality_reduction(
    X_scaled,
    dim_red_technique="pca",
    pca_expl_var_ratio=0.90,
    random_state=42,
)

# %%
prediction_pipeline.coefficient_importance(
    X, y, "Linear Support Vector Classifier", version=version, pca=False
)

# %%
prediction_pipeline.coefficient_importance(
    X, y, "Linear Support Vector Classifier", version=version, pca=True
)

# %% [markdown]
# ### Error Analysis
#
# #### Does not need to be reviewed just yet.

# %%
# Loading the predictions and errors
df = pd.read_csv(
    hp_status_prediction.SUPERVISED_MODEL_OUTPUT
    + "household_based_future_predictions_with_logistic_regression.csv"
)
df.head()

# %%
df["proba 1"].max()

# %%
tar = "HP_ADDED"

test = df.loc[df["training set"] == False]

social = test["TENURE: rental (social)"] == True
private = test["TENURE: rental (private)"] == True
owner_occupied = test["TENURE: owner-occupied"] == True
high_conf_hp = test["proba 1"] > 0.9

false_positives = (test[tar + ": prediction"] == True) & (test[tar] == False)
false_negatives = (test[tar + ": prediction"] == False) & (test[tar] == True)
true_positives = (test[tar + ": prediction"] == True) & (test[tar] == True)
true_negatives = (test[tar + ": prediction"] == False) & (test[tar] == False)

print(social.shape)
print(private.shape)
print(owner_occupied.shape)

# %%
test[true_positives & owner_occupied & high_conf_hp]

# %%

# %%
