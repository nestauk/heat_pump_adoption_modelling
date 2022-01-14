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

from heat_pump_adoption_modelling.pipeline.supervised_model import (
    plotting_utils,
    data_aggregation,
    data_preprocessing,
    hp_growth_prediction,
    hp_status_prediction,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

from heat_pump_adoption_modelling.pipeline.encoding import (
    feature_encoding,
    category_reduction,
)


import pandas as pd
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

from ipywidgets import interact


# %%
# epc_df = data_preprocessing.epc_sample_loading(subset="5m", preload=True)
# epc_df = data_preprocessing.data_preprocessing(epc_df, encode_features=False)

epc_df = pd.read_csv(
    data_preprocessing.SUPERVISED_MODEL_OUTPUT + "epc_df_preprocessed.csv"
)
epc_df = data_preprocessing.feature_encoding_for_hp_status(epc_df)

# %%
drop_features = [
    # "POSTCODE",
    "POSTCODE_AREA",
    "POSTCODE_DISTRICT",
    "POSTCODE_SECTOR",
    "POSTCODE_UNIT",
    "HP_INSTALL_DATE",
]

X, y = hp_status_prediction.get_data_with_labels(
    epc_df, version="Future HP Status", drop_features=drop_features, balanced_set=True
)


# %%
X.head()

# %%
model = hp_status_prediction.predict_heat_pump_status(X, y, save_predictions=True)

# %%
df = pd.read_csv(
    hp_status_prediction.SUPERVISED_MODEL_OUTPUT
    + "Predictions_with_Support Vector Classifier.csv"
)

# df = pd.read_csv(hp_status_prediction.SUPERVISED_MODEL_OUTPUT + "Predictions_with_Logistic Regression.csv")

df.head()

# %%
test = df.loc[df["training set"] == False]

social = test["TENURE: rental (social)"] == True
private = test["TENURE: rental (private)"] == True
owner_occupied = test["TENURE: owner-occupied"] == True
high_conf_hp = test["proba 2"] > 0.95

false_positives = (test["prediction"] == True) & (test["ground truth"] == False)
false_negatives = (test["prediction"] == False) & (test["ground truth"] == True)
true_positives = (test["prediction"] == True) & (test["ground truth"] == True)
true_negatives = (test["prediction"] == False) & (test["ground truth"] == False)

print(social.shape)
print(private.shape)
print(owner_occupied.shape)

# %%
test[true_positives & owner_occupied & high_conf_hp]

# %% [markdown]
# ## Scaling and Dimensionality Reduction
#
# ... and some functions
#
#

# %%
X_scaled = hp_status_prediction.prepr_pipeline_no_pca.fit_transform(X)

# Reduce dimensionality to level of 90% explained variance ratio
X_dim_reduced = plotting_utils.dimensionality_reduction(
    X_scaled,
    dim_red_technique="pca",
    pca_expl_var_ratio=0.90,
    random_state=42,
)

# %%
hp_status_prediction.coefficient_importance(
    X, y, "Linear Support Vector Classifier", version="Future HP Status", pca=False
)

# %%
hp_status_prediction.coefficient_importance(
    X, y, "Linear Support Vector Classifier", version="Future HP Status", pca=True
)

# %%