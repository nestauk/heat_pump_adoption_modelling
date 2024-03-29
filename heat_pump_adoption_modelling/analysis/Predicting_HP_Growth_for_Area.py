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

# %% [markdown]
# # Predicting the Heat Pump Growth for Areas

# %% [markdown]
# ### Imports

# %%
# %load_ext autoreload
# %autoreload 2


from heat_pump_adoption_modelling.pipeline.supervised_model import (
    data_preprocessing,
    hp_growth_prediction,
    prediction_pipeline,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    feature_engineering,
)

from heat_pump_adoption_modelling.pipeline.supervised_model.utils import error_analysis
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import kepler

import pandas as pd
from keplergl import KeplerGl

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

from ipywidgets import interact

# %% [markdown]
# ### Loading, preprocessing, agglomerating on postcode level, and encoding

# %%
# If preloaded file is not yet available:
epc_df = data_preprocessing.load_epc_samples(subset="5m", preload=True)

# %%
epc_df = data_preprocessing.preprocess_data(epc_df, encode_features=False)

# %%
epc_df = pd.read_csv(
    data_preprocessing.SUPERVISED_MODEL_OUTPUT + "epc_df_5m_preprocessed.csv",
    parse_dates=["INSPECTION_DATE", "HP_INSTALL_DATE"],
)

# %%
epc_df = epc_df.drop(columns=data_preprocessing.drop_features)
epc_df.head()

# %%
drop_features = [
    "HP_INSTALL_DATE",
    "INSPECTION_DATE",
    "HP_ADDED",
    "SECONDHEAT_DESCRIPTION",
]

postcode_level = "POSTCODE_UNIT"

aggr_temp = data_preprocessing.get_aggregated_temp_data(
    epc_df, 2012, 2021, postcode_level, drop_features=drop_features
)
print(aggr_temp.shape)

# %% [markdown]
# ### Get training data and labels

# %%
X, y = prediction_pipeline.get_data_with_labels(
    aggr_temp, version="HP Growth by Area", drop_features=[], balanced_set=True
)

print(X.shape)
X.head()

# %%
b = y.shape[0]
a = y.loc[y["HP_COVERAGE_FUTURE"] == 0.0].shape[0]

print(b)
print(a)

100 / b * a


# %% [markdown]
# #### Train the Predictive Model

# %%
y

# %%
# Commented out to save time

# hyperparameter_screening.grid_screening(
#     "Random Forest Regressor",
#     #"Linear Support Vector Classifier",
#     X,
#     y['GROWTH'],
#     "neg_mean_squared_error",
#     drop_features,
# )

# %%
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import model_settings

model_settings.best_params["Random Forest Regressor"]

# %%
prediction_pipeline.predict_heat_pump_adoption(
    X, y, "HP Growth by Area", save_predictions=True
)

# %% [markdown]
# ### Error Analysis
#
# ##### Does not yet have to be reviewed.

# %% [markdown]
# Predicting HP Growth by Area...
#
# Number of samples: 292391
# Number of features: 149
#
#
# ******************************
# Random Forest Regressor GROWTH
# ******************************
# Category Accuracy with 5% steps : 0.99
# Category Accuracy with 5% steps : 0.98
#
# Predicting HP Growth by Area...
#
# Number of samples: 292391
# Number of features: 149
#
#
# ******************************
# Random Forest Regressor
# ******************************
# Category Accuracy with 5% steps : 0.99
#
# Category Accuracy with 5% steps : 0.98
#
#
# 5-fold Cross Validation: Training Set
# ----------------------------------------
#
# neg_mean_squared_error mean: -0.0
# max_error mean: -0.46
# rsme mean: 0.01
#
# 5-fold Cross Validation: Test Set
# ----------------------------------------
#
# neg_mean_squared_error mean: -0.0
# max_error mean: -1.0
# rsme mean: 0.03
#
#
# ******************************
# Random Forest Regressor HP COVERAGE
# ******************************
# Category Accuracy with 5% steps : 0.97
# Category Accuracy with 5% steps : 0.93
#

# %%
# Load predictions, errors etc.
df = pd.read_csv(
    hp_growth_prediction.SUPERVISED_MODEL_OUTPUT
    + "Predictions_with_Random_Forest_Regressor.csv"
).drop(columns="Unnamed: 0")
df.head()

# %%
# Areas with no heat pump at latest stage
df["No heat pumps"] = df["HP_COVERAGE_FUTURE"] == 0.0

# %%
error_analysis.print_prediction_and_error(df, "HP_COVERAGE_FUTURE")
error_analysis.print_prediction_and_error(df, "GROWTH")

# %%
df = error_analysis.decode_tenure(df)
df["tenure color"] = df["TENURE"].map(error_analysis.tenure_color_dict)

test = df.loc[df["training set"] == False]
train = df.loc[df["training set"] == True]

set_dict = {"Validation Set": test, "Training Set": train, "Full Set": df}

# %%
pd.options.mode.chained_assignment = None  # default='warn'


@interact(
    target=["GROWTH", "HP_COVERAGE_FUTURE"],
    y_axis=["Ground Truth", "Predictions"],
    model_name=["Random Forest Regressor"],
    set_name=["Validation Set", "Training Set", "Full Set"],
)
def plot_error(target, y_axis, model_name, set_name):

    df = set_dict[set_name]

    error_analysis.plot_error_by_tenure(
        df, target, y_axis=y_axis, model_name=model_name, set_name=set_name
    )


# %%
postcode = "CM07FY"


@interact(feature=df.columns)
def value_counts(feature):

    sample_x = df.loc[df["POSTCODE_UNIT"] == postcode]
    print(feature)
    print(sample_x[feature].value_counts(dropna=False))

    print(sample_x[feature].unique())
    print(sample_x[feature].max())
    print(sample_x[feature].min())


# %% [markdown]
# ### Kepler
#
# #### Still work in progress.

# %%
kepler_df = df.rename(
    columns={
        "POSTCODE_UNIT": "POSTCODE",
        "POSTCODE_UNIT_TOTAL": "# Properties",
        "HP_COVERAGE_FUTURE: prediction": "HP Coverage: prediction",
        "HP_COVERAGE_FUTURE: error": "HP Coverage: error",
        "HP_COVERAGE_FUTURE": "HP Coverage",
        "GROWTH: prediction": "Growth: prediction",
        "GROWTH: error": "Growth: error",
        "GROWTH": "Growth",
    }
)
kepler_df = feature_engineering.get_postcode_coordinates(kepler_df)

kepler_df["HP Coverage: error"] = round(kepler_df["HP Coverage: error"], 3)
kepler_df["HP Coverage: prediction"] = round(kepler_df["HP Coverage: prediction"], 3)
kepler_df["HP Coverage"] = round(kepler_df["HP Coverage"], 3)

kepler_df["Growth: error"] = round(kepler_df["Growth"], 3)
kepler_df["Growth: prediction"] = round(kepler_df["Growth"], 3)
kepler_df["Growth"] = round(kepler_df["Growth"], 3)

kepler_df.head()

# %%
config = kepler.get_config(kepler.KEPLER_PATH + "coverage.txt")

temp_model_map_coverage = KeplerGl(height=500, config=config)

temp_model_map_coverage.add_data(
    data=kepler_df[
        [
            "LONGITUDE",
            "LATITUDE",
            "HP Coverage: prediction",
            "HP Coverage: error",
            "HP Coverage",
            "# Properties",
            "training set",
            # "No growth",
            "POSTCODE",
        ]
    ],
    name="coverage",
)

temp_model_map_coverage

# %%
kepler.save_config(temp_model_map_coverage, kepler.KEPLER_PATH + "coverage.txt")

temp_model_map_coverage.save_to_html(file_name=kepler.KEPLER_PATH + "Coverage.html")

# %%
config = kepler.get_config(kepler.KEPLER_PATH + "growth.txt")

temp_model_map_growth = KeplerGl(height=500, config=config)

temp_model_map_growth.add_data(
    data=kepler_df[
        [
            "LONGITUDE",
            "LATITUDE",
            "Growth: prediction",
            "Growth: error",
            "Growth",
            "# Properties",
            "training set",
            # "No growth",
            "POSTCODE",
        ]
    ],
    name="coverage",
)

temp_model_map_growth

# %%
kepler.save_config(temp_model_map_growth, kepler.KEPLER_PATH + "growth.txt")
temp_model_map_growth.save_to_html(file_name=kepler.KEPLER_PATH + "Growth.html")

# %%
