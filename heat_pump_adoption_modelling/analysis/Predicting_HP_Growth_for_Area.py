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

from heat_pump_adoption_modelling.pipeline.supervised_model.utils import (
    plotting_utils,
    error_analysis,
    kepler,
    hyperparameter_screening,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

from ipywidgets import interact

import matplotlib as mpl
import pandas as pd

from keplergl import KeplerGl

mpl.rcParams.update(mpl.rcParamsDefault)

# %%
# epc_df = data_preprocessing.epc_sample_loading(subset="5m", preload=True)
# epc_df = data_preprocessing.data_preprocessing(epc_df, encode_features=False)

epc_df = pd.read_csv(
    data_preprocessing.SUPERVISED_MODEL_OUTPUT + "epc_df_preprocessed.csv"
)
epc_df = epc_df.drop(columns=data_preprocessing.drop_features)
epc_df.head()

# %%
aggr_temp.head()

# %%
drop_features = ["HP_INSTALL_DATE"]
postcode_level = "POSTCODE_UNIT"

aggr_temp = data_preprocessing.get_aggregated_temp_data(
    epc_df, 2015, 2018, postcode_level, drop_features=drop_features
)

# %%
X, y = hp_growth_prediction.get_data_with_labels(
    aggr_temp, ["GROWTH", "HP_COVERAGE_FUTURE"], drop_features=[]
)

X.head()

# %%
X.columns[X.isna().any()].tolist()

# %%
X.head()

# %%
hyperparameter_screening.grid_screening(
    "Random Forest Regressor", X, y, "neg_mean_squared_error"
)

# %%
model = hp_growth_prediction.predict_hp_growth_for_area(X, y, save_predictions=True)

# %%
df = pd.read_csv(
    hp_growth_prediction.SUPERVISED_MODEL_OUTPUT
    + "area_based_predictions_with_random_forest_regressor.csv"
)
df.head()

# %%
print("POSTCODE_UNIT" in list(df.columns) or "POSTCODE" in list(df.columns))

# %%
df["No heat pumps"] = df["HP_COVERAGE_FUTURE"] == 0.0

# %%
print("HP Coverage Future")
print("----------------")
error_analysis.print_prediction_and_error(df, "HP_COVERAGE_FUTURE")
print()

print("Growth")
print("----------------")
error_analysis.print_prediction_and_error(df, "GROWTH")

# %%
df = error_analysis.decode_tenure(df)
df["tenure color"] = df["TENURE"].map(error_analysis.tenure_color_dict)

test = df.loc[df["training set"] == False]
train = df.loc[df["training set"] == True]

set_dict = {"Validation Set": test, "Training Set": train, "Full Set": df}

# %%
pd.options.mode.chained_assignment = None


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
            "No growth",
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
            "No growth",
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
