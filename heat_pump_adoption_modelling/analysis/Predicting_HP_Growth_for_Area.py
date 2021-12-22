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
    plotting_utils,
    data_aggregation,
    data_preprocessing,
    hp_growth_prediction,
    hp_status_prediction,
    error_analysis,
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

from ipywidgets import interact

import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

# %%
epc_df = data_preprocessing.epc_sample_loading(subset="5m", preload=True)
epc_df = data_preprocessing.data_preprocessing(epc_df, encode_features=False)

# %%
epc_df = epc_df.drop(columns=data_preprocessing.drop_features)

# %%
drop_features = ["HP_INSTALL_DATE"]
postcode_level = "POSTCODE_UNIT"

aggr_temp = data_preprocessing.get_aggregated_temp_data(
    epc_df, 2015, 2018, postcode_level, drop_features=drop_features
)

# %%
X, y = hp_growth_prediction.get_data_with_labels(
    aggr_temp, ["GROWTH", "HP_COVERAGE_FUTURE"], drop_features=[postcode_level]
)

model = hp_growth_prediction.predict_hp_growth_for_area(X, y, save_predictions=True)

# %%
import pandas as pd

df = pd.read_csv(
    hp_growth_prediction.SUPERVISED_MODEL_OUTPUT
    + "Predictions_with_Random Forest Regressor.csv"
)
df.head()

# %%
df["training set"].value_counts(dropna=False)

# %%
df["No growth"] = df["HP_COVERAGE_FUTURE"] == 0.0

# %%
print("HP Coverage Future")
print("----------------")
error_analysis.print_prediction_and_error(df, "HP_COVERAGE_FUTURE")
print()

print("Growth")
print("----------------")
error_analysis.print_prediction_and_error(df, "GROWTH")

# %%
# df['coverage ground truth int'] = round(df['coverage ground truth']*100,0)
# df['coverage prediction int'] = round(df['coverage prediction']*100,0)
# df['coverage error int'] = round(df['coverage error']*100,0)

# df['growth ground truth int'] = round(df['growth ground truth']*100,0)
# df['growth prediction int'] = round(df['growth prediction']*100,0)
# df['growth error int'] = round(df['growth error']*100,0)

# %%
df = error_analysis.decode_tenure(df)
df["tenure color"] = df["TENURE"].map(error_analysis.tenure_color_dict)

test = df.loc[df["training set"] == False]
train = df.loc[df["training set"] == True]

set_dict = {"Validation Set": test, "Training Set": train, "Full Set": df}


# %%
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

    sample_x = df.loc[df["POSTCODE"] == postcode]
    print(feature)
    print(sample_x[feature].value_counts(dropna=False))

    print(sample_x[feature].unique())
    print(sample_x[feature].max())
    print(sample_x[feature].min())


# %%
