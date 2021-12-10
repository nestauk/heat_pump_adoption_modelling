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
)
from heat_pump_adoption_modelling.pipeline.preprocessing import (
    data_cleaning,
    feature_engineering,
)

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path

from ipywidgets import interact

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
    aggr_temp, "GROWTH", drop_features=["GROWTH", "HP_COVERAGE_FUTURE", postcode_level]
)

model = hp_growth_prediction.predict_hp_growth_for_area(
    X, y, target_variable="GROWTH", save_predictions=True
)

# %%

# %%
