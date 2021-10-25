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
import pandas as pd
import matplotlib.pyplot as plt

import re
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr

import sklearn
from sklearn.linear_model import LinearRegression

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.getters import epc_data

# %%
# Get parameters from widgets
version = "preprocessed_dedupl"

features_of_interest = [
    "TOTAL_FLOOR_AREA",
    "ENERGY_CONSUMPTION_CURRENT",
    "CURRENT_ENERGY_RATING",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "MAINS_GAS_FLAG",
    "TENURE",
    "TRANSACTION_TYPE",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "ENTRY_YEAR_INT",
    "N_ENTRIES_BUILD_ID",
    "HEATING_FUEL",
    "HP_INSTALLED",
]

epc_df = epc_data.load_preprocessed_epc_data(
    version=version, usecols=features_of_interest
)
# epc_df = preprocess_epc_data.load_and_preprocess_epc_data()

# %%
epc_df.columns

# %%
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

new_epc_df = epc_df.copy()

for feature in [
    "ENERGY_CONSUMPTION_CURRENT",
    "TOTAL_FLOOR_AREA",
    "CURRENT_ENERGY_RATING",
    "CO2_EMISS_CURR_PER_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "MAINS_GAS_FLAG",
    "TENURE",
    "TRANSACTION_TYPE",
    "BUILT_FORM",
    "PROPERTY_TYPE",
    "ENTRY_YEAR_INT",
    "N_ENTRIES_BUILD_ID",
    "HEATING_FUEL",
    "HP_INSTALLED",
]:

    new_epc_df[feature] = encoder.fit_transform(epc_df[feature])

# %%
new_epc_df.head()

# %%


# Using Pearson Correlation
plt.figure(figsize=(12, 10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()


# %%
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(new_epc_df, 0.1)

# %%
X = np.array(train_set[features_of_interest[:-1]])
y = np.array(train_set["HP_INSTALLED"])

# %%
y

# %%
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X, y)


# %%
from sklearn.model_selection import cross_val_score

y_train_pred = cross_val_score(sgd_clf, X, y, cv=3, scoring="accuracy")

# %%
