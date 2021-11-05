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
from heat_pump_adoption_modelling.pipeline.encoding import feature_encoding
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

# Load all available columns
epc_df = epc_data.load_preprocessed_epc_data(
    version=version, nrows=500000, usecols=None
)
epc_df.columns


# %% [markdown]
# ## Categorical Feature Encoding

# %%


@interact(feature=epc_df.columns)
def value_counts(feature):
    print(feature)
    print(epc_df[feature].value_counts(dropna=False))

    print(epc_df[feature].unique())
    print(epc_df[feature].max())
    print(epc_df[feature].min())


# %%
encoded_features = epc_df.copy()
# Get all only numeric features
num_features = encoded_features.select_dtypes(include=np.number).columns.tolist()
print(len(num_features))
print(num_features)

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
    #  "MAINHEAT_ENERGY_EFF",
    "MAINHEATC_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    # "NUMBER_HABITABLE_ROOMS",
    "MAINS_GAS_FLAG",
    "CONSTRUCTION_AGE_BAND_ORIGINAL",
    "CONSTRUCTION_AGE_BAND",
    "N_ENTRIES",
    "N_ENTRIES_BUILD_ID",
    "ENERGY_RATING_CAT",
]
other_cat_features = [
    feature
    for feature in encoded_features.columns
    if (feature not in ordinal_cat_features) and (feature not in num_features)
]

encoded_features = feature_encoding.encode_ordinal_cat_features(
    encoded_features, ordinal_cat_features
)

encoded_features.head()

# %%
encoded_features[ordinal_cat_features].head()

# %%
# encoded_features = feature_encoding.encode_ordinal_cat_features(df, ordinal_cat_features)

# Set encoder and copy data
encoder = LabelEncoder()

# Encode categorical features
for feature in other_cat_features:
    encoded_features[feature] = encoder.fit_transform(encoded_features[feature])

# %%
print(encoded_features.shape)
encoded_features.head()

# %% [markdown]
# ## Identify and Filter Out Highly Correlated Features

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
print(encoded_features.columns)
encoded_features = encoded_features.drop(
    columns=[
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
    ]
)
# print(encoded_features.shape)

# %%
# Get upper diagonal triangle (non duplication of features)
# cor_matrix = encoded_features.corr().abs()
# upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Identify highly correlated features
# to_drop = [
#    column
#    for column in upper_tri.columns
#    if any(upper_tri[column] > 0.75)
#    if column not in ["HP_TYPE", "HP_INSTALLED"]
# ]
# print(to_drop)

# Drop features
# uncorrelated_features = encoded_features.drop(to_drop, axis=1)
# print(uncorrelated_features.shape)

# %% [markdown]
# ## Prepare Training and Eval Data

# %%
encoded_features.fillna(999, inplace=True)
# Check for NaN patterns
# or/and: use mean/median or impyute
# or: can we drop them?

balanced_set = True

if balanced_set:

    # Seperate samples with and without heat pumps
    X_hp = encoded_features.loc[encoded_features.HP_INSTALLED == True]
    X_no_hp = encoded_features.loc[encoded_features.HP_INSTALLED == False]

    # Shuffle and adjust size
    X_no_hp = X_no_hp.sample(frac=1)
    X_no_hp = X_no_hp[: X_hp.shape[0]]

    print(X_hp.shape)
    print(X_no_hp.shape)
    X = pd.concat([X_hp, X_no_hp], axis=0)
    print(X.shape)

else:
    X = encoded_features.copy()

# Set target value and remove from input
y = X["HP_INSTALLED"]
del X["HP_INSTALLED"]
del X["HP_TYPE"]
del X["HEATING_SYSTEM"]
# del X["MAINHEAT_DESCRIPTION"]
print()
print(X.shape)
print(y.shape)

# %% [markdown]
# ## Scaling and Dimensionality Reduction
#
# ... and some functions

# %%
# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Reduce dimensionality to level of 90% explained variance ratio
X_dim_reduced = utils.dimensionality_reduction(
    X_scaled,
    dim_red_technique="pca",
    pca_expl_var_ratio=0.90,
    random_state=42,
)

# %% [markdown]
# ## Training and Testing

# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_dim_reduced, y, test_size=0.1, random_state=42, stratify=y
)

# %%

print("Numer of samples:", X.shape[0])
print("Numer of features:", X.shape[1])
print()

model_dict = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear Support Vector Classifier": SGDClassifier(random_state=42),
}

if balanced_set:
    cv = 10
else:
    cv = 3


def train_and_evaluate(model_name):

    model = model_dict[model_name]
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("Model Name:", model_name)

    # Plot the confusion matrix for training set
    utils.plot_confusion_matrix(y_train, pred_train, ["No HP", "HP"], "Training set")

    # Plot the confusion matrix for validation set
    utils.plot_confusion_matrix(y_test, pred_test, ["No HP", "HP"], "Validation set")

    # Print Accuracies
    print(
        "Accuracy train: {}%".format(
            np.round(sklearn.metrics.accuracy_score(y_train, pred_train) * 100), 2
        )
    )
    print(
        "Accuracy test:   {}%".format(
            np.round(sklearn.metrics.accuracy_score(y_test, pred_test) * 100), 2
        )
    )

    # print(model.coef_, model.intercept_)

    acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
    f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    recall = cross_val_score(model, X_train, y_train, cv=cv, scoring="recall")
    precision = cross_val_score(model, X_train, y_train, cv=cv, scoring="precision")
    print()
    print("10-fold Cross Validation\n---------\n")
    print("Accuracy:", round(acc.mean(), 2))
    print("F1 Score:", round(f1.mean(), 2))
    print("Recall:", round(recall.mean(), 2))
    print("Precision:", round(precision.mean(), 2))
    print()


for model in model_dict.keys():
    train_and_evaluate(model)

# %% [markdown]
# ## Coefficients Inspection

# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_dim_reduced, y, test_size=0.1, random_state=42, stratify=y
)

model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)

print(X_train.shape)
print(model.coef_.shape)
feature_names = np.array(["PC " + str(num) for num in range(X_train.shape[1])])
print(feature_names)
label_set = ["Has HP"]

# Get feature indices sorted by coefficient strength
sort_idx, _, _ = utils.get_sorted_coefficients(model, feature_names)

# Plot the classifier's coefficients for each feature and label
plt = utils.plot_feature_coefficients(
    model, feature_names, label_set, "Coefficient Contributions PCA"
)

# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, stratify=y
)

model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)

print(X_train.shape)
print(model.coef_.shape)
feature_names = np.array(X.columns)
label_set = ["Has HP"]

# Get feature indices sorted by coefficient strength
sort_idx, _, _ = utils.get_sorted_coefficients(model, feature_names)

# Plot the classifier's coefficients for each feature and label
plt = utils.plot_feature_coefficients(
    model, feature_names, label_set, "Coefficient Contributions PCA"
)

# %%

# %%
