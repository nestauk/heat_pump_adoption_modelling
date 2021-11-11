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
epc_df = category_reduction.category_reduction(epc_df)


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
    # "FLOOR_LEVEL",
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

encoded_features = feature_encoding.encode_ordinal_cat_features(
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

# one_hot_features = ['MECHANICAL_VENTILATION', 'ENERGY_TARIFF', 'SOLAR_WATER_HEATING_FLAG', 'TENURE', 'TRANSACTION_TYPE', 'BUILT_FORM', 'PROPERTY_TYPE', 'COUNTRY', 'HEATING_FUEL']
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
plt.savefig(FIGPATH + "correlation_matrix.png", dpi=200)
plt.show()

# %% [markdown]
# ## Future Heat Pump Status

# %%
current_hp_df = feature_engineering.filter_by_year(
    encoded_features, "BUILDING_ID", None, selection="latest entry"
)

# %%
future_hp_df = encoded_features.copy()

# %%
future_hp_df = future_hp_df.loc[future_hp_df["N_ENTRIES_BUILD_ID"] >= 2]
latest_df = feature_engineering.filter_by_year(
    future_hp_df, "BUILDING_ID", None, selection="latest entry"
)
first_df = feature_engineering.filter_by_year(
    future_hp_df, "BUILDING_ID", None, selection="first entry"
)


# %%
latest_df["NOW_HP"] = latest_df["HP_INSTALLED"]
first_df["PAST_HP"] = first_df["HP_INSTALLED"]

print(first_df.shape)
print(latest_df.shape)

# unique_epc_df =  feature_engineering.filter_by_year(encoded_copy, "BUILDING_ID", None, selection="first entry")
before_after_df = pd.merge(
    latest_df[["NOW_HP", "BUILDING_ID"]],
    first_df[["PAST_HP", "BUILDING_ID"]],
    on=["BUILDING_ID"],
)
print(before_after_df.shape)
before_after_df.head()


# %%
before_after_df["HP_ADDED"] = (before_after_df["NOW_HP"] == True) & (
    before_after_df["PAST_HP"] == False
)

before_after_df["HP_REMOVED"] = (before_after_df["NOW_HP"] == False) & (
    before_after_df["PAST_HP"] == True
)

before_after_df["ALWAYS_HP"] = (before_after_df["NOW_HP"] == True) & (
    before_after_df["PAST_HP"] == True
)

before_after_df["NEVER_HP"] = (before_after_df["NOW_HP"] == False) & (
    before_after_df["PAST_HP"] == False
)

print(before_after_df["HP_ADDED"].sum())
print(before_after_df["NEVER_HP"].sum())
before_after_df.head()

# %%
del first_df["PAST_HP"]
future_hp_df = pd.merge(first_df, before_after_df, on=["BUILDING_ID"])
print(future_hp_df.shape)
future_hp_df.head()

# %%
future_hp_df = future_hp_df.loc[(future_hp_df["PAST_HP"] == False)]

future_hp_df = future_hp_df.drop(
    columns=["PAST_HP", "NOW_HP", "HP_REMOVED", "ALWAYS_HP", "NEVER_HP"]
)

# %%
future_hp_df.columns

# %% [markdown]
# ## Prepare Training and Eval Data

# %%
static_model = ["Current HP Status", "Future HP Status"][1]
balanced_set = True


def balance_set(X, target_variable, ratio=0.5):

    multiplicator = ratio / (1 - ratio)

    # Seperate samples with and without heat pumps
    X_true = X.loc[X[target_variable] == True]
    X_false = X.loc[X[target_variable] == False]

    # Shuffle and adjust size
    X_false = X_false.sample(frac=1)

    X_false = X_false[: int(X_true.shape[0] * multiplicator)]

    print(X_true.shape)
    print(X_false.shape)

    X = pd.concat([X_true, X_false], axis=0)
    print(X.shape)

    X = X.sample(frac=1)

    return X


if static_model == "Current HP Status":

    X = current_hp_df
    target_variable = "HP_INSTALLED"

elif static_model == "Future HP Status":

    X = future_hp_df
    target_variable = "HP_ADDED"

else:
    print("not defined")


X.fillna(X.mean(), inplace=True)
if balanced_set:
    X = balance_set(X, target_variable)

y = X[target_variable]

for feat in ["HP_INSTALLED", "HP_ADDED"]:
    if feat in X.columns:
        del X[feat]

# Check for NaN patterns
# or/and: use mean/median or impyute
# or: can we drop them?

print()
print(X.shape)
print(y.shape)


# %%
@interact(feature=X.columns)
def value_counts(feature):
    print(feature)
    print(X[feature].value_counts(dropna=False))

    print(X[feature].unique())


# %%
X.columns

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

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
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
    model, feature_names, label_set, static_model + ": Coefficient Contributions PCA"
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
    model, feature_names, label_set, static_model + ": Coefficient Contributions PCA"
)

# %%

# %%

# %%

# %%
