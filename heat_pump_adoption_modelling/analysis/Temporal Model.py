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
    utils,
    data_aggregation,
)
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
    version=version, nrows=500000, usecols=None
)


epc_df.head()

# %%
print(epc_df.shape)
imd_df = deprivation_data.get_gb_imd_data()
epc_df = deprivation_data.merge_imd_with_other_set(
    imd_df, epc_df, postcode_label="POSTCODE"
)
print(imd_df.shape)
print(epc_df.shape)

# %%
epc_df["original_address"] = (
    epc_df["ADDRESS1"] + epc_df["ADDRESS2"] + epc_df["POSTCODE"]
)

# %%
epc_df.head()

# %%
POSTCODE_LEVEL = "POSTCODE_SECTOR"
epc_df = data_aggregation.get_postcode_levels(epc_df, only_keep=POSTCODE_LEVEL)

print(epc_df.columns)

# %% [markdown]
# 60690 MCS samples
# 51150 with EPC match

# %%
mcs_data = pd.read_csv(
    str(PROJECT_DIR) + "/outputs/mcs_epc.csv",
    usecols=["date", "tech_type", "original_address"],
)

print(mcs_data.shape)
mcs_data.head()

# %%
mcs_data = mcs_data.loc[~mcs_data["original_address"].isna()]
print(mcs_data.shape)
mcs_data.columns = ["HP_INSTALL_DATE", "Type of HP", "original_address"]
mcs_data.head()

# %%
epc_df["original_address"].unique()

# %%
mcs_data["original_address"].unique()

# %%
list(set(epc_df["original_address"]) & set(mcs_data["original_address"]))

# %%
print(epc_df.shape)
print(mcs_data.shape)
combo = pd.merge(epc_df, mcs_data, on=["original_address"])
print(combo.shape)
combo.head()

# %%

# %%
epc_df.loc[epc_df[POSTCODE_LEVEL].isna()].head()

# %%
epc_df[POSTCODE_LEVEL].value_counts(dropna=True)

# %%
ordinal_features = [
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
    "INSPECTION_DATE",
    "MAIN_FUEL",
    # "HEATING_SYSTEM",
    "HP_TYPE",
]

# %%
epc_df_encoded = feature_encoding.feature_encoding_pipeline(
    epc_df,
    ordinal_features,
    reduce_categories=True,
    onehot_features=None,
    target_variables=None,
    drop_features=drop_features,
)

epc_df_encoded.head()

# %%
numeric_features = epc_df_encoded.select_dtypes(include=np.number).columns.tolist()
print(numeric_features)
print(len(numeric_features))
categorical_features = [
    feature
    for feature in epc_df_encoded.columns
    if (feature not in ordinal_features) and (feature not in numeric_features)
]

categorical_features = [
    f for f in categorical_features if f not in [POSTCODE_LEVEL, "HP_INSTALLED"]
]
print(categorical_features)
print(len(categorical_features))


# %%
def get_year_range_data(df, years):
    year_range_df = pd.concat(
        [
            feature_engineering.filter_by_year(
                df,
                "BUILDING_ID",
                year,
                selection="latest entry",
                up_to=True,
            )
            for year in years
        ],
        axis=0,
    )

    return year_range_df


# training_years = get_year_range_data(epc_df_encoded, [2008, 2009, 2010, 2011, 2012])
# prediction_years = get_year_range_data(epc_df_encoded, [2014, 2013, 2015, 2016])

data_time_t = feature_engineering.filter_by_year(
    epc_df_encoded, "BUILDING_ID", 2015, selection="latest entry", up_to=True
)
data_time_t_plus_one = feature_engineering.filter_by_year(
    epc_df_encoded, "BUILDING_ID", 2018, selection="latest entry", up_to=True
)

# %%
num_agglomerated = (
    data_time_t[numeric_features + [POSTCODE_LEVEL]].groupby([POSTCODE_LEVEL]).median()
)
num_agglomerated = num_agglomerated.reset_index()
num_agglomerated.tail()

# %%
cat_agglomerated = data_aggregation.aggreate_categorical_features(
    data_time_t, categorical_features, agglo_feature=POSTCODE_LEVEL
)

cat_agglomerated.tail()


# %%
def get_feature_count_grouped(
    df, feature, groupby_f, name=None
):  # Get the feature categories by the agglomeration feature

    if name is None:
        name = feature + ": True"

    feature_cats_by_agglo_f = (
        df.groupby([groupby_f, feature]).size().unstack(fill_value=0)
    ).reset_index()

    feature_cats_by_agglo_f.rename(columns={True: name}, inplace=True)

    return feature_cats_by_agglo_f[[groupby_f, name]]


n_hp_installed_current = get_feature_count_grouped(
    data_time_t, "HP_INSTALLED", POSTCODE_LEVEL, name="# HP at Time t"
)
n_hp_installed_future = get_feature_count_grouped(
    data_time_t_plus_one, "HP_INSTALLED", POSTCODE_LEVEL, name="# HP at Time t+1"
)
n_hp_installed_total = get_feature_count_grouped(
    epc_df_encoded, "HP_INSTALLED", POSTCODE_LEVEL, name="Total # HP"
)

total_basis = [data_time_t_plus_one, epc_df_encoded][0]
n_prop_total = (
    total_basis.groupby([POSTCODE_LEVEL]).size().reset_index(name="# Properties")
)

print(n_prop_total.shape)
n_prop_total.head()

# %%
target_var_df = pd.merge(
    n_hp_installed_current, n_hp_installed_future, on=POSTCODE_LEVEL
)
target_var_df = pd.merge(target_var_df, n_prop_total, on=POSTCODE_LEVEL)
target_var_df["HP_COVERAGE_CURRENT"] = (
    target_var_df["# HP at Time t"] / target_var_df["# Properties"]
)
target_var_df["HP_COVERAGE_FUTURE"] = (
    target_var_df["# HP at Time t+1"] / target_var_df["# Properties"]
)
target_var_df["GROWTH"] = (
    target_var_df["HP_COVERAGE_FUTURE"] - target_var_df["HP_COVERAGE_CURRENT"]
)

print(target_var_df.shape)
target_var_df.head()

# %%
samples_to_discard = list(
    target_var_df.loc[target_var_df["GROWTH"] < 0.0][POSTCODE_LEVEL]
)
print(
    "Number of samples to discard because of negative growth: {}".format(
        len(samples_to_discard)
    )
)

# %%
agglomerated_df = pd.concat(
    [cat_agglomerated, num_agglomerated.drop(columns=[POSTCODE_LEVEL])], axis=1
)

print(agglomerated_df.shape)
print(target_var_df.shape)

agglomerated_with_target = pd.merge(
    agglomerated_df,
    target_var_df[
        ["HP_COVERAGE_CURRENT", "HP_COVERAGE_FUTURE", "GROWTH", POSTCODE_LEVEL]
    ],
    on=POSTCODE_LEVEL,
)

print(agglomerated_with_target.shape)
agglomerated_with_target.head()

# %%
most_freq_features = [
    col for col in agglomerated_with_target.columns if "MOST_FREQUENT" in col
]
agglomerated_with_target = agglomerated_with_target.drop(columns=most_freq_features)

print(agglomerated_with_target.shape)
agglomerated_with_target = agglomerated_with_target[
    ~agglomerated_with_target[POSTCODE_LEVEL].isin(samples_to_discard)
]
print(agglomerated_with_target.shape)

# %%
target_variables = ["GROWTH", "HP_COVERAGE_FUTURE"]
TARGET_VARIABLE = target_variables[1]

X = agglomerated_with_target
y = np.array(X[TARGET_VARIABLE])

for col in target_variables + [POSTCODE_LEVEL]:
    if col in X.columns:
        del X[col]

print(X.shape)
X = X.dropna(axis="columns", how="all")
print(X.shape)
# X.fillna(X.mean(), inplace=True)

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

prepr_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9, random_state=42)),
    ]
)

# %%
X = prepr_pipeline.fit_transform(X)
print(X.shape)

# %%
# Reduce dimensionality to level of 90% explained variance ratio
# X_dim_reduced = utils.dimensionality_reduction(
#    X_scaled,
#    dim_red_technique="pca",
#    pca_expl_var_ratio=0.90,
#    random_state=42,
# )
# print(X_dim_reduced.shape)

# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.1,
    random_state=42,  # stratify=y
)

# %%

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

model_dict = {
    "SVM Regressor": svm.SVR(),
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
}
cv = 3
interval = 5

best_params = {
    "SVM Regressor": {"C": 10, "gamma": 0.01, "kernel": "rbf"},
    "Linear Regression": {},
    "Decision Tree Regressor": {},  # {'max_depth': 11, 'max_features': None,
    # 'max_leaf_nodes': 10, 'min_samples_leaf': 5,
    # 'min_weight_fraction_leaf': 0.05, 'splitter': 'random'},
    "Random Forest Regressor": {"max_features": 12, "n_estimators": 30},
}


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def train_and_evaluate(model_name):

    model = model_dict[model_name]
    model.set_params(**best_params[model_name])
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("\n*****************\nModel Name: {}\n*****************".format(model_name))

    if TARGET_VARIABLE in ["HP_COVERAGE_FUTURE", "GROWTH"]:

        variable_name = (
            "HP Coverage at t+1"
            if TARGET_VARIABLE == "HP_COVERAGE_FUTURE"
            else "Growth"
        )

        for set_name in ["train", "val"]:

            if set_name == "train":
                preds = pred_train
                sols = y_train
                set_name = "Training Set"
            elif set_name == "val":
                preds = pred_test
                sols = y_test
                set_name = "Validation Set"

            print("\n-----------------\n{}\n-----------------".format(set_name))
            print()

            preds[preds < 0] = 0.0
            preds[preds > 1.0] = 1.0

            predictions, solutions, label_dict = utils.map_percentage_to_bin(
                preds, sols, interval=interval
            )

            overlap = round((predictions == solutions).sum() / predictions.shape[0], 2)
            print("Category Accuracy with {}% steps : {}".format(interval, overlap))

            # Plot the confusion matrix for training set
            utils.plot_confusion_matrix(
                solutions,
                predictions,
                [label_dict[label] for label in sorted(set(solutions))],
                title="Confusion Matrix:\n{} using {} on {}".format(
                    variable_name, model_name, set_name
                ),
            )

            plt.scatter(preds, sols)
            plt.title("{} using {} on {}".format(variable_name, model_name, set_name))
            plt.xlabel("Prediction")
            plt.ylabel("Ground Truth")
            plt.show()

            scores = cross_val_score(
                model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
            )
            rsme_scores = np.sqrt(-scores)
            display_scores(rsme_scores)


for model in model_dict.keys():
    train_and_evaluate(model)

# %%
from sklearn.model_selection import GridSearchCV

param_grid_dict = {
    "Random Forest Regressor": [
        {
            "n_estimators": [2, 5, 10, 22, 25, 30, 35, 40, 45, 50],
            "max_features": [1, 2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 20],
        },
        # {'bootstrap': [False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
    ],
    "SVM Regressor": {
        "gamma": [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        "kernel": ["rbf"],
        "C": [0.001, 0.01, 0.1, 1, 2, 5, 10, 15, 20],
    },
    "Decision Tree Regressor": {
        "splitter": ["best", "random"],
        "max_depth": [1, 3, 5, 7, 9, 11, 12],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "min_weight_fraction_leaf": [0.25, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        "max_features": ["auto", "log2", "sqrt", None],
        "max_leaf_nodes": [None, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    },
}


def parameter_screening(model_name, X, y):

    model = model_dict[model_name]
    param_grid = param_grid_dict[model_name]

    grid_search = GridSearchCV(
        model, param_grid, cv=5, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)
    print(grid_search.best_params_)


for model in model_dict.keys():
    if model not in ["Linear Regression"]:
        print(model)
        parameter_screening(model, X_train, y_train)
        print()

# %%
para_grid = {
    "penalty": ["l2", "l1", "elasticnet"],
    "alpha": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0],
    "epsilon": [0.001, 0.01, 0.1, 1.0],
    "tol": [0.0001, 0.001, 0.01, 0.1, 1.0],
}

# %%

# %%

# %%

# %%

# %%
