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
    version=version,
    # nrows=10000000,
    usecols=None,
)


epc_df.head()

# %%
epc_df.shape

# %%
for i in range(250000, 5000000, 1000):
    grouped_by = epc_df.groupby("POSTCODE").size().reset_index(name="count")[:i]
    n_samples = grouped_by["count"].sum()
    print(i)
    print(n_samples)
    print()
    if n_samples > 5000000:
        sample_ids = list(grouped_by["POSTCODE"])
        print(sample_ids)
        epc_df_reduced = epc_df.loc[epc_df["POSTCODE"].isin(sample_ids)]
        break


epc_df_reduced.shape

# %%
epc_df_reduced.shape

# %%
epc_df_reduced.to_csv(str(PROJECT_DIR) + "/outputs/epc_df_reduced.csv")

# %%
epc_df = epc_df_reduced

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
epc_df["original_address"] = (
    epc_df["original_address"].str.lower().replace(r"\s+", "", regex=True)
)

# %%
epc_df.loc[epc_df["HP_TYPE"] == "air source heat pump"].head()

# 	northlastsfarmhouseunknownab140pe
#   northlastsfarmhouseunknownab140pe

# %%
POSTCODE_LEVEL = "POSTCODE_SECTOR"
epc_df = data_aggregation.get_postcode_levels(epc_df)  # , only_keep=POSTCODE_LEVEL)

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
print(mcs_data.shape)
mcs_data = mcs_data.loc[~mcs_data["original_address"].isna()]
mcs_data["original_address"] = (
    mcs_data["original_address"].str.lower().replace(r"\s+", "", regex=True)
)
print(mcs_data.shape)
mcs_data.columns = ["HP_INSTALL_DATE", "Type of HP", "original_address"]
mcs_data["HP_INSTALL_DATE"] = (
    mcs_data["HP_INSTALL_DATE"].str.replace(r"-", "", regex=True).astype("float")
)
mcs_data.head()

# %%
date_dict = mcs_data.set_index("original_address").to_dict()["HP_INSTALL_DATE"]
hp_type_dict = mcs_data.set_index("original_address").to_dict()["Type of HP"]

# %%
epc_df["HP_INSTALL_DATE"] = epc_df["original_address"].map(date_dict)

# %%
first_hp_appearance = (
    epc_df.loc[epc_df["HP_INSTALLED"] == True]
    .groupby("BUILDING_ID")
    .min(["INSPECTION_DATE_AS_NUM"])
    .reset_index()[["INSPECTION_DATE_AS_NUM", "BUILDING_ID"]]
)
first_hp_appearance.columns = ["FIRST_HP_MENTION", "BUILDING_ID"]
first_hp_appearance.head()

# %%
print(epc_df.shape)
epc_df = pd.merge(epc_df, first_hp_appearance, on="BUILDING_ID", how="outer")
print(epc_df.shape)
epc_df.head()


# %%
epc_df["INSPECTION_YEAR"] = round(epc_df["INSPECTION_DATE_AS_NUM"] / 10000)
epc_df["HP_INSTALL_YEAR"] = round(epc_df["HP_INSTALL_DATE"] / 10000)
epc_df["FIRST_HP_MENTION_YEAR"] = round(epc_df["FIRST_HP_MENTION"] / 10000)
epc_df["FIRST_HP_MENTION_YEAR"].loc[~epc_df["HP_INSTALL_YEAR"].isna()].head()

# %%
epc_df["MCS_AVAILABLE"] = np.where(epc_df["HP_INSTALL_DATE"].isna(), False, True)
epc_df["HAS_HP_AT_SOME_POINT"] = np.where(
    epc_df["FIRST_HP_MENTION"].isna(), False, True
)
epc_df["HAS_HP_AT_SOME_POINT"].value_counts(dropna=False)
epc_df.head()


# %%
epc_df[
    (epc_df["HAS_HP_AT_SOME_POINT"] == True) & (epc_df["HP_INSTALLED"] == False)
].head()

# %%
mcs_available = epc_df["MCS_AVAILABLE"]

no_mcs_or_epc = (~epc_df["MCS_AVAILABLE"]) & (epc_df["HP_INSTALLED"] == False)

no_mcs_but_epc_hp = (~epc_df["MCS_AVAILABLE"]) & (epc_df["HP_INSTALLED"] == True)

mcs_and_epc_hp = (epc_df["MCS_AVAILABLE"]) & (epc_df["HP_INSTALLED"] == True)

no_epc_but_mcs_hp = (epc_df["MCS_AVAILABLE"]) & (epc_df["HP_INSTALLED"] == False)


either_hp = (epc_df["MCS_AVAILABLE"]) | (epc_df["HP_INSTALLED"] == True)


epc_hp_mention_before_mcs = epc_df["FIRST_HP_MENTION_YEAR"] < epc_df["HP_INSTALL_YEAR"]
mcs_before_epc_hp_mention = epc_df["FIRST_HP_MENTION_YEAR"] > epc_df["HP_INSTALL_YEAR"]
first_mention_same_as_mcs = epc_df["FIRST_HP_MENTION_YEAR"] == epc_df["HP_INSTALL_YEAR"]

epc_entry_before_mcs = epc_df["INSPECTION_YEAR"] < epc_df["HP_INSTALL_YEAR"]
mcs_before_epc_entry = epc_df["INSPECTION_YEAR"] > epc_df["HP_INSTALL_YEAR"]
epc_entry_same_as_mcs = epc_df["INSPECTION_YEAR"] == epc_df["HP_INSTALL_YEAR"]

# %%
# -----
# NO MCS/EPC HP entry
epc_df["HP_INSTALLED"] = np.where((no_mcs_or_epc), False, epc_df["HP_INSTALLED"])

epc_df["HP_INSTALL_DATE"] = np.where((no_mcs_or_epc), np.nan, epc_df["HP_INSTALL_DATE"])

# -----
# NO MCS entry but EPC HP
epc_df["HP_INSTALLED"] = np.where((no_mcs_but_epc_hp), True, epc_df["HP_INSTALLED"])

epc_df["HP_INSTALL_DATE"] = np.where(
    (no_mcs_but_epc_hp), epc_df["FIRST_HP_MENTION"], epc_df["HP_INSTALL_DATE"]
)

# -----
# MCS and EPC HP entry
epc_df["HP_INSTALLED"] = np.where((mcs_and_epc_hp), True, epc_df["HP_INSTALLED"])

epc_df["HP_INSTALL_DATE"] = np.where(
    (mcs_and_epc_hp),
    epc_df[["FIRST_HP_MENTION", "HP_INSTALL_DATE"]].min(axis=1),
    epc_df["HP_INSTALL_DATE"],
)
# -----
# MCS but EPC HP entry with same year

epc_df["HP_INSTALLED"] = np.where(
    (no_epc_but_mcs_hp & epc_entry_same_as_mcs), True, epc_df["HP_INSTALLED"]
)

epc_df["HP_INSTALL_DATE"] = np.where(
    (no_epc_but_mcs_hp & epc_entry_same_as_mcs),
    epc_df["HP_INSTALL_DATE"],
    epc_df["HP_INSTALL_DATE"],
)

# ---
# MCS but EPC HP with MCS before first EPC entry

epc_df["HP_INSTALLED"] = np.where(
    (no_epc_but_mcs_hp & mcs_before_epc_entry), False, epc_df["HP_INSTALLED"]
)

epc_df["HP_INSTALL_DATE"] = np.where(
    (no_epc_but_mcs_hp & mcs_before_epc_entry), np.nan, epc_df["HP_INSTALL_DATE"]
)

# ---
# MCS but EPC HP with MCS after first EPC entry

epc_df["HP_INSTALLED"] = np.where(
    (no_epc_but_mcs_hp & epc_entry_before_mcs), False, epc_df["HP_INSTALLED"]
)

epc_df["HP_INSTALL_DATE"] = np.where(
    (no_epc_but_mcs_hp & epc_entry_before_mcs), np.nan, epc_df["HP_INSTALL_DATE"]
)

# %%
no_future_hp_entry = epc_df[
    no_epc_but_mcs_hp & epc_entry_before_mcs & (epc_df["HAS_HP_AT_SOME_POINT"] == False)
]

no_future_hp_entry["HP_INSTALLED"] = True

print(epc_df.shape)
print(no_future_hp_entry.shape)
epc_df = pd.concat([epc_df, no_future_hp_entry])
print(epc_df.shape)

# %%
print(epc_df.shape)

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
    "ADDRESS2",
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
    # "MAIN_FUEL",
    #  "HEATING_SYSTEM",
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
# epc_df_encoded.to_csv(str(PROJECT_DIR)+'/outputs/epc_encoded_reduced.csv')
epc_df_encoded = pd.read_csv(
    str(PROJECT_DIR) + "/outputs/epc_df_5000000_preprocessed.csv"
)

# %%
epc_df_encoded.shape

# %%
drop_features = [
    "MCS_AVAILABLE",
    "Unnamed: 0",
    "Unnamed: 0.1",
    "HAS_HP_AT_SOME_POINT",
    "HP_INSTALL_DATE",
    "FIRST_HP_MENTION",
    "INSPECTION_YEAR",
    "N_ENTRIES_BUILD_ID",
    "HP_INSTALL_YEAR",
    "FIRST_HP_MENTION_YEAR",  #'HEATING_SYSTEM', 'HEATING_FUEL
]

epc_df_encoded.drop(columns=drop_features, inplace=True)

# %%
epc_df_encoded.columns

# %%
POSTCODE_LEVEL = "POSTCODE_SECTOR"
POSTCODE_LEVEL = "POSTCODE_UNIT"
all_postcode_levels = [
    "POSTCODE_AREA",
    "POSTCODE_DISTRICT",
    "POSTCODE_SECTOR",
    "POSTCODE_UNIT",
    "POSTCODE",
]
other_postcodes = [
    postcode for postcode in all_postcode_levels if postcode != POSTCODE_LEVEL
]
remove_cols = []
for feat in other_postcodes:
    remove_cols += [col for col in epc_df_encoded.columns if feat == col]

print(remove_cols)

epc_df_encoded = epc_df_encoded.drop(columns=remove_cols)


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
categorical_features


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

t = 2015
t_plus_one = 2020

data_time_t = feature_engineering.filter_by_year(
    epc_df_encoded, "BUILDING_ID", t, selection="latest entry", up_to=True
)
data_time_t_plus_one = feature_engineering.filter_by_year(
    epc_df_encoded, "BUILDING_ID", t_plus_one, selection="latest entry", up_to=True
)


# %%
future = feature_engineering.filter_by_year(
    epc_df_encoded, "BUILDING_ID", 2021, selection="latest entry", up_to=True
)
data_time_t = future

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

most_frequent = [col for col in cat_agglomerated.columns if "MOST_FREQUENT" in col]

print(cat_agglomerated.shape)
cat_agglomerated = feature_encoding.one_hot_encoding(
    cat_agglomerated, most_frequent, verbose=True
)
print(cat_agglomerated.shape)
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
print(agglomerated_with_target.shape)
agglomerated_with_target = agglomerated_with_target[
    ~agglomerated_with_target[POSTCODE_LEVEL].isin(samples_to_discard)
]
print(agglomerated_with_target.shape)

# %%

# %%
agglomerated_with_target.head()

# %%
remove_cols = [col for col in agglomerated_with_target.columns if "False" in col]
agglomerated_with_target = agglomerated_with_target.drop(columns=remove_cols)
print(agglomerated_with_target.columns)

# %%
for x in agglomerated_with_target.columns:
    print(x)

# %%
target_variables = ["GROWTH", "HP_COVERAGE_FUTURE"]
TARGET_VARIABLE = target_variables[0]

X = agglomerated_with_target.copy()  # int(X.shape[0]/2)]
y = np.array(X[TARGET_VARIABLE])

for col in target_variables + [POSTCODE_LEVEL]:
    if col in X.columns:
        del X[col]

print(X.shape)
X = X.dropna(axis="columns", how="all")
print(X.shape)
# X.fillna(X.mean(), inplace=True)

# %%
feature_names = X.columns

# %%
feature_names

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

prepr_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
        ("pca", PCA(n_components=27, random_state=42)),
    ]
)

# %%
X_prep = prepr_pipeline.fit_transform(X)
print(X_prep.shape)

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
    X_prep, y, test_size=0.1, random_state=42
)

# %%
train_postcode = X_train["POSTCODE_UNIT"]

# %%
list(train_postcode)

# %%
FIGPATH = str(PROJECT_DIR) + "/outputs/figures/"

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print()

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

model_dict = {
    #  "SVM Regressor": svm.SVR(),
    "Random Forest Regressor": RandomForestRegressor(),
    #  "Linear Regression": LinearRegression(),
    #   "Decision Tree Regressor": DecisionTreeRegressor(),
}
cv = 1
interval = 5

best_params = {
    # "SVM Regressor": {"C": 5, "gamma": 0.01, "kernel": "rbf"},
    "Linear Regression": {},
    "Decision Tree Regressor":
    # {'max_depth': 5, 'max_features': None,'max_leaf_nodes': 10, 'min_samples_leaf': 7,'min_weight_fraction_leaf': 0.05, 'splitter': 'random'},
    {
        "max_depth": 15,
        "max_features": "auto",
        "max_leaf_nodes": 15,
        "min_samples_leaf": 0.05,
        "min_weight_fraction_leaf": 0.05,
        "splitter": "best",
    },
    "Random Forest Regressor":  # {"max_features": 12, "n_estimators": 10, "min_samples_leaf" : 0.05},
    # {'bootstrap': False, 'max_features': 10, 'min_samples_leaf': 0.05,  'n_estimators': 20}
    {"max_features": 12, "n_estimators": 30},
}


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


def train_and_evaluate(model_name):

    model = model_dict[model_name]
    # model.set_params(**best_params[model_name])
    model.fit(X_train, y_train)

    return model

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

            # scores = cross_val_score(
            #     model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
            # )
            # rsme_scores = np.sqrt(-scores)
            # display_scores(rsme_scores)

            # errors = abs(sols-preds)
            # plt.scatter(errors, sols)
            # plt.title("Error: {} using {} on {}".format(variable_name, model_name, set_name))
            # plt.xlabel("Error")
            # plt.ylabel("Ground Truth")
            # plt.show()

            if model_name == "Decision Tree Regressor" and set_name == "Training Set":
                from sklearn import tree

                tree.plot_tree(model, feature_names=feature_names, label="all")
                plt.tight_layout()
                plt.savefig(FIGPATH + "decision_tree.png", dpi=300, bbox_inches="tight")
                plt.show()


for model in model_dict.keys():
    model = train_and_evaluate(model)

# %%
# perc_predictions =  model.predict(X_prep)
# np.save('perc_preds.npy', perc_predictions)

# %%
growth_predictions = model.predict(X_prep)
np.save("growth_preds.npy", growth_predictions)

# %%
future_growth = model.predict(X_prep)


# %%
target_var_df["GROWTH"].mean() * 100

# %%
future_growth.mean() * 100

# %%
agglomerated_with_target["coverage prediction"] = perc_predictions

# %%
full_set = agglomerated_with_target.copy()
full_set["growth prediction"] = growth_predictions

# %%
full_set.to_csv(str(PROJECT_DIR) + "/outputs/data_and_predictions_x.csv")

# %%
full_set["train_set"] = full_set["POSTCODE_UNIT"].isin(list(train_postcode))

# %%
full_set.head()

# %%
full_set = pd.merge(
    full_set,
    target_var_df[["# Properties", POSTCODE_LEVEL]],
    on=POSTCODE_LEVEL,
)

# %%
full_set.columns

# %%
full_set.head()

# %%
growth_predictions

# %%
full_set["growth prediction"] = growth_predictions

# %%

# %%
errors = abs(sols - preds)

# %%
agglomerated_with_target.columns

# %%
agglomerated_with_target["GROWTH"].head()

# %%
agglomerated_with_target.shape

# %%
agglomerated_with_target["growth predictions"] = growth_predictions

# %%
agglomerated_with_target.head()

# %%
agglomerated_with_target.to_csv(str(PROJECT_DIR) + "/outputs/X.csv")

# %%
from sklearn.model_selection import GridSearchCV

param_grid_dict = {
    "Random Forest Regressor": {
        "n_estimators": [10, 15, 20, 25],
        "max_features": [5, 10, 15, 20, 25, "auto", "sqrt"],
        "min_samples_leaf": [0.05],
        "bootstrap": [False],
    },  # {'bootstrap': False, 'max_features': 10, 'min_samples_leaf': 0.05, 'n_estimators': 20}
    "SVM Regressor": {
        "gamma": [0.0001, 0.001, 0.01, 0.1, 1.0, 10],
        "kernel": ["rbf"],
        "C": [0.001, 0.01, 0.1, 1, 2, 5, 10, 15, 20],
    },
    "Decision Tree Regressor": {
        "splitter": ["best", "random"],
        "max_depth": [1, 5, 10, 15],
        "min_samples_leaf": [0.05],
        "min_weight_fraction_leaf": [0.05, 0.1, 0.3, 0.5],
        "max_features": ["auto", "log2", "sqrt", None],
        "max_leaf_nodes": [None, 5, 10, 15, 20, 30, 50],
        # {'max_depth': 15, 'max_features': 'auto', 'max_leaf_nodes': 15,
        #'min_samples_leaf': 0.05, 'min_weight_fraction_leaf': 0.05, 'splitter': 'best'}
    },
}


def parameter_screening(model_name, X, y):

    model = model_dict[model_name]
    param_grid = param_grid_dict[model_name]

    grid_search = GridSearchCV(
        model, param_grid, cv=3, scoring="neg_mean_squared_error"
    )
    grid_search.fit(X, y)
    print(grid_search.best_params_)


para_screening = True

if para_screening:

    for model in model_dict.keys():
        if model not in ["Linear Regression", "Decision Tree Regressor"]:
            print(model)
            parameter_screening(model, X_train, y_train)
            print()

# %%

para_screening = True

if para_screening:

    for model in model_dict.keys():
        if model in ["Decision Tree Regressor"]:
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
