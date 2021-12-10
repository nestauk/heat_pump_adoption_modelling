from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering
from heat_pump_adoption_modelling.pipeline.supervised_model import plotting_utils

import pandas as pd
import numpy as np
from sklearn import preprocessing

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]
SUPERVISED_MODEL_OUTPUT = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]


prepr_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9, random_state=42)),
    ]
)

prepr_pipeline_no_pca = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
    ]
)


param_grid_dict = {
    "Linear Support Vector Classifier": {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "epsilon": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        "tol": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1],
    }
}

best_params = {
    "Linear Support Vector Classifier":  # { "alpha": 0.0001,"epsilon": 0.0001,
    # "penalty": "l2","tol": 0.001,}
    {
        "alpha": 0.0001,
        "epsilon": 0.0001,
        "penalty": "elasticnet",
        "tol": 0.0005,
        "max_iter": 5000,
    }
}

model_dict = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear Support Vector Classifier": SGDClassifier(random_state=42),
}


def get_HP_status_changes(df):

    multiple_entry_epcs = df.loc[df["N_ENTRIES_BUILD_ID"] >= 2]

    latest_epc = feature_engineering.filter_by_year(
        multiple_entry_epcs, "BUILDING_ID", None, selection="latest entry"
    )
    first_epc = feature_engineering.filter_by_year(
        multiple_entry_epcs, "BUILDING_ID", None, selection="first entry"
    )

    latest_epc["NOW_HP"] = latest_epc["HP_INSTALLED"]
    first_epc["PAST_HP"] = first_epc["HP_INSTALLED"]

    before_after_status = pd.merge(
        latest_epc[["NOW_HP", "BUILDING_ID"]],
        first_epc[["PAST_HP", "BUILDING_ID"]],
        on=["BUILDING_ID"],
    )

    before_after_status["HP_ADDED"] = (before_after_status["NOW_HP"] == True) & (
        before_after_status["PAST_HP"] == False
    )

    before_after_status["HP_REMOVED"] = (before_after_status["NOW_HP"] == False) & (
        before_after_status["PAST_HP"] == True
    )

    before_after_status["ALWAYS_HP"] = (before_after_status["NOW_HP"] == True) & (
        before_after_status["PAST_HP"] == True
    )

    before_after_status["NEVER_HP"] = (before_after_status["NOW_HP"] == False) & (
        before_after_status["PAST_HP"] == False
    )

    del first_epc["PAST_HP"]
    future_hp_status = pd.merge(first_epc, before_after_status, on=["BUILDING_ID"])

    future_hp_status = future_hp_status.loc[(future_hp_status["PAST_HP"] == False)]

    future_hp_status = future_hp_status.drop(
        columns=["PAST_HP", "NOW_HP", "HP_REMOVED", "ALWAYS_HP", "NEVER_HP"]
    )

    return future_hp_status


def balance_set(X, target_variable, ratio=0.90):

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


def get_data_with_labels(
    df, version="Future HP Status", drop_features=[], balanced_set=True
):

    if version == "Current HP Status":

        X = feature_engineering.filter_by_year(
            df, "BUILDING_ID", None, selection="latest entry"
        )
        target_variable = "HP_INSTALLED"
        drop_features += ["HEATING_SYSTEM", "MAINHEAT_DESCRIPTION"]

    elif version == "Future HP Status":

        X = get_HP_status_changes(df)
        target_variable = "HP_ADDED"

    else:
        raise IOError("Version '{}' is unknown.".format(version))

    print("Before", X.shape)
    X = X.dropna(axis="columns", how="all")
    print("After", X.shape)

    if balanced_set:
        X = balance_set(X, target_variable)

    y = X[target_variable]

    for col in [target_variable] + drop_features:
        if col in X.columns:
            del X[col]

    return X, y


def train_and_evaluate(
    model_name, X_train, y_train, X_test, y_test, target_variable, feature_names, cv=5
):

    model = model_dict[model_name]

    if model in best_params.keys():
        model.set_params(**best_params[model_name])
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("Model Name:", model_name)

    # Plot the confusion matrix for training set
    plotting_utils.plot_confusion_matrix(
        y_train, pred_train, ["No HP", "HP"], "Training set"
    )

    # Plot the confusion matrix for validation set
    plotting_utils.plot_confusion_matrix(
        y_test, pred_test, ["No HP", "HP"], "Validation set"
    )

    # Print Accuracies
    print(
        "Accuracy train: {}%".format(
            np.round(accuracy_score(y_train, pred_train) * 100), 2
        )
    )
    print(
        "Accuracy test:   {}%".format(
            np.round(accuracy_score(y_test, pred_test) * 100), 2
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

    return model


def predict_heat_pump_status(
    X, y, target_variable="HP_Installed", save_predictions=False
):

    trained_models = {}

    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print()

    feature_names = X.columns
    indices = np.arange(X.shape[0])

    X_prep = prepr_pipeline.fit_transform(X)

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_prep, y, indices, test_size=0.1, random_state=42, stratify=y
    )

    for model in model_dict.keys():
        trained_model = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, "HP Installed", feature_names
        )

        trained_models[model] = trained_model

        if save_predictions:

            predictions = trained_model.predict(X_prep)

            data_with_label_and_pred = X.copy()
            data_with_label_and_pred[target_variable] = y
            data_with_label_and_pred["prediction"] = predictions

            data_with_label_and_pred["error"] = abs(
                data_with_label_and_pred["predictions"]
                - data_with_label_and_pred[target_variable]
            )

            data_with_label_and_pred.loc[indices_train, "training set"] = True
            data_with_label_and_pred.loc[indices_test, "training set"] = False

            pd.to_csv(
                SUPERVISED_MODEL_OUTPUT
                + "{}_predictions_with_{}.csv".format(target_variable, model)
            )

    return trained_models


def coefficient_importance(X, y, model_name, version="Future HP Status", pca=False):

    if pca:
        X_prep = prepr_pipeline.fit_transform(X)
        pca_tag = "using PCA"
    else:
        X_prep = prepr_pipeline_no_pca.fit_transform(X)
        pca_tag = ""

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_prep, y, test_size=0.1, random_state=42, stratify=y
    )

    model = model_dict[model_name]
    model.set_params(**best_params[model_name])
    model.fit(X_train, y_train)

    acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    print("5-fold Cross Validation\n---------\n")
    print("Accuracy:", round(acc.mean(), 2))
    print("F1 Score:", round(f1.mean(), 2))

    if pca:
        feature_names = np.array(["PC " + str(num) for num in range(X_train.shape[1])])
    else:
        feature_names = np.array(X.columns)

    # Plot the classifier's coefficients for each feature and label
    plotting_utils.plot_feature_coefficients(
        model,
        feature_names,
        ["HP Installed"],
        "{}: Coefficient Contributions {}".format(version, pca_tag),
    )

    plotting_utils.get_most_important_coefficients(
        model,
        feature_names,
        "{}: Coefficient Importance {}".format(version, pca_tag),
        X_prep,
    )


def parameter_screening(model_name, X, y):

    model = model_dict[model_name]

    if model in param_grid_dict.keys():
        param_grid = param_grid_dict[model_name]

        grid_search = GridSearchCV(model, param_grid, cv=3, scoring="f1")
        grid_search.fit(X, y)
        print(model_name)
        print(grid_search.best_params_)