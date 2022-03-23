# File: heat_pump_adoption_modelling/pipeline/supervised_model/hp_status_prediction.py
"""
Predict the HP status of a household using a supervised learning model.
"""

# ----------------------------------------------------------------------------------

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import plotting_utils

import pandas as pd
import numpy as np
from sklearn import svm

import re

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]
SUPERVISED_MODEL_OUTPUT = str(PROJECT_DIR) + config["SUPERVISED_MODEL_OUTPUT"]


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
    "Support Vector Classifier": svm.SVC(probability=True, random_state=42),
}


def get_HP_status_changes(df):
    """For properties with multiple entries, get heat pump status at first
    and last entry time.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe for not de-duplicated EPC data.

    Return
    ---------
    df_with_hp_status: pandas.DataFrame
        Dataframe with added future heat pump status."""

    # Get properties with multiple entries
    multiple_entry_epcs = df.loc[df["N_ENTRIES_BUILD_ID"] >= 2]

    # Get latest and first EPC entry
    latest_epc = feature_engineering.filter_by_year(
        multiple_entry_epcs, "BUILDING_ID", None, selection="latest entry"
    )
    first_epc = feature_engineering.filter_by_year(
        multiple_entry_epcs, "BUILDING_ID", None, selection="first entry"
    )

    # Heat pump status now / in the past
    latest_epc.rename(columns={"HP_INSTALLED": "NOW_HP"}, inplace=True)
    first_epc.rename(columns={"HP_INSTALLED": "PAST_HP"}, inplace=True)

    # Merging before/after
    before_after_status = pd.merge(
        latest_epc[["NOW_HP", "BUILDING_ID"]],
        first_epc[["PAST_HP", "BUILDING_ID"]],
        on=["BUILDING_ID"],
    )

    # Heat pump was added in the meantime
    before_after_status["HP_ADDED"] = (before_after_status["NOW_HP"]) & (
        ~before_after_status["PAST_HP"]
    )

    # Get future heat pump status
    df_with_hp_status = pd.merge(
        first_epc,
        before_after_status[["BUILDING_ID", "HP_ADDED"]],
        on=["BUILDING_ID"],
    )

    # Exclude samples that had a HP from the very start
    df_with_hp_status = df_with_hp_status.loc[(df_with_hp_status["PAST_HP"] == False)]

    # Drop unnecesary columns
    df_with_hp_status = df_with_hp_status.drop(columns=["PAST_HP"])

    return df_with_hp_status


def balance_set(X, target_variable, false_ratio=0.9):
    """Balance the training set.
    If false ratio set to 0.9, then 90% of the training data
    will have "False/No HP Installed" as a label.

    Parameters
    ----------
    X: pandas.DataFrame
        Training set.

    target_variable : str
        Variable/feature that is going to be predicted.

    false_ratio : float, default=0.9
        When re-balancing the set, use the false_ratio
        to determine the amount of False labels.

    Return
    ---------
    X: pandas.DataFrame
        Re-balanced training set."""

    multiplier = false_ratio / (1 - false_ratio)

    # Seperate samples with and without heat pumps
    X_true = X.loc[X[target_variable] == True]
    X_false = X.loc[X[target_variable] == False]

    # Shuffle and adjust size
    X_false = X_false.sample(frac=1)

    # Get the appropriate amount of "false" samples
    X_false = X_false[: int(X_true.shape[0] * multiplier)]

    # Concatenate "true" and "false" samples
    X = pd.concat([X_true, X_false], axis=0)

    # Reshuffle
    X = X.sample(frac=1)

    return X


def get_data_with_labels(
    df, version="Future HP Status", drop_features=[], balanced_set=True
):
    """Get the training data and labels (X, y) for training the models.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with features.

    version : str, default='Future HP Status'
        Target variable.
        Options: Future HP Status, Current HP Status

    drop_features : list, default=[]
        Features to discard.

    balanced_set : bool, default=True
        Balance the training set with HP/no HP samples.

    Return
    ---------
    X : pandas.Dataframe
        Training data.

    y : pandas.Dataframe
        Labels / ground truth."""

    df = df.sample(frac=1)

    # For the current status, get latest EPC entry only
    if version == "Current HP Status":

        X = feature_engineering.filter_by_year(
            df, "BUILDING_ID", None, selection="latest entry"
        )
        target_variable = "HP_INSTALLED"
        drop_features += ["HEATING_SYSTEM", "MAINHEAT_DESCRIPTION"]

    # For future status, use changed heat pump status data
    elif version == "Future HP Status":

        X = get_HP_status_changes(df)
        target_variable = "HP_ADDED"

    else:
        raise IOError("Version '{}' is unknown.".format(version))

    # Drop columns that are all NaN (not expected)
    X = X.dropna(axis="columns", how="all")

    # Balance the feature set
    if balanced_set:
        X = balance_set(X, target_variable)

    X.reset_index(inplace=True, drop=True)

    # Select the label / ground truth
    y = X[target_variable]

    # Remove unneccesary features, including target variables
    for col in [target_variable] + drop_features:
        if col in X.columns:
            del X[col]

    return X, y


def train_and_evaluate(
    model_name, X_train, y_train, X_test, y_test, target_variable, feature_names, cv=5
):
    """
    Train and evaluate growth prediction model.

    Parameters
    ----------
    model_name: str
        Model to train and evaluate.

    X_train: np.array
        Training data.

    y_train: np.array
        Solutions for training data.

    X_test: np.array
        Test data.

    y_test: np.array
        Solutions for test data.

    target_variable : str
        Target variable to predict,
        e.g. HP_ADDED, HP_INSTALLED

    feature_names: list
        Feature names for training features.

    cv : int, default=5
        Number of cross validations, by default 5-fold.

    Return
    ---------
    model : sklearn model
        Trained model.
    """

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

    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=("accuracy", "f1", "recall", "precision"),
        return_train_score=True,
        cv=cv,
    )

    print()
    print("{}-fold Cross Validation: Train \n---------\n".format(cv))
    print("Accuracy:", round(scores["train_accuracy"].mean(), 2))
    print("F1 Score:", round(scores["train_f1"].mean(), 2))
    print("Recall:", round(scores["train_recall"].mean(), 2))
    print("Precision:", round(scores["train_precision"].mean(), 2))
    print()
    print("{}-fold Cross Validation: Test \n---------\n".format(cv))
    print("Accuracy:", round(scores["test_accuracy"].mean(), 2))
    print("F1 Score:", round(scores["test_f1"].mean(), 2))
    print("Recall:", round(scores["test_recall"].mean(), 2))
    print("Precision:", round(scores["test_precision"].mean(), 2))
    print()
    print("-------------------------------")

    return model


def predict_heat_pump_status(
    X, y, target_variable="HP_Installed", save_predictions=False
):
    """Predict the heat pump status for household.

    Parameters
    ----------
    X: pandas.DataFrame
        Feature data.

    y: pandas.DataFrame
        Labels / ground truth.

    target_variables : str, default="HP_Installed"
        Target variables to predict.

    save_predictions : boolean, default=False
        Save the predictions, errors and other information for error analysis.

    Return: None"""

    # Save original training data (with all columns, e.g. POSTCODE, target variables)
    if save_predictions:
        original_df = X.copy()

    # Remove unnecessary columns
    for feat in X.columns:
        if feat.startswith("POSTCODE"):
            del X[feat]

    # Print the number of samples and features
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print()

    trained_models = {}
    feature_names = X.columns
    y = np.array(y)

    # Apply preprocesisng pipeline
    X_prep = prepr_pipeline.fit_transform(X)

    # Split into train and test sets
    X_train, X_test, y_train, y_test, indices_train, _ = train_test_split(
        X_prep, y, list(X.index), test_size=0.1, random_state=42, stratify=y
    )

    X["training set"] = False
    X.loc[indices_train, "training set"] = True

    # Mark training samples
    if save_predictions:
        original_df["training set"] = False
        original_df.loc[indices_train, "training set"] = True

    # For each model train, make predictions and evaluate
    for model in model_dict.keys():
        trained_model = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, "HP Installed", feature_names
        )

        trained_models[model] = trained_model

        if save_predictions:

            data_with_label_and_pred = original_df.copy()

            predictions = trained_model.predict(X_prep)

            # Get probabilities for target classes
            if model == "Logistic Regression":
                data_with_label_and_pred["proba 1"] = trained_model.predict_proba(
                    X_prep
                )[0][0]
                data_with_label_and_pred["proba 2"] = trained_model.predict_proba(
                    X_prep
                )[0][1]

            elif model == "Support Vector Classifier":

                data_with_label_and_pred["proba 1"] = trained_model.predict_proba(
                    X_prep
                )[:, 0]
                data_with_label_and_pred["proba 2"] = trained_model.predict_proba(
                    X_prep
                )[:, 1]
            else:
                pass

            # Save predictions, errors and ground truths for later error analysis
            data_with_label_and_pred["ground truth"] = y
            data_with_label_and_pred["prediction"] = predictions

            data_with_label_and_pred["error"] = (
                data_with_label_and_pred["prediction"]
                != data_with_label_and_pred["ground truth"]
            )

            output_filename = "household_based_predictions_with_{}.csv".format(
                re.sub(" ", "_", model).lower()
            )

            print(
                "Output saved to {}".format(SUPERVISED_MODEL_OUTPUT + output_filename)
            )
            data_with_label_and_pred.to_csv(
                SUPERVISED_MODEL_OUTPUT + output_filename, index=False
            )

    return trained_models
