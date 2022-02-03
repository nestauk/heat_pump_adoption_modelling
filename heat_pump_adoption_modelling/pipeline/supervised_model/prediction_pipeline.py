from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
from heat_pump_adoption_modelling.pipeline.preprocessing import feature_engineering
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import (
    plotting_utils,
    model_settings,
)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib.pyplot import ylim

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import tree

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re


# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]
SUPERVISED_MODEL_OUTPUT = str(PROJECT_DIR) + config["SUPERVISED_MODEL_OUTPUT"]


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


def get_HP_status_changes(df):
    """For properties with multiple entries, get heat pump status at first
    and last entry time and determine whether a HP was added.

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

    # Whether heat pump was added in the meantime
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


def get_data_with_labels(
    df, version="Future HP Status", drop_features=[], balanced_set=True
):
    """Get the training data and labels (X, y) for training the models.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with features.

    version : str, default='Future HP Status'
        Options: Future HP Status, Current HP Status, HP Growth by Area

    drop_features : list, default=[]
        Features to discard.
        Target variable(s) are added to this list.

    balanced_set : bool, default=True
        Balance the training set with HP/no HP samples.
        Is set to False for version 'HP Growth by Area'.

    Return
    ---------
    X : pandas.Dataframe
        Feature data.

    y : pandas.Dataframe
        Labels / ground truth."""

    settings_dict = {
        "Current HP Status": [["HP_INSTALLED"], balanced_set],
        "Future HP Status": [["HP_ADDED"], balanced_set],
        "HP Growth by Area": [["GROWTH", "HP_COVERAGE_FUTURE"], False],
    }

    # Get settings based on version
    target_variable, balanced_set = settings_dict[version]

    # Shuffle data
    X = df.copy().sample(frac=1)

    # For the current status, get latest EPC entry only
    if version == "Current HP Status":

        X = feature_engineering.filter_by_year(
            df, "BUILDING_ID", None, selection="latest entry"
        )
        drop_features += [
            "HEATING_SYSTEM",
            "MAINHEAT_DESCRIPTION",
            "HP_INSTALL_DATE",
            "INSPECTION_DATE_AS_NUM",
        ] + target_variable

    # For future status, use changed heat pump status data
    elif version == "Future HP Status":

        X = get_HP_status_changes(df)
        drop_features += ["HP_INSTALL_DATE", "INSPECTION_DATE_AS_NUM"] + target_variable

    elif version == "HP Growth by Area":
        drop_features += ["HP_INSTALL_DATE"] + target_variable

    else:
        raise IOError("Version '{}' is unknown.".format(version))

    # Balance the feature set
    if balanced_set:
        X = balance_set(X, target_variable[0])

    # Reset index and select the label / ground truth
    X.reset_index(inplace=True, drop=True)
    y = X[target_variable]

    # Remove unneccesary features
    X.drop(columns=drop_features, errors="ignore", inplace=True)

    return X, y


def predict_heat_pump_adoption(X, y, version, save_predictions=False):
    """Predict the heat pump adoption.
    Depending on the settings, predict the current or future heat pump status of a property
    or the growth and heat pump coverage of a postcode area.

    Parameters
    ----------
    X: pandas.DataFrame
        Feature data.

    y: pandas.DataFrame
        Labels / ground truth.

    version : str, default='Future HP Status'
        Options: Future HP Status, Current HP Status, HP Growth by Area

    save_predictions : boolean, default=False
        Save the predictions, errors and other information for error analysis.

    Return
    ---------
    trained_models : dict
        All the trained models."""

    settings_dict = {
        "Current HP Status": (
            "household_based_curr_predictions_with_{}.csv",
            y,
            ["HP_INSTALLED"],
            model_settings.household_level_model_dict,
        ),
        "Future HP Status": (
            "household_based_future_predictions_with_{}.csv",
            y,
            ["HP_ADDED"],
            model_settings.household_level_model_dict,
        ),
        "HP Growth by Area": (
            "postcode_based_predictions_with_{}.csv",
            None,
            ["GROWTH", "HP_COVERAGE_FUTURE"],
            model_settings.postcode_level_model_dict,
        ),
    }

    # Get settings for version
    outfile_name, stratify, target_variables, model_dict = settings_dict[version]

    # Save original training data (with all columns, e.g. POSTCODE, target variables)
    if save_predictions:
        original_df = X.copy()

    # Remove unnecessary columns
    X.drop(
        columns=[col for col in X.columns if col.startswith("POSTCODE")], inplace=True
    )

    # Print the number of samples and features
    print("Predicting {}...\n".format(version))
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print()

    trained_models = {}
    feature_names = X.columns

    # Apply preprocesisng pipeline
    X_prep = model_settings.prepr_pipeline.fit_transform(X)

    # Split into train and test sets, keep indices
    X_train, X_test, y_train, y_test, indices_train, _ = train_test_split(
        X_prep, y, list(X.index), test_size=0.1, random_state=42, stratify=stratify
    )

    # Mark training samples
    X["training set"] = False
    X.loc[indices_train, "training set"] = True

    if save_predictions:
        original_df["training set"] = False
        original_df.loc[indices_train, "training set"] = True

    # For each model train, make predictions and evaluate
    for model_name in model_dict.keys():

        # For each model, get a fresh copy of the original df
        if save_predictions:
            data_with_label_and_pred = original_df.copy()

        # Evaluate each target variable
        for target in target_variables:

            # Get respective labels / ground truths
            y_train_target = np.array(y_train[target])
            y_test_target = np.array(y_test[target])

            # Get model and parameters
            model = model_dict[model_name]

            if model in model_settings.best_params.keys():
                model.set_params(**model_settings.best_params[model_name])

            # Train and evluate the model
            trained_model = train_and_evaluate(
                model_name,
                model,
                X_train,
                y_train_target,
                X_test,
                y_test_target,
                target,
                feature_names,
            )

            # Save model
            trained_models[model_name] = trained_model

            # Get predictions and probabilities
            if save_predictions:
                predictions, probabilities = get_predictions_and_probs(
                    model_name, trained_model, X_prep
                )

                data_with_label_and_pred = fill_in_predictions_and_probs(
                    data_with_label_and_pred, y, target, predictions, probabilities
                )

        # Write out predictions
        if save_predictions:

            # Filename including model name
            output_filename = outfile_name.format(re.sub(" ", "_", model_name).lower())

            print(
                "Output saved to {}".format(SUPERVISED_MODEL_OUTPUT + output_filename)
            )

            data_with_label_and_pred.to_csv(
                SUPERVISED_MODEL_OUTPUT + output_filename, index=False
            )

    return trained_models


def get_predictions_and_probs(model_name, trained_model, X_prep):
    """Get the predictions and probabilities for model.

    Parameters
    ----------
    model_name: str
        Model name.

    trained_model: sklearn.model
        Trained supervised model for HP prediction.

    X_prep : np.array
        Preprocessed feature set.

    Return
    ----------
    predictions: np.array
        Predictions for X_prep using given model.

    probabilities: np.array
        Probabilities for prediction classes.
        Not available for every model."""

    predictions = trained_model.predict(X_prep)

    # Get probabilities for target classes
    if model_name in ["Logistic Regression", "Support Vector Classifier"]:

        probabilities = trained_model.predict_proba(X_prep)
        proba_0_1 = (probabilities[:, 0], probabilities[:, 1])

    else:
        proba_0_1 = None

    return predictions, proba_0_1


def fill_in_predictions_and_probs(
    data_with_label_and_pred, y, target, predictions, probabilities=None
):
    """Fill in ground truth, prediction, error and probabilities.

    Parameters
    ----------
    data_with_label_and_pred : pandas.DataFrame
        Dateframe containing feature set.

    y: pandas.DataFrame
        Labels / ground truths.

    target : str
        Target variable, e.g. 'GROWTH' or 'HP_ADDED'.

    predictions : np.array
        Model predictions on feature set.

    probabilities : np.array, default=None
        If not None, probabilities for two target classes are added,

    Return
    ----------
    predictions: np.array
        Predictions for X_prep using given model.

    probabilities: np.array
        Probabilities for prediction classes.
        Not available for every model."""

    if probabilities is not None:
        proba_0, proba_1 = probabilities
        data_with_label_and_pred["proba 0"] = proba_0
        data_with_label_and_pred["proba 1"] = proba_1

    data_with_label_and_pred[target] = np.array(y[target])
    data_with_label_and_pred[target + ": prediction"] = predictions

    if target in ["HP_ADDED", "HP_INSTALLED"]:

        data_with_label_and_pred[target + ": error"] = (
            data_with_label_and_pred[target + ": prediction"]
            != data_with_label_and_pred[target]
        )

    else:
        data_with_label_and_pred[target + ": error"] = abs(
            data_with_label_and_pred[target + ": prediction"]
            - data_with_label_and_pred[target]
        )

    return data_with_label_and_pred


def train_and_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    target_variable,
    feature_names,
    perc_interval=5,
    cv=5,
):

    """
    Train and evaluate growth prediction model.

    Parameters
    ----------
    model_name: str
        Model name.

    model: sklearn.model
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
        e.g. HP_ADDED, HP_INSTALLED, GROWTH, HP_COVERAGE_FUTURE

    feature_names: list
        Feature names for training features.

    perc_interval : int, default=5
        Percentage intervals for creating confusion matrix
        and estimate accuracy.

    cv : int, default=5
        Number of cross validations, by default 5-fold.

    Return
    ---------
    model : sklearn model
        Trained model.
    """

    level = (
        "postcode level"
        if target_variable in ["GROWTH", "HP_COVERAGE_FUTURE"]
        else "household level"
    )

    score_dict = {
        "postcode level": ("neg_mean_squared_error", "max_error"),
        "household level": ("accuracy", "f1", "recall", "precision"),
    }

    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("\n{}\n{}\n{}".format(30 * "*", model_name, 30 * "*"))

    for preds, sols, set_name in [
        (pred_train, y_train, "Training Set"),
        (pred_test, y_test, "Test Set"),
    ]:

        if level == "household level":

            # Print Accuracies
            acc = np.round(accuracy_score(sols, preds) * 100, 2)
            print("\nAccuracy {}: {}%".format(set_name, acc))

            # Plot the confusion matrix for training set
            plotting_utils.plot_confusion_matrix(
                sols,
                preds,
                ["No HP", "HP"],
                title="Confusion Matrix:\n {} on {}".format(model_name, set_name),
            )

        else:

            # Fix below zero and above one values
            preds[preds < 0] = 0.0
            preds[preds > 1.0] = 1.0

            # Map prediction to percentage bin
            predictions, solutions, label_dict = plotting_utils.map_percentage_to_bin(
                preds, sols, interval=perc_interval
            )

            # How many predictions are correct according to percentage bin
            overlap = round((predictions == solutions).sum() / predictions.shape[0], 2)
            print(
                "Category Accuracy with {}% steps : {}".format(perc_interval, overlap)
            )

            # Plot the confusion matrix for training set
            plotting_utils.plot_confusion_matrix(
                solutions,
                predictions,
                [label_dict[label] for label in sorted(set(solutions))],
                title="Confusion Matrix:\n{} using {} on {}".format(
                    target_variable, model_name, set_name
                ),
            )
    scores = cross_validate(
        model,
        X_train,
        y_train,
        scoring=score_dict[level],
        return_train_score=True,
        cv=cv,
    )

    if level == "postcode level":
        scores["train_rsme"] = np.sqrt(-scores["train_neg_mean_squared_error"])
        scores["test_rsme"] = np.sqrt(-scores["test_neg_mean_squared_error"])

    plotting_utils.display_scores(scores, cv=cv)

    if level == "postcode level":

        for preds, sols, set_name in [
            (pred_train, y_train, "Training Set"),
            (pred_test, y_test, "Test Set"),
        ]:

            print("Error plots for", set_name)

            # Plot prediction by ground truth
            plotting_utils.scatter_plot(
                preds,
                sols,
                "{} using {} on {}".format(target_variable, model_name, set_name),
                "Prediction",
                "Ground Truth",
            )

            errors = abs(sols - preds)
            plotting_utils.scatter_plot(
                errors,
                sols,
                "Error: {} using {} on {}".format(
                    target_variable, model_name, set_name
                ),
                "Error",
                "Ground Truth",
            )

            # Print and save the decision tree
            if model_name == "Decision Tree Regressor" and set_name == "Training Set":

                tree.plot_tree(model, feature_names=feature_names, label="all")
                plt.tight_layout()
                plt.savefig(
                    FIG_PATH + "decision_tree.png", dpi=300, bbox_inches="tight"
                )
                plt.show()

    return model


def coefficient_importance(X, y, model_name, version="Future HP Status", pca=False):
    """Plot the coefficient importance for features used to train model.

    Parameters
    ----------
    X: pandas.DataFrame
        Training data.

    y: pandas.DataFrame
        Labels / ground truth.

    model_name : str
        Model name

    version : str, default='Future HP Status'
        Target variable.
        Options: Future HP Status, Current HP Status

    pca : bool, default=False
        Whether or not to use PCA on features before training.

    Return: None"""

    target = "HP_INSTALLED" if version == "Current HP Status" else "HP_ADDED"
    y = y[target]

    # Preprocessing pipeline with or without PCA
    if pca:
        X_prep = model_settings.prepr_pipeline.fit_transform(X)
        pca_tag = "using PCA"
    else:
        X_prep = model_settings.prepr_pipeline_no_pca.fit_transform(X)
        pca_tag = ""

    # Split into train and test sets
    X_train, _, y_train, _ = train_test_split(
        X_prep, y, test_size=0.1, random_state=42, stratify=y
    )

    # Train model
    model = model_settings.household_level_model_dict[model_name]

    if model_name in model_settings.best_params.keys():
        model.set_params(**model_settings.best_params[model_name])

    model.fit(X_train, y_train)

    # Evaluation
    acc = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    f1 = cross_val_score(model, X_train, y_train, cv=5, scoring="f1")

    print("5-fold Cross Validation\n---------\n")
    print("Accuracy:", round(acc.mean(), 2))
    print("F1 Score:", round(f1.mean(), 2))

    # Get feature names
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

    # Plot most important features
    plotting_utils.get_most_important_coefficients(
        model,
        feature_names,
        "{}: Coefficient Importance {}".format(version, pca_tag),
        X_prep,
    )
