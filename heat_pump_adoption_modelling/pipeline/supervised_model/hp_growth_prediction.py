# File: heat_pump_adoption_modelling/pipeline/supervised_model/hp_growth_prediction.py
"""
Predict the HP growth of an area using a supervised learning model.
"""

# ----------------------------------------------------------------------------------

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
from heat_pump_adoption_modelling.pipeline.supervised_model.utils import plotting_utils

import re

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.model_selection import cross_val_score
from sklearn import tree

import matplotlib.pyplot as plt


import numpy as np

# ----------------------------------------------------------------------------------

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]
SUPERVISED_MODEL_OUTPUT = str(PROJECT_DIR) + config["SUPERVISED_MODEL_OUTPUT"]

model_dict = {
    "Random Forest Regressor": RandomForestRegressor(),
    #  "SVM Regressor": svm.SVR(),
    #  "Linear Regression": LinearRegression(),
    #   "Decision Tree Regressor": DecisionTreeRegressor(),
}

prepr_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9, random_state=42)),
    ]
)


def train_and_evaluate(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    target_variable,
    feature_names,
    perc_interval=5,
    cv=5,
):
    """Train and evaluate growth prediction model.

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
        e.g. HP_COVERAGE_FUTURE, GROWTH

    feature_names: list
        Feature names for training features.

    perc_interval : int, default=5
        Percentage intervals for creating confusion matrix
        and estimate accuracy.

    cv : int, default=5
        Cross validation, by default 5-fold.

    Return
    ---------
    model : sklearn model
        Trained model."""

    model = model_dict[model_name]
    # model.set_params(**best_params[model_name])

    model.fit(X_train, y_train)

    # Predict training and test set
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("\n*****************\nModel Name: {}\n*****************".format(model_name))

    # Set the full target variable name
    variable_name = (
        "Target Year HP Coverage"
        if target_variable == "HP_COVERAGE_FUTURE"
        else "Growth"
    )

    # For each set, evaluate performance
    for set_name in ["train", "test"]:

        # Get training or test data
        if set_name == "train":
            preds = pred_train
            sols = y_train
            set_name = "Training Set"

        elif set_name == "test":
            preds = pred_test
            sols = y_test
            set_name = "Test Set"

        print("\n-----------------\n{}\n-----------------".format(set_name))
        print()

        # Fix below zero and above one values
        preds[preds < 0] = 0.0
        preds[preds > 1.0] = 1.0

        # Map prediction to percentage bin
        predictions, solutions, label_dict = plotting_utils.map_percentage_to_bin(
            preds, sols, interval=perc_interval
        )

        # How many predictions are correct according to percentage bin
        overlap = round((predictions == solutions).sum() / predictions.shape[0], 2)
        print("Category Accuracy with {}% steps : {}".format(perc_interval, overlap))

        # Plot the confusion matrix for training set
        plotting_utils.plot_confusion_matrix(
            solutions,
            predictions,
            [label_dict[label] for label in sorted(set(solutions))],
            title="Confusion Matrix:\n{} using {} on {}".format(
                variable_name, model_name, set_name
            ),
        )

        # Plot prediction by ground truth
        plotting_utils.scatter_plot(
            preds,
            sols,
            "{} using {} on {}".format(variable_name, model_name, set_name),
            "Prediction",
            "Ground Truth",
        )

        # Get, print and plot the scores
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="neg_mean_squared_error"
        )
        rsme_scores = np.sqrt(-scores)
        plotting_utils.display_scores(rsme_scores)

        errors = abs(sols - preds)
        plotting_utils.scatter_plot(
            errors,
            sols,
            "Error: {} using {} on {}".format(variable_name, model_name, set_name),
            "Error",
            "Ground Truth",
        )

        # Print and save the decision tree
        if model_name == "Decision Tree Regressor" and set_name == "Training Set":

            tree.plot_tree(model, feature_names=feature_names, label="all")
            plt.tight_layout()
            plt.savefig(FIG_PATH + "decision_tree.png", dpi=300, bbox_inches="tight")
            plt.show()

    return model


def get_data_with_labels(df, target_variables, drop_features=[]):
    """Get the training data and labels (X, y) for training the models.

    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe with features.

    target_variables: list
        List of target variables.

    drop_features : list, default=[]
        Features to discard.

    Return
    ---------
    X : pandas.Dataframe
        Training data.

    y : pandas.Dataframe
        Labels / ground truth."""

    # Get the
    X = df.copy()
    y = X[target_variables]  # np.array(X[target_variables])

    # Remove unnecessary features
    for col in target_variables + drop_features:
        if col in X.columns:
            del X[col]

    # Just in case, remove features with all NaN values
    X = X.dropna(axis="columns", how="all")

    return X, y


def predict_hp_growth_for_area(
    X,
    y,
    target_variables=["GROWTH", "HP_COVERAGE_FUTURE"],
    save_predictions=False,
):
    """Predict the heat pump growth for area.

    Parameters
    ----------
    X: pandas.DataFrame
        Training data.

    y: pandas.DataFrame
        Labels / ground truth.

    target_variables : list, default="GROWTH", "HP_COVERAGE_FUTURE"]
        Target variables to predict.

    save_predictions : boolean, default=False
        Save the predictions, errors and other information for error analysis.

    Return: None"""

    # Reset indices and create new index row
    X.reset_index(drop=True, inplace=True)
    indices = np.arange(X.shape[0])
    X["index"] = np.arange(X.shape[0])

    # Save original training data (with all columns, e.g. POSTCODE, target variables)
    if save_predictions:
        original_df = X.copy()

    # Remove unnecessary columns
    for feat in X.columns:
        if feat.startswith("POSTCODE"):
            del X[feat]

    # Get feature names
    feature_names = X.columns

    # Print the number of samples and features
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print()

    # Apply preprocesisng pipeline
    X_prep = prepr_pipeline.fit_transform(X)

    # Split into train and test sets
    (
        X_train,
        X_test,
        y_train,
        y_test,
        indices_train,
        indices_test,
    ) = train_test_split(X_prep, y, indices, test_size=0.1, random_state=42)

    # Mark training samples
    X.at[indices_train, "training set"] = True

    if save_predictions:
        original_df["training set"] = False
        original_df.at[indices_train, "training set"] = True

    # For each model and target variable, train, make predictions and evaluate
    for model in model_dict.keys():

        if save_predictions:
            data_with_label_and_pred = original_df.copy()

        for target in target_variables:

            # Get respective labels / ground truths
            y_train_target = np.array(y_train[target])
            y_test_target = np.array(y_test[target])

            # Train and evluate the model
            trained_model = train_and_evaluate(
                model,
                X_train,
                y_train_target,
                X_test,
                y_test_target,
                target,
                feature_names,
            )

            # Save predictions, errors and ground truths for later error analysis
            if save_predictions:

                predictions = trained_model.predict(X_prep)

                data_with_label_and_pred[target] = np.array(y[target])
                data_with_label_and_pred[target + ": prediction"] = predictions
                data_with_label_and_pred[target + ": error"] = abs(
                    data_with_label_and_pred[target + ": prediction"]
                    - data_with_label_and_pred[target]
                )

        if save_predictions:

            output_filename = "area_based_predictions_with_{}.csv".format(
                re.sub(" ", "_", model).lower()
            )

            print(
                "Output saved to {}".format(SUPERVISED_MODEL_OUTPUT + output_filename)
            )

            print(list(data_with_label_and_pred.columns))
            data_with_label_and_pred.to_csv(
                SUPERVISED_MODEL_OUTPUT + output_filename, index=False
            )
