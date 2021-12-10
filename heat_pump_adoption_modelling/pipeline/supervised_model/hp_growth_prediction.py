from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
from sklearn.model_selection import GridSearchCV


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV


from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, TruncatedSVD

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn import tree

from heat_pump_adoption_modelling.pipeline.supervised_model import plotting_utils

import numpy as np
import pandas as pd

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]
SUPERVISED_MODEL_OUTPUT = PROJECT_DIR / config["SUPERVISED_MODEL_FIG_PATH"]

model_dict = {
    #  "SVM Regressor": svm.SVR(),
    "Random Forest Regressor": RandomForestRegressor(),
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

    model = model_dict[model_name]
    # model.set_params(**best_params[model_name])
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("\n*****************\nModel Name: {}\n*****************".format(model_name))

    variable_name = (
        "Target Year HP Coverage"
        if target_variable == "HP_COVERAGE_FUTURE"
        else "Growth"
    )

    for set_name in ["train", "test"]:

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

        preds[preds < 0] = 0.0
        preds[preds > 1.0] = 1.0

        predictions, solutions, label_dict = plotting_utils.map_percentage_to_bin(
            preds, sols, interval=perc_interval
        )

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

        plotting_utils.scatter_plot(
            preds,
            sols,
            "{} using {} on {}".format(variable_name, model_name, set_name),
            "Prediction",
            "Ground Truth",
        )

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

        if model_name == "Decision Tree Regressor" and set_name == "Training Set":

            tree.plot_tree(model, feature_names=feature_names, label="all")
            plt.tight_layout()
            plt.savefig(FIG_PATH + "decision_tree.png", dpi=300, bbox_inches="tight")
            plt.show()

    return model


def get_data_with_labels(df, target_variable, drop_features=[]):

    X = df.copy()
    y = np.array(X[target_variable])

    for col in [target_variable] + drop_features:
        if col in X.columns:
            del X[col]

    print(X.shape)
    X = X.dropna(axis="columns", how="all")
    print(X.shape)

    return X, y


def predict_hp_growth_for_area(X, y, target_variable="GROWTH", save_predictions=False):

    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])
    print()

    feature_names = X.columns

    indices = np.arange(X.shape[0])

    X_prep = prepr_pipeline.fit_transform(X)
    # Split into train and test sets
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        X_prep, y, indices, test_size=0.1, random_state=42
    )

    for model in model_dict.keys():
        trained_model = train_and_evaluate(
            model, X_train, y_train, X_test, y_test, target_variable, feature_names
        )

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

            output_filename = "{}_predictions_with_{}.csv".format(
                target_variable, model
            )

            print("Output saved {}".format(output_filename))

            pd.to_csv(SUPERVISED_MODEL_OUTPUT + output_filename)


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
    print(model_name)
    print(grid_search.best_params_)
