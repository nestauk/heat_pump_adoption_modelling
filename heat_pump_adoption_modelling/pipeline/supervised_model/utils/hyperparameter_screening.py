# File: heat_pump_adoption_modelling/pipeline/supervised_model/utils/hyperparameter_screening.py
"""
Hyperparameter screening for supervised models.
"""

# ----------------------------------------------------------------------------------


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import numpy as np

prepr_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
        ("pca", PCA(n_components=0.9, random_state=42)),
    ]
)

model_dict = {
    "Random Forest Regressor": RandomForestRegressor(),
    #  "SVM Regressor": svm.SVR(),
    #  "Linear Regression": LinearRegression(),
    #   "Decision Tree Regressor": DecisionTreeRegressor(),
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear Support Vector Classifier": SGDClassifier(random_state=42),
    "Support Vector Classifier": svm.SVC(probability=True, random_state=42),
}


param_grid_dict = {
    "Linear Support Vector Classifier": {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "epsilon": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        "tol": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1],
    },
    "Random Forest Regressor": {
        "n_estimators": [10, 15, 18, 20, 23],
        "max_features": [5, 10, 15, 20, 25, "auto", "sqrt"],
        "min_samples_leaf": [0.05],
        "bootstrap": [False],
    },  # {'bootstrap': False, 'max_features': 10, 'min_samples_leaf': 0.05, 'n_estimators': 20}
    # {'bootstrap': False, 'max_features': 15, 'min_samples_leaf': 0.05, 'n_estimators': 15}
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


def grid_screening(model_name, X, y, scoring, drop_features):
    """Screen hyper parameters for supervised learning models
    and print out best combiantion.

    Parameters
    ----------
    model_name : str
        Supervised learning model to use.

    X : pandas.Dataframe
        Training data.

    y : pandas.DataFrame
        Label / ground truth.

    scoring : sklearn.Score
        Score or error by which to evaluate and optimise parameters.


    Return: None
    """

    for feat in X.columns:
        if feat in drop_features or feat.startswith("POSTCODE"):
            del X[feat]

    X_prep = prepr_pipeline.fit_transform(X)

    print(X_prep.shape)

    # Get model and parameter dictionary
    model = model_dict[model_name]
    param_grid = param_grid_dict[model_name]

    # Apply grid search for finding the best parameters
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring)

    # Fit the model and find best parameters
    grid_search.fit(X_prep, y)
    print(model_name)
    print(grid_search.best_params_)
