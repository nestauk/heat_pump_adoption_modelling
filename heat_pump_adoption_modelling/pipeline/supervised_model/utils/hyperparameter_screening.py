from sklearn.model_selection import GridSearchCV

param_grid_dict = {
    "Linear Support Vector Classifier": {
        "penalty": ["l2", "l1", "elasticnet"],
        "alpha": [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1],
        "epsilon": [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        "tol": [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.01, 0.1],
    },
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


def hyperparameter_screening(model, model_name, X, y, scoring):
    """Screen hyper parameters for supervised learning models
    and print out best combiantion.

    Parameters
    ----------
    model : sklearn.model
        Model for which to screen hyperparameters.

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

    # Get model and parameter dictionary
    param_grid = param_grid_dict[model_name]

    # Apply grid search for finding the best parameters
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring=scoring)

    # Fit the model and find best parameters
    grid_search.fit(X, y)
    print(model_name)
    print(grid_search.best_params_)
