from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


from sklearn.ensemble import RandomForestRegressor

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import SGDClassifier
# from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn import svm

import numpy as np

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

household_level_model_dict = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear Support Vector Classifier": SGDClassifier(random_state=42),
    # "Support Vector Classifier": svm.SVC(probability=True, random_state=42),
}


postcode_level_model_dict = {
    "Random Forest Regressor": RandomForestRegressor(),
    #  "SVM Regressor": svm.SVR(),
    #  "Linear Regression": LinearRegression(),
    #   "Decision Tree Regressor": DecisionTreeRegressor(),
}
