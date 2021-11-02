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

# %% [markdown]
# ## Imports and Data Loading

# %%
from heat_pump_adoption_modelling.getters import epc_data
from heat_pump_adoption_modelling import PROJECT_DIR

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

# Load all available columns
epc_df = epc_data.load_preprocessed_epc_data(version=version, usecols=None)
epc_df.columns


# %% [markdown]
# ## Categorical Feature Encoding

# %%
@interact(feature=epc_df.columns)
def value_counts(feature):
    print(epc_df[feature].value_counts())


# %%
encoded_features = epc_df.copy()

# Get all only numeric features
num_features = encoded_features.select_dtypes(include=np.number).columns.tolist()
print(len(num_features))
print(num_features)

# List of "no data" values
no_data_values = ["unknown", "NO DATA!", "NODATA!", "nodata!", "no data!", "INVALID!"]

# Replace NaN and "unknown" string values with -1 in non-numeric features
for feature in encoded_features.columns:
    if feature not in num_features:
        encoded_features[feature] = encoded_features[feature].replace(np.nan, -1)
        encoded_features[feature] = encoded_features[feature].replace(
            no_data_values, -1
        )

# Can we catch more numeric features with that?
num_features = encoded_features.select_dtypes(include=np.number).columns.tolist()
print(len(num_features))
print(num_features)

# Replace -1 value with "unknown" for categorical features
for feature in encoded_features.columns:
    if feature not in num_features:
        encoded_features[feature] = encoded_features[feature].replace(-1, "unknown")

# %%
# Set encoder and copy data
encoder = LabelEncoder()

# Encode categorical features
for feature in encoded_features.columns:
    if feature not in num_features:
        print(feature)
        encoded_features[feature] = encoder.fit_transform(encoded_features[feature])

# %%
print(encoded_features.shape)
encoded_features.head()

# %% [markdown]
# ## Identify and Filter Out Highly Correlated Features

# %%
# Pearson Correlation Matrix
plt.figure(figsize=(25, 15))
cor = encoded_features.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.bwr)
plt.title("Correlation Matrix (complete)")
plt.tight_layout()
plt.savefig(FIGPATH + "correlation_matrix_complete.png", dpi=200)
plt.show()

# %%
print(encoded_features.shape)

# Get upper diagonal triangle (non duplication of features)
cor_matrix = encoded_features.corr().abs()
upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))

# Identify highly correlated features
to_drop = [
    column
    for column in upper_tri.columns
    if any(upper_tri[column] > 0.75)
    if column not in ["HP_TYPE", "HP_INSTALLED"]
]
print(to_drop)

# Drop features
uncorrelated_features = encoded_features.drop(to_drop, axis=1)
print(uncorrelated_features.shape)

# %% [markdown]
# ## Prepare Training and Eval Data

# %%
# Seperate samples with and without heat pumps
X_hp = uncorrelated_features.loc[uncorrelated_features.HP_INSTALLED == True]
X_no_hp = uncorrelated_features.loc[uncorrelated_features.HP_INSTALLED == False]

# Shuffle and adjust size
X_no_hp = X_no_hp.sample(frac=1)
X_no_hp = X_no_hp[: X_hp.shape[0]]

print(X_hp.shape)
print(X_no_hp.shape)
X = pd.concat([X_hp, X_no_hp], axis=0)
print(X.shape)

# Set target value and remove from input
y = X["HP_INSTALLED"]
del X["HP_INSTALLED"]
del X["HP_TYPE"]
del X["MAINHEAT_DESCRIPTION"]
print()
print(X.shape)
print(y.shape)


# %% [markdown]
# ## Scaling and Dimensionality Reduction
#
# ... and some functions

# %%
def plot_explained_variance(dim_reduction, title):
    """Plot percentage of variance explained by each of the selected components
    after performing dimensionality reduction (e.g. PCA, LSA).
    Parameters
    ----------
    dim_reduction: sklearn.decomposition.PCA, sklearn.decomposition.TruncatedSVD
        Dimensionality reduction on features with PCA or LSA.
    title: str
        Title for saving plot.
    Return
    ----------
    None"""

    print(title)
    # Explained variance ratio (how much is covered by how many components)

    # Per component
    plt.plot(dim_reduction.explained_variance_ratio_)
    # Cumulative
    plt.plot(np.cumsum(dim_reduction.explained_variance_ratio_))

    # Assign labels and title
    plt.xlabel("Dimensions")
    plt.ylabel("Explained variance")
    plt.legend(["Explained Variance Ratio", "Summed Expl. Variance Ratio"])
    plt.title("Explained Variance Ratio by Dimensions " + title)

    plt.savefig(FIGPATH + "title", format="png", dpi=500)

    # Save plot
    # plotting.save_fig(plt, "Explained Variance Ratio by Dimensions " + title)

    # Show plot
    plt.show()


def dimensionality_reduction(
    features,
    dim_red_technique="LSA",
    lsa_n_comps=90,
    pca_expl_var_ratio=0.90,
    random_state=42,
):
    """Perform dimensionality reduction on given features.
    Parameters
    ----------
    features: np.array
        Original features on which to perform dimensionality reduction.
    dim_red_tequnique: 'LSA', 'PCA', default='LSA'
        Dimensionality reduction technique.
    lsa_n_comps: int, default=90
        Number of LSA components to use.
    pca_expl_var_ratio: float (between 0.0 and 1.0), default=0.90
        Automatically compute number of components that fulfill given explained variance ratio (e.g. 90%).
    random_state: int, default=42
        Seed for reproducible results.
    Return
    ---------
    lsa_transformed or pca_reduced_features: np.array
        Dimensionality reduced features."""

    if dim_red_technique.lower() == "lsa":

        # Latent semantic analysis (truncated SVD)
        lsa = TruncatedSVD(n_components=lsa_n_comps, random_state=random_state)
        lsa_transformed = lsa.fit_transform(features)

        plot_explained_variance(lsa, "LSA")

        print("Number of features after LSA: {}".format(lsa_transformed.shape[1]))

        return lsa_transformed

    elif dim_red_technique.lower() == "pca":

        # Principal component analysis
        pca = PCA(random_state=random_state)

        # Transform features
        pca_transformed = pca.fit_transform(features)

        plot_explained_variance(pca, "PCA")

        # Get top components (with combined explained variance ratio of e.g. 90%)
        pca_top = PCA(n_components=pca_expl_var_ratio)
        pca_reduced_features = pca_top.fit_transform(features)

        # print
        print("Number of features after PCA: {}".format(pca_reduced_features.shape[1]))

        return pca_reduced_features

    else:
        raise IOError(
            "Dimensionality reduction technique '{}' not implemented.".format(
                dim_red_technique
            )
        )


def plot_confusion_matrix(solutions, predictions, label_set, title):
    """Plot the confusion matrix for different classes given correct labels and predictions.

    Paramters:

            solutions (np.array) -- correct labels
            predictions (np.array) -- predicted labels
            label_set (list) -- labels/classes to predict
            title (string) -- plot title displayed above plot
    Return: None"""

    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(
        solutions, predictions, labels=range(len(label_set))
    )

    # Set figure size
    if len(label_set) > 5:
        plt.figure(figsize=(10, 10))
    else:
        plt.figure(figsize=(5, 5))

    # Plot  confusion matrix with blue color map
    plt.imshow(cm, interpolation="none", cmap="Blues")

    # Write out the number of instances per cell
    for (i, j), z in np.ndenumerate(cm):
        plt.text(j, i, z, ha="center", va="center")

    # Assign labels and title
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.title(title)

    # Set x ticks and labels
    plt.gca().set_xticks(range(len(label_set)))
    plt.gca().set_xticklabels(label_set, rotation=50)

    # Set y ticks and labels
    plt.gca().set_yticks(range(len(label_set)))
    plt.gca().set_yticklabels(label_set)
    plt.gca().invert_yaxis()

    plt.savefig(FIGPATH + title, format="png", dpi=500)

    # Show plot
    plt.show()


# %%
# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# %%
# Reduce dimensionality to level of 90% explained variance ratio
X_dim_reduced = dimensionality_reduction(
    X_scaled,
    dim_red_technique="pca",
    pca_expl_var_ratio=0.90,
    random_state=42,
)

# %% [markdown]
# ## Training and Testing

# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_dim_reduced, y, test_size=0.1, random_state=42, stratify=y
)

# %%

print("Numer of samples:", X.shape[0])
print("Numer of features:", X.shape[1])
print()

model_dict = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "Linear Support Vector Classifier": SGDClassifier(random_state=42),
}


def train_and_evaluate(model_name):

    model = model_dict[model_name]
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    print("Model Name:", model_name)

    # Plot the confusion matrix for training set
    plot_confusion_matrix(y_train, pred_train, ["No HP", "HP"], "Training set")

    # Plot the confusion matrix for validation set
    plot_confusion_matrix(y_test, pred_test, ["No HP", "HP"], "Validation set")

    # Print Accuracies
    print(
        "Accuracy train: {}%".format(
            np.round(sklearn.metrics.accuracy_score(y_train, pred_train) * 100), 2
        )
    )
    print(
        "Accuracy test:   {}%".format(
            np.round(sklearn.metrics.accuracy_score(y_test, pred_test) * 100), 2
        )
    )

    # print(model.coef_, model.intercept_)

    acc = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
    f1 = cross_val_score(model, X_train, y_train, cv=10, scoring="f1")
    recall = cross_val_score(model, X_train, y_train, cv=10, scoring="recall")
    precision = cross_val_score(model, X_train, y_train, cv=10, scoring="precision")
    print()
    print("10-fold Cross Validation\n---------\n")
    print("Accuracy:", round(acc.mean(), 2))
    print("F1 Score:", round(f1.mean(), 2))
    print("Recall:", round(recall.mean(), 2))
    print("Precision:", round(precision.mean(), 2))
    print()


for model in model_dict.keys():
    train_and_evaluate(model)


# %% [markdown]
# ## Coefficients Inspection

# %%
def get_sorted_coefficients(classifier, feature_names):
    """Get features and coefficients sorted by coeffience strength in Linear SVM.

    Parameter:

        classifier (sklearn.svm._classes.LinearSVC) -- linear SVM classifier (has to be fitted!)
        feature_names (list) -- feature names as list of strings

    Return:

        sort_idx (np.array) -- sorting array for features (feature with strongest coeffienct first)
        sorted_coef (np.array) -- sorted coefficient values
        sorted_fnames (list) -- feature names sorted by coefficient strength"""

    # Sort the feature indices according absolute coefficients (highest coefficient first)
    sort_idx = np.argsort(-abs(classifier.coef_).max(axis=0))

    # Get sorted coefficients and feature names
    sorted_coef = classifier.coef_[:, sort_idx]
    sorted_fnames = feature_names[sort_idx].tolist()

    sorted_fnames = [feature_names[i] for i in sort_idx]

    return sort_idx, sorted_coef, sorted_fnames


def plot_feature_coefficients(classifier, feature_names, label_set):
    """Plot the feature coefficients for each label given an SVM classifier.

    Paramters:

            classifier (sklearn.svm._classes.LinearSVC) -- linear SVM classifier (has to be fitted!)
            feature_names (list) -- feature names as list of strings
            label_set (list) -- label set as a list of strings
    Return: None
    """

    # Layout settings depending un number of labels
    if len(label_set) > 4:
        FIGSIZE = (80, 30)
        ROTATION = 35
        RIGHT = 0.81
    else:
        FIGSIZE = (40, 12)
        ROTATION = 45
        RIGHT = 0.58

    # Sort the feature indices according coefficients (highest coefficient first)
    sort_idx = np.argsort(-abs(classifier.coef_).max(axis=0))

    # Get sorted coefficients and feature names
    sorted_coef = classifier.coef_[:, sort_idx]
    sorted_fnames = feature_names[sort_idx]
    # sorted_fnames = [feature_names[i] for i in sort_idx]

    print(FIGSIZE)
    # Make subplots

    import matplotlib.pyplot as plt

    print(plt)
    x_fig, x_axis = plt.subplots(2, 1, figsize=FIGSIZE)

    # Plot coefficients on two different lines
    im_0 = x_axis[0].imshow(
        sorted_coef[:, : sorted_coef.shape[1] // 2],
        interpolation="none",
        cmap="seismic",
        vmin=-2.5,
        vmax=2.5,
    )
    im_1 = x_axis[1].imshow(
        sorted_coef[:, sorted_coef.shape[1] // 2 :],
        interpolation="none",
        cmap="seismic",
        vmin=-2.5,
        vmax=2.5,
    )

    # Set y ticks (number of classes)
    x_axis[0].set_yticks(range(len(label_set)))
    x_axis[1].set_yticks(range(len(label_set)))

    # Set the y labels (classes/labels)
    x_axis[0].set_yticklabels(label_set, fontsize=24)
    x_axis[1].set_yticklabels(label_set, fontsize=24)

    # Set x ticks (half the number of features) and labels
    x_axis[0].set_xticks(range(len(feature_names) // 2))
    x_axis[1].set_xticks(range(len(feature_names) // 2))

    # Set the x labels (feature names)
    x_axis[0].set_xticklabels(
        sorted_fnames[: len(feature_names) // 2],
        rotation=ROTATION,
        ha="right",
        fontsize=20,
    )
    x_axis[1].set_xticklabels(
        sorted_fnames[len(feature_names) // 2 :],
        rotation=ROTATION,
        ha="right",
        fontsize=20,
    )

    # Move plot to the right
    x_fig.subplots_adjust(right=RIGHT)

    # Set color bar
    cbar_ax = x_fig.add_axes([0.605, 0.15, 0.02, 0.7])
    cbar = x_fig.colorbar(im_0, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=24)

    # Show
    plt.show()


# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_dim_reduced, y, test_size=0.1, random_state=42, stratify=y
)

model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)

print(X_train.shape)
print(model.coef_.shape)
feature_names = np.array(["PC " + str(num) for num in range(X_train.shape[1])])
print(feature_names)
label_set = ["Has HP"]

# Get feature indices sorted by coefficient strength
sort_idx, _, _ = get_sorted_coefficients(model, feature_names)

# Plot the classifier's coefficients for each feature and label
plt = plot_feature_coefficients(model, feature_names, label_set)

# %%
# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.1, random_state=42, stratify=y
)

model = SGDClassifier(random_state=42)
model.fit(X_train, y_train)

print(X_train.shape)
print(model.coef_.shape)
feature_names = np.array(X.columns)
print(feature_names)
label_set = ["Has HP"]

# Get feature indices sorted by coefficient strength
sort_idx, _, _ = get_sorted_coefficients(model, feature_names)

# Plot the classifier's coefficients for each feature and label
plt = plot_feature_coefficients(model, feature_names, label_set)

# %%

# %%
