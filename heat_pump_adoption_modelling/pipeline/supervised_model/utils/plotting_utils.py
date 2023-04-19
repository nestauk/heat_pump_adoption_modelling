# File: heat_pump_adoption_modelling/pipeline/supervised_model/utils/plotting_utils.py
"""
Predict the HP growth using a supervised learning model.
"""

# ----------------------------------------------------------------------------------


import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
import sklearn

from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path


# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = Path(str(PROJECT_DIR) + config["SUPERVISED_MODEL_FIG_PATH"])


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

    plt.savefig(FIG_PATH / title, format="png", dpi=500)

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


def plot_confusion_matrix(solutions, predictions, label_set=None, title=""):
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
        plt.figure(figsize=(3, 3))

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

    plt.savefig(
        FIG_PATH / (title.replace(":\n", " -") + ".png"),
        format="png",
        dpi=500,
        bbox_inches="tight",
    )

    # Show plot
    plt.show()


def get_most_important_coefficients(model, feature_names, title, X, top_features=10):

    if model.__class__.__name__ == "SVC":
        coef = model.coef_
        coef2 = coef.toarray().ravel() * X.std(axis=0)
        coef1 = coef2[: len(feature_names)]
    else:
        coef2 = model.coef_.ravel() * X.std(axis=0)
        coef1 = coef2[: len(feature_names)]

    top_positive_coefficients = np.argsort(coef1)[-top_features:]
    top_negative_coefficients = np.argsort(coef1)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef1[top_coefficients]]
    plt.bar(
        np.arange(2 * top_features),
        coef1[top_coefficients],
        color=colors,
        align="center",
    )
    feature_names = np.array(feature_names)
    plt.xticks(
        np.arange(0, 2 * top_features),
        feature_names[top_coefficients],
        rotation=45,
        ha="right",
    )
    plt.title(title)
    plt.savefig(
        FIG_PATH / (title.replace(":", " -") + ".png"), dpi=200, bbox_inches="tight"
    )

    plt.show()


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


def plot_feature_coefficients(classifier, feature_names, label_set, title):
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

    # Make subplots
    x_fig, x_axis = plt.subplots(2, 1, figsize=FIGSIZE)

    odd_n_rows = False
    if (sorted_coef.shape[1] % 2) != 0:
        odd_n_rows = True
        second_row_n = (sorted_coef.shape[1] // 2) + 1

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

    x_axis[0].set_title(
        title + "\n",
        fontdict={"fontsize": 30, "fontweight": "medium"},
    )

    # Set y ticks (number of classes)
    x_axis[0].set_yticks(range(len(label_set)))
    x_axis[1].set_yticks(range(len(label_set)))

    # Set the y labels (classes/labels)
    x_axis[0].set_yticklabels(label_set, fontsize=24)
    x_axis[1].set_yticklabels(label_set, fontsize=24)

    # Set x ticks (half the number of features) and labels
    x_axis[0].set_xticks(range(len(feature_names) // 2))
    if odd_n_rows:
        x_axis[1].set_xticks(range(second_row_n))
    else:
        x_axis[1].set_xticks(range(len(feature_names) // 2))

    # Set the x labels (feature names)
    x_axis[0].set_xticklabels(
        sorted_fnames[: len(feature_names) // 2],
        rotation=ROTATION,
        ha="right",
        fontsize=20,
    )
    x_axis[1].set_xticklabels(
        sorted_fnames[sorted_coef.shape[1] // 2 :],
        rotation=ROTATION,
        ha="right",
        fontsize=20,
    )

    plt.tight_layout()

    # Move plot to the right
    x_fig.subplots_adjust(right=RIGHT)

    # Set color bar
    cbar_ax = x_fig.add_axes([0.605, 0.15, 0.02, 0.7])
    cbar = x_fig.colorbar(im_0, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=24)

    # plt.title(title)
    plt.savefig(FIG_PATH / (title.replace(":", " -") + ".png"), format="png", dpi=500)

    # Show
    plt.show()


def map_percentage_to_bin(predictions, solutions, interval=5):

    values = [
        [str(i) + "-" + str(i + interval) + "%"] * interval
        for i in range(0, 100, interval)
    ]
    values = [item for sublist in values for item in sublist]
    keys = [i / 100 for i in range(0, 100)]

    cat_dict = dict(zip(keys, values))
    n_categories = len(set(cat_dict.values()))
    categories = [cat_dict[key] for key in sorted(set(cat_dict.keys()))]
    categories = sorted(set(categories), key=categories.index)

    cat_dict[1.0] = categories[-1]

    ordinal_labels = [int(i) for i in range(n_categories)]
    label_dict_1 = dict(zip(categories, ordinal_labels))
    label_dict_2 = dict(zip(ordinal_labels, categories))
    label_dict = {**label_dict_1, **label_dict_2}

    predictions = np.round(predictions, 2).astype("float")
    solutions = np.round(solutions, 2).astype("float")

    # print("-----")
    # print(predictions)
    # print(solutions)
    # print(np.unique(predictions, return_counts=True))
    # print(np.unique(solutions, return_counts=True))

    predictions_cats = np.vectorize(cat_dict.get)(predictions)
    solutions_cats = np.vectorize(cat_dict.get)(solutions)

    # print(solutions_cats)
    # print(np.unique(predictions_cats, return_counts=True))
    # print(np.unique(solutions_cats, return_counts=True))
    # print(label_dict)

    predictions_labels = np.vectorize(label_dict.get)(predictions_cats)
    solutions_labels = np.vectorize(label_dict.get)(solutions_cats)

    return predictions_labels, solutions_labels, label_dict


def scatter_plot(x, y, title, xlabel, ylabel):

    plt.scatter(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    plt.savefig(FIG_PATH / (title + ".png"), format="png", dpi=500)


def display_scores(scores, cv="Multi"):

    train_scores = [s for s in scores.keys() if "train" in s]
    test_scores = [s for s in scores.keys() if "test" in s]

    for score_set, set_name in [
        (train_scores, "Training Set"),
        (test_scores, "Test Set"),
    ]:

        print("\n{}-fold Cross Validation: {}\n{}\n".format(cv, set_name, 40 * "-"))

        for score in score_set:
            score_name = score.replace("test_", "")
            score_name = score_name.replace("train_", "")
            print(score_name, "mean:", round(scores[score].mean(), 2))

    print()
