import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA, TruncatedSVD
import sklearn

from heat_pump_adoption_modelling import PROJECT_DIR

FIGPATH = PROJECT_DIR / "outputs/figures/"


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

    plt.savefig(FIGPATH / title, format="png", dpi=500)

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

    plt.savefig(FIGPATH / title, format="png", dpi=500)

    # Show plot
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

    print(plt)

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

    x_axis[0].set_title(title + "\n", fontdict={"fontsize": 30, "fontweight": "medium"})

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
    plt.savefig(FIGPATH / (title + ".png"), format="png", dpi=500)

    # Show
    plt.show()
