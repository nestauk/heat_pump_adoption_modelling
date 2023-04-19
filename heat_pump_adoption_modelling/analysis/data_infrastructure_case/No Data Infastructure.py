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
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: heat_pump_adoption_modelling
#     language: python
#     name: heat_pump_adoption_modelling
# ---

# %% [markdown]
# # Version with no Data Infrastructure
#
# ### Load Wales EPC Data and Plot Construction Age by Property Type

# %%
# %load_ext autoreload
# %autoreload 2
from heat_pump_adoption_modelling import PROJECT_DIR

import os
import matplotlib as plt

import matplotlib.pyplot as plt

plt.style.use("default")


# %%
def load_england_wales_data(
    RAW_ENG_WALES_DATA_PATH,
    batch=None,
    subset=None,
    usecols=None,
    n_samples=None,
    low_memory=False,
):
    """Load the England and/or Wales EPC data.

    Parameters
    ----------
    subset : {'England', 'Wales', None}, default=None
        EPC certificate area subset.
        If None, then the data for both England and Wales will be loaded.

    usecols : list, default=None
        List of features/columns to load from EPC dataset.
        If None, then all features will be loaded.

    n_samples : int, default=None
        Number of rows of file to read.

    low_memory : bool, default=False
        Internally process the file in chunks, resulting in lower memory use while parsing,
        but possibly mixed type inference.
        To ensure no mixed types either set False, or specify the type with the dtype parameter.

    Return
    ---------
    EPC_certs : pandas.DateFrame
        England/Wales EPC certificate data for given features."""

    #     if subset in [None, "GB", "all"]:

    #         additional_samples = 0

    #         if n_samples is not None:
    #             additional_samples = n_samples % 2
    #             n_samples = n_samples // 2

    #         wales_epc = load_england_wales_data(
    #             data_path=data_path,
    #             rel_data_path=rel_data_path,
    #             batch=batch,
    #             subset="Wales",
    #             usecols=usecols,
    #             n_samples=n_samples,
    #             low_memory=False,
    #         )

    #         england_epc = load_england_wales_data(
    #             data_path=data_path,
    #             rel_data_path=rel_data_path,
    #             batch=batch,
    #             subset="England",
    #             usecols=usecols,
    #             n_samples=None if n_samples is None else n_samples + additional_samples,
    #             low_memory=False,
    #         )

    #         epc_certs = pd.concat([wales_epc, england_epc], axis=0)

    #         return epc_certs

    #     RAW_ENG_WALES_DATA_PATH = get_version_path(
    #         Path(data_path) / rel_data_path, data_path=data_path, batch=batch
    #     )
    #     RAW_ENG_WALES_DATA_ZIP = get_version_path(
    #         Path(data_path) / base_config.RAW_ENG_WALES_DATA_ZIP,
    #         data_path=data_path,
    #         batch=batch,
    #     )

    #     # If sample file does not exist (probably just not unzipped), unzip the data
    #     if not Path(
    #         RAW_ENG_WALES_DATA_PATH / "domestic-W06000015-Cardiff/certificates.csv"
    #     ).is_file():
    #         extract_data(RAW_ENG_WALES_DATA_ZIP)

    #     # -----------------

    # Get all directories
    directories = [
        dir
        for dir in os.listdir(RAW_ENG_WALES_DATA_PATH)
        if not (dir.startswith(".") or dir.endswith(".txt") or dir.endswith(".zip"))
    ]

    # Set subset dict to select respective subset directories
    start_with_dict = {"Wales": "domestic-W", "England": "domestic-E"}

    directories = [
        dir for dir in directories if dir.startswith(start_with_dict[subset])
    ]

    # Load EPC certificates for given subset
    # Only load columns of interest (if given)
    epc_certs = [
        pd.read_csv(
            RAW_ENG_WALES_DATA_PATH / directory / "certificates.csv",
            low_memory=low_memory,
            usecols=usecols,
        )
        for directory in directories
    ]

    # Concatenate single dataframes into dataframe
    epc_certs = pd.concat(epc_certs, axis=0)
    epc_certs["COUNTRY"] = subset

    if "UPRN" in epc_certs.columns:
        epc_certs["UPRN"].fillna(epc_certs.BUILDING_REFERENCE_NUMBER, inplace=True)

    if n_samples is not None:
        epc_certs = epc_certs.sample(frac=1).reset_index(drop=True)[:n_samples]

    return epc_certs


# %%
# import pandas as pd

# %%
wales_epc = load_england_wales_data(
    PROJECT_DIR / "inputs/data/EPC/raw_data/England_Wales/", subset="Wales"
)
wales_epc.head()


# %%
def plot_subcats_by_other_subcats(
    df,
    feature_1,
    feature_2,
    feature_1_order=None,
    feature_2_order=None,
    normalize=True,
    plot_title=None,
    fig_save_path=None,
    y_label="",
    x_label="",
    plot_kind="bar",
    plotting_colors=None,
    y_ticklabel_type=None,
    x_tick_rotation=0,
    legend_loc="inside",
    with_labels=False,
    width=0.8,
    figsize=None,
):
    """Plot subcategories of given feature by subcategories of another feature.
     For example, plot and color-code the distribution of heating types (feature 2)
     on the different tenure types (feature 1).

     Parameters
     ----------

     df : pd.DataFrame
         Dataframe to analyse and plot.

     feature_1 : str
         Feature for which subcategories are plotted on x-axis.

     feature_2 : str
         Feature for which distribution is shown split per subcategory
         of feature 1. Feature 2 subcategories are represented with differnet colors,
         explained with a color legend.

     feature_1_subcat_order : list, None, default=None
         The order in which feature 1 subcategories are displayed.

     feature_2_subcat_order : list, None, default=None
         The order in which feature 2 subcategories are displayed.

    normalize : bool, default=True
        If True, relative numbers (percentage) instead of absolute numbers.

     plot_title : str, None, default=None
         Title to display above plot.
         If None, title is created automatically.
         Plot title is also used when saving file.

    fig_save_path : str, None, default=None
        Location where to save plot.

     y_label : str, default=""
         Label for y-axis.

     x_label : str, default=""
         Label for x-axis

     plot_kind : {"hist", "bar"}, default="hist"
         Type of plot.

     plotting_colors : list, str, None, default=None
         Ordered list of colors or color map to use when plotting feature 2.
         If list, use list of colors.
         If str, use corresponding matplotlib color map.
         If None, use default color list.

     y_ticklabel_type : {'', 'm', 'k' or '%'}, default=None
         Label for yticklabel, e.g. 'k' when displaying numbers
         in more compact way for easier readability (50000 --> 50k).

    x_tick_rotation : int, default=0
         Rotation of x-tick labels.
         If rotation set to 45, make end of label align with tick (ha="right")."""

    # Remove all samples for which feature 1 or feature 2 is NaN.
    # df = df[df[feature_1].notna()]
    # df = df[df[feature_2].notna()]

    # Get set of values/subcategories for features.
    feature_1_values = list(set(df[feature_1].sort_index()))
    feature_2_values = list(set(df[feature_2].sort_index()))

    # Set order for feature 1 values/subcategories
    if feature_1_order is not None:
        feature_1_values = feature_1_order

    # Create a feature-bar dict
    feat_bar_dict = {}

    # Get totals for noramlisation
    totals = df[feature_1].value_counts(dropna=False)

    # For every feature 2 value/subcategory, get feature 1 values
    # e.g. for every tenure type, get windows energy efficiencies
    for feat2 in feature_2_values:
        dataset_of_interest = df.loc[df[feature_2] == feat2][feature_1]
        data_of_interest = dataset_of_interest.value_counts(dropna=False)

        if normalize:
            feat_bar_dict[feat2] = data_of_interest / totals * 100
        else:
            feat_bar_dict[feat2] = data_of_interest

    # Save feature 2 subcategories by feature 1 subcategories as dataframe
    subcat_by_subcat = pd.DataFrame(feat_bar_dict, index=feature_1_values)

    # If feature 2 order is given, rearrange
    if feature_2_order is not None:
        subcat_by_subcat = subcat_by_subcat[feature_2_order]

    # If not defined, set default colors for plotting
    if plotting_colors is None:
        plotting_colors = ["green", "greenyellow", "yellow", "orange", "red"]

    # Use given colormap
    if isinstance(plotting_colors, str):
        cmap = plotting_colors
        subcat_by_subcat.plot(
            kind=plot_kind, cmap=cmap, width=width
        )  # recommended RdYlGn

    # or: use given color list
    elif isinstance(plotting_colors, list):
        subcat_by_subcat.plot(kind=plot_kind, color=plotting_colors, width=width)

    else:
        raise IOError("Invalid plotting_colors '{}'.".format(plotting_colors))

    # Adjust figsize
    if figsize is not None:
        fig = plt.gcf()
        fig.set_size_inches(figsize[0], figsize[1])

    # Get updated yticklabels
    ax = plt.gca()
    yticklabels, ax, _, _ = get_readable_tick_labels(plt, y_ticklabel_type, "y")
    ax.set_yticklabels(yticklabels)

    # Set plot title
    if plot_title is None:
        plot_title = feature_2 + " by " + feature_1

    # Describe plot with title and axes
    plt.title(plot_title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(
        rotation=x_tick_rotation, ha="right"
    ) if x_tick_rotation == 45 else plt.xticks(rotation=x_tick_rotation)

    if legend_loc == "outside":
        leg = plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        leg.set_draggable(state=True)

    if with_labels:
        labels = subcat_by_subcat.T.fillna(0.0).to_numpy().flatten()

        labels = [str(round(l)) + "%" for l in labels]
        rects = ax.patches

        # Make some labels.
        #  labels = [f"label{i}" for i in range(len(rects))]

        for rect, label in zip(rects, labels):
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.5,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Save figure
    save_figure(plt, plot_title, fig_path=fig_save_path)

    # Show plot
    plt.show()


# %%
plot_subcats_by_other_subcats(
    wales_epc,
    "PROPERTY_TYPE",
    "CONSTRUCTION_AGE_BAND",
    fig_save_path=PROJECT_DIR / "outputs/figures/",
)

# %% [markdown]
# # No!!!!
#
#
# ![cry.jpg](cry.jpg)
#
# - What if we want the same thing for Scotland?
# - Didn't even notice we're not using the most recent data
# - Multiple copies of same data eating up space on computer
#
# <br>
#
# - What if someone asks for the code?

# %%
