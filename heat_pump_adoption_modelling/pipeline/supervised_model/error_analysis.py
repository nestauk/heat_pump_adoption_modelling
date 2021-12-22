from heat_pump_adoption_modelling import PROJECT_DIR, get_yaml_config, Path
import matplotlib.pyplot as plt

# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

FIG_PATH = str(PROJECT_DIR) + config["SUPERVISED_MODEL_FIG_PATH"]


def decode_tenure(df):

    tenures = []
    for index, row in df.iterrows():
        if row["TENURE: owner-occupied"] == 1.0:
            tenure = "owner-occupied"
        elif row["TENURE: rental (private)"] == 1.0:
            tenure = "rental (private)"
        elif row["TENURE: rental (social)"] == 1.0:
            tenure = "rental (social)"
        elif row["TENURE: unknown"] == 1.0:
            tenure = "unknown"

        tenures.append(tenure)

    df["TENURE"] = tenures
    return df


tenure_color_dict = {
    "owner-occupied": "orange",
    "rental (social)": "teal",
    "rental (private)": "blue",
    "unknown": "purple",
}


def print_prediction_and_error(df, target):

    print("Mean ground truth:\t", round(df[target].mean() * 100, 2))
    print("Mean prediction:\t", round(df[target + ": prediction"].mean() * 100, 2))
    print("Mean error:\t\t", round(df[target + ": error"].mean() * 100, 2))


def plot_error_by_tenure(
    df,
    target,
    y_axis="Ground Truth",
    model_name="Random Forest Regressor",
    set_name="Validation Set",
):

    if "color" not in df.columns:
        df = decode_tenure(df)
        df["tenure color"] = df["TENURE"].map(tenure_color_dict)

    if y_axis == "Ground Truth":
        y = df[target]
    elif y_axis == "Predictions":
        y = df[target + ": prediction"]
    else:
        raise IOError("Not defined")

    title = "{} Error by {}\nusing {} on {}".format(
        target, y_axis, model_name, set_name
    )

    scatter = plt.scatter(
        df[target + ": error"],
        df[target],
        color=list(df["tenure color"]),
        alpha=0.5,
    )
    plt.title(title)
    plt.xlabel("Error")
    plt.ylabel(y_axis)

    color_patches = []

    for tenure in tenure_color_dict.keys():
        mpatch = mpatches.Patch(color=tenure_color_dict[tenure], label=tenure)
        color_patches.append(mpatch)

    plt.legend(handles=color_patches)

    plt.savefig(FIG_PATH + title.replace(" ", "_"), dpi=300, bbox_inches="tight")

    plt.show()
