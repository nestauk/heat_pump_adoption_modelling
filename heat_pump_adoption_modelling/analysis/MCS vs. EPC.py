# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: epc_data_analysis
#     language: python
#     name: epc_data_analysis
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from epc_data_analysis import PROJECT_DIR, get_yaml_config, Path

from epc_data_analysis.pipeline.preprocessing import feature_engineering
from epc_data_analysis.visualisation import easy_plotting, feature_settings


import pandas as pd
from keplergl import KeplerGl
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

from ipywidgets import interact

# %%
# epc_df = pd.read_csv(
#    data_preprocessing.SUPERVISED_MODEL_OUTPUT + "epc_df_5m_preprocessed.csv"
# )


epc_df = pd.read_csv(str(PROJECT_DIR) + "/inputs/epc_df_complete_preprocessed.csv")

# %%
import numpy as np

epc_df.loc[(epc_df["HP_INSTALLED"] == False) & ~(epc_df["FIRST_HP_MENTION"].isna())][
    ["HP_INSTALLED", "FIRST_HP_MENTION", "BUILDING_ID"]
].head()

# %%
epc_df.loc[epc_df["BUILDING_ID"] == 1036632212202526][
    ["HP_INSTALLED", "FIRST_HP_MENTION_YEAR", "INSPECTION_YEAR"]
]

# %%
epc_df.sort_values("INSPECTION_DATE", ascending=True)["INSPECTION_DATE"].head()

# %%
epc_df["INSPECTION_DATE"] = epc_df["INSPECTION_DATE"].astype("datetime64[ns]")
epc_df.dtypes


# %%
# epc_df["DATE_INT"] = epc_df['INSPECTION_DATE'].dt.year +epc_df['INSPECTION_DATE'].dt.month +epc_df['INSPECTION_DATE'].dt.day


dedupl_epc_df = feature_engineering.filter_by_year(
    epc_df, "BUILDING_ID", year=2021, up_to=True, selection="latest entry"
)


# %%
print(dedupl_epc_df.shape)
print(epc_df.shape)

# %%
dedupl_epc_df.loc[dedupl_epc_df["BUILDING_ID"] == 1036632212202526][
    ["HP_INSTALLED", "FIRST_HP_MENTION_YEAR", "INSPECTION_YEAR"]
]

# %%
dedupl_epc_df["MCS_AVAILABLE"].value_counts()

# %%
mcs_available = dedupl_epc_df["MCS_AVAILABLE"] == True
epc_hp_mention = ~(dedupl_epc_df["FIRST_HP_MENTION"].isna())

# %%
print("Number of EPC entries:", dedupl_epc_df.shape[0])
print("Number of EPC entries with HP mention", dedupl_epc_df[epc_hp_mention].shape[0])
print("Number of MCS entries with EPC match: ", dedupl_epc_df[mcs_available].shape[0])
print()
print(
    "Both EPC HP and MCS install:",
    dedupl_epc_df.loc[epc_hp_mention & mcs_available].shape[0],
)
print(
    "EPC HP or MCS install:", dedupl_epc_df.loc[epc_hp_mention | mcs_available].shape[0]
)
print()
print(
    "Only EPC HP mention:", dedupl_epc_df.loc[epc_hp_mention & ~mcs_available].shape[0]
)
print("Only MCS mention:", dedupl_epc_df.loc[~epc_hp_mention & mcs_available].shape[0])

# %%
dedupl_epc_df.loc[~epc_hp_mention & ~mcs_available, "HP set"] = "None"
dedupl_epc_df.loc[epc_hp_mention | mcs_available, "HP set"] = "either"
dedupl_epc_df.loc[epc_hp_mention & mcs_available, "HP set"] = "EPC & MCS"
dedupl_epc_df.loc[epc_hp_mention & ~mcs_available, "HP set"] = "EPC only"
dedupl_epc_df.loc[~epc_hp_mention & mcs_available, "HP set"] = "MCS only"

# %%
# dedupl_epc_df.loc[epc_hp_mention, "HP set"] = "EPC (all)"
# dedupl_epc_df.loc[mcs_available, "HP set"] = "MCS (all)"

# %%
dedupl_epc_df["HP set"].value_counts(dropna=False)

# %%
dedupl_epc_df.head()

# %%
any_hp = dedupl_epc_df.loc[dedupl_epc_df["HP set"] != "None"]

feature_1_order = ["EPC & MCS", "EPC only", "MCS only"]
# feature_1_order = ['EPC (all)',  'MCS (all)']

# %%
from ipywidgets import interact


@interact(feature_2=any_hp.columns[1:])
def plot_feature_subcats_by_other_feature_subcats(feature_2):

    easy_plotting.plot_subcats_by_other_subcats(
        any_hp,
        "HP set",
        feature_2,
    )


# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "TENURE",
    feature_1_order=feature_1_order,
    feature_2_order=[
        "owner-occupied",
        "rental (private)",
        "rental (social)",
        "unknown",
    ],
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Tenure",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "BUILT_FORM",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Built Form",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "PROPERTY_TYPE",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.prop_type_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Property Type",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "CURRENT_ENERGY_RATING",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.rating_order,
    plotting_colors="RdYlGn_r",
    plot_title="HP Install Information Source split by EPC Rating",
)

# %%
any_hp["ENTRY_YEAR_INT"] = any_hp["ENTRY_YEAR_INT"].astype("int")

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "ENTRY_YEAR_INT",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Entry Year",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "IMD Decile",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="RdYlGn",
    plot_title="HP Install Information Source split by IMD Decile",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "Country",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Country",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "CONSTRUCTION_AGE_BAND",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.const_year_order_merged,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Construction Age Band",
    legend_loc="outside",
)

# %%
any_hp["FIRST_HP_MENTION_YEAR"].fillna(0, inplace=True)
any_hp["FIRST_HP_MENTION_YEAR"] = any_hp["FIRST_HP_MENTION_YEAR"].astype("int")

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "FIRST_HP_MENTION_YEAR",
    feature_1_order=feature_1_order,
    feature_2_order=[
        0,
        2008,
        2009,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
    ],
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by First HP Mention",
)

# %%

# any_hp['N_ENTRIES'] = any_hp['N_ENTRIES'].astype('str')
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "N_ENTRIES",
    feature_1_order=feature_1_order,
    feature_2_order=["1", "2", "3", "4"],
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by # Entries",
)

# %%
any_hp["HP_INSTALL_YEAR"].value_counts(dropna=False)


# %%
any_hp["HP_INSTALL_YEAR"].fillna(0, inplace=True)
any_hp["HP_INSTALL_YEAR"] = any_hp["HP_INSTALL_DATE"].astype(str).str[:4]


# %%
any_hp["HP_INSTALL_YEAR"].replace([np.nan, -0.0], -1, inplace=True)
any_hp["HP_INSTALL_YEAR"].value_counts(dropna=False)

# %%

any_hp["HP_INSTALL_YEAR_NEW"] = round(any_hp["HP_INSTALL_DATE"] / 10000.0)

any_hp["HP_INSTALL_YEAR_NEW"].value_counts(dropna=False)

# %%
any_hp["HP_INSTALL_YEAR"].value_counts(dropna=False)

# %%

any_hp["HP_INSTALL_YEAR"] = any_hp["HP_INSTALL_YEAR"].astype("int")

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "HP_INSTALL_YEAR",
    feature_2_order=[
        -1,
        2010,
        2011,
        2012,
        2013,
        2014,
        2015,
        2016,
        2017,
        2018,
        2019,
        2020,
        2021,
    ],
    feature_1_order=feature_1_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by HP Install Year",
)

# %%

any_hp["ENTRY_YEAR_INT"] = any_hp["ENTRY_YEAR_INT"].astype("int")

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "ENTRY_YEAR_INT",
    # feature_2_order=[-1,  2010,2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    feature_1_order=feature_1_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by ENTRY_YEAR_INT",
)

# %%
any_hp["TRANSACTION_TYPE"].value_counts()

# %%

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "TRANSACTION_TYPE",
    # feature_2_order=[-1,  2010,2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    feature_1_order=feature_1_order,
    feature_2_order=[
        "new dwelling",
        "RHI application",
        "assessment for green deal",
        "rental",
        "rental (social)",
        "rental (private)",
        "not sale or rental",
        "ECO assessment",
        "marketed sale",
        "FiT application",
    ],
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Transaction Type",
    legend_loc="outside",
)

# %%
any_hp.loc[
    any_hp["CONSTRUCTION_AGE_BAND"].isin(
        ["1991-1998", "1996-2002", "2003-2007", "2007 onwards", "unknown"]
    )
    & (any_hp["HP set"] == "EPC only")
]["N_ENTRIES"].value_counts(dropna=False, normalize=True)

# %%
any_hp.loc[
    any_hp["CONSTRUCTION_AGE_BAND"].isin(
        ["1991-1998", "1996-2002", "2003-2007", "2007 onwards", "unknown"]
    )
    & (any_hp["HP set"] == "MCS only")
]["N_ENTRIES"].value_counts(dropna=False, normalize=True)

# %%
any_hp.loc[
    any_hp["CONSTRUCTION_AGE_BAND"].isin(
        ["1996-2002", "2003-2007", "2007 onwards", "unknown"]
    )
].shape

# %%
any_hp.loc[
    ~any_hp["CONSTRUCTION_AGE_BAND"].isin(
        ["1991-1998", "1996-2002", "2003-2007", "2007 onwards", "unknown"]
    )
    & (any_hp["HP set"] == "EPC only")
]["N_ENTRIES"].value_counts(dropna=False, normalize=True)

# %%
any_hp.loc[
    ~any_hp["CONSTRUCTION_AGE_BAND"].isin(
        ["1991-1998", "1996-2002", "2003-2007", "2007 onwards", "unknown"]
    )
    & (any_hp["HP set"] == "MCS only")
]["N_ENTRIES"].value_counts(dropna=False, normalize=True)

# %%
# any_hp['YEAR_DIFF_MENTION'] = any_hp['FIRST_HP_MENTION_YEAR'] - any_hp['ENTRY_YEAR_INT']
no_zeroes = any_hp.loc[any_hp["HP_INSTALL_YEAR"] != -1]

print(no_zeroes["ENTRY_YEAR_INT"].value_counts(dropna=False))
print(no_zeroes["HP_INSTALL_YEAR"].value_counts(dropna=False))
no_zeroes["YEAR_DIFF_INSTALL"] = (
    no_zeroes["ENTRY_YEAR_INT"] - no_zeroes["HP_INSTALL_YEAR"]
)

no_zeroes["YEAR_DIFF_INSTALL"].value_counts(dropna=False)

# %%
any_hp["YEAR_DIFF_INSTALL"] = any_hp["ENTRY_YEAR_INT"] - any_hp["HP_INSTALL_YEAR"]
any_hp.loc[any_hp["HP set"] == "EPC & MCS"][
    "YEAR_DIFF_INSTALL"
].value_counts() / any_hp.loc[any_hp["HP set"] == "EPC & MCS"].shape[0]

# %%
any_hp["HP set"].value_counts(dropna=False)

# %%
any_hp.loc[
    (any_hp["HP set"] != "EPC only") & (any_hp["YEAR_DIFF_INSTALL"].isin([0, 1]))
]["TRANSACTION_TYPE"].value_counts(normalize=True)

# %%
any_hp["YEAR_DIFF_INSTALL"] = any_hp["ENTRY_YEAR_INT"] - any_hp["HP_INSTALL_YEAR"]
any_hp.loc[
    (any_hp["HP set"] != "EPC only") & (any_hp["YEAR_DIFF_INSTALL"].isin([0, 1]))
]["TRANSACTION_TYPE"].value_counts(normalize=True)

# %%
any_hp.loc[any_hp["HP set"] == "MCS only"]["YEAR_DIFF_INSTALL"].value_counts(
    normalize=True
)

# %%

# any_hp['HP_INSTALL_YEAR'] = any_hp['HP_INSTALL_YEAR'].astype('int')

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "YEAR_DIFF_INSTALL",
    # feature_2_order=[0,  2010,2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
    feature_1_order=feature_1_order,
    plotting_colors="viridis",
    plot_title="HP Install Information Source split by Diff between Install and Entry Year",
)

# %%
any_hp.loc[
    any_hp["HP set"]
    == "EPC & MCS" & any_hp["FIRST_HP_MENTION_YEAR"]
    == any_hp["ENTRY_YEAR_INT"]
].shape

# %%
# epc_mcs_df = pd.read_csv(str(PROJECT_DIR) + "/inputs/mcs_epc.csv")
epc_mcs_df.shape

# %%
epc_mcs_df.columns

# %%
epc_mcs_df.loc[epc_mcs_df["epc_address"].isna()].shape

# %%
epc_mcs_df.loc[~epc_mcs_df["epc_address"].isna()].shape

# %%
epc_mcs_df.loc[~epc_mcs_df["epc_address"].isna()][
    "BUILDING_REFERENCE_NUMBER"
].unique().shape

# %%
51150 - 50939

# %%
epc_mcs_df["date"].min()

# %%
any_hp.columns

# %%
from keplergl import KeplerGl
from epc_data_analysis.getters import epc_data, util_data
from epc_data_analysis.pipeline import data_agglomeration

any_hp = feature_engineering.get_coordinates(any_hp)

print(any_hp.columns)
any_hp = data_agglomeration.add_hex_id(any_hp, resolution=7.5)

# %%
mcs_epc = any_hp.loc[any_hp["HP set"] == "EPC & MCS"][
    ["LONGITUDE", "LATITUDE", "hex_id", "TENURE"]
]
only_mcs = any_hp.loc[any_hp["HP set"] == "MCS only"][
    ["LONGITUDE", "LATITUDE", "hex_id", "TENURE"]
]
only_epc = any_hp.loc[any_hp["HP set"] == "EPC only"][
    ["LONGITUDE", "LATITUDE", "hex_id", "TENURE"]
]

all_mcs = any_hp.loc[any_hp["MCS_AVAILABLE"] == True][
    ["LONGITUDE", "LATITUDE", "hex_id", "TENURE"]
]
all_epc = any_hp[~(any_hp["FIRST_HP_MENTION_YEAR"].isna())][
    ["LONGITUDE", "LATITUDE", "hex_id", "TENURE"]
]

# dedupl_epc_df.loc[epc_hp_mention, "HP set"] = "EPC (all)"
# dedupl_epc_df.loc[mcs_available, "HP set"] = "MCS (all)"

# %%
mcs_epc = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    mcs_epc, "TENURE", agglo_feature="hex_id"
)
only_mcs = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    only_mcs, "TENURE", agglo_feature="hex_id"
)
only_epc = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    only_epc, "TENURE", agglo_feature="hex_id"
)

all_mcs = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    all_mcs, "TENURE", agglo_feature="hex_id"
)
all_epc = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    all_epc, "TENURE", agglo_feature="hex_id"
)


everything = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    any_hp, "TENURE", agglo_feature="hex_id"
)
everything.rename(columns={"hex_id_TOTAL": "TOTAL"}, inplace=True)
# mcs_epc = pd.merge(mcs_epc, hex_df, on='hex_id', how='right')

mcs_epc = pd.merge(mcs_epc, everything, on="hex_id", how="right")
only_mcs = pd.merge(only_mcs, everything, on="hex_id", how="right")
only_epc = pd.merge(only_epc, everything, on="hex_id", how="right")

all_mcs = pd.merge(all_mcs, everything, on="hex_id", how="right")
all_epc = pd.merge(all_epc, everything, on="hex_id", how="right")


mcs_epc["perc"] = round((mcs_epc["hex_id_TOTAL"] / mcs_epc["TOTAL"]).astype(float), 2)
only_mcs["perc"] = round(
    (only_mcs["hex_id_TOTAL"] / only_mcs["TOTAL"]).astype(float), 2
)
only_epc["perc"] = round((only_epc["hex_id_TOTAL"] / only_epc["TOTAL"]), 2)

all_mcs["perc"] = round((all_mcs["hex_id_TOTAL"] / all_mcs["TOTAL"]).astype(float), 2)
all_epc["perc"] = round((all_epc["hex_id_TOTAL"] / all_epc["TOTAL"]), 2)

only_epc["perc"].fillna(0.0, inplace=True)
only_mcs["perc"].fillna(0.0, inplace=True)
mcs_epc["perc"].fillna(0.0, inplace=True)
all_mcs["perc"].fillna(0.0, inplace=True)
all_epc["perc"].fillna(0.0, inplace=True)

# %%
config = kepler.get_config(MAPS_CONFIG_PATH + "epc_mcs_comparison.txt")

epc_mcs_comp = KeplerGl(height=500, config=config)

epc_mcs_comp.add_data(
    data=mcs_epc[["hex_id", "perc"]],
    name="EPC & MCS",
)

epc_mcs_comp.add_data(
    data=only_mcs[["hex_id", "perc"]],
    name="MCS only",
)

epc_mcs_comp.add_data(
    data=only_epc[["hex_id", "perc"]],
    name="EPC only",
)

epc_mcs_comp.add_data(
    data=all_epc[["hex_id", "perc"]],
    name="EPC (all)",
)


epc_mcs_comp.add_data(
    data=all_mcs[["hex_id", "perc"]],
    name="MCS (all)",
)


epc_mcs_comp

# %%
kepler.save_config(epc_mcs_comp, MAPS_CONFIG_PATH + "epc_mcs_comparison.txt")

epc_mcs_comp.save_to_html(file_name=MAPS_OUTPUT_PATH + "EPC_vs_MCS.html")

# %%
config = kepler.get_config(MAPS_CONFIG_PATH + "epc_mcs_comparison.txt")

epc_mcs_comp = KeplerGl(height=500, config=config)

epc_mcs_comp.add_data(
    data=mcs_epc[["hex_id", "perc"]],
    name="EPC & MCS",
)

epc_mcs_comp.add_data(
    data=only_mcs[["hex_id", "perc"]],
    name="MCS only",
)

epc_mcs_comp.add_data(
    data=only_epc[["hex_id", "perc"]],
    name="EPC only",
)

epc_mcs_comp

# %%

from epc_data_analysis.config.kepler.kepler_config import (
    MAPS_CONFIG_PATH,
    MAPS_OUTPUT_PATH,
)

# %%
kepler.save_config(epc_mcs_comp, MAPS_CONFIG_PATH + "epc_mcs_comparison.txt")

epc_mcs_comp.save_to_html(file_name=MAPS_OUTPUT_PATH + "EPC_vs_MCS.html")

# %%
