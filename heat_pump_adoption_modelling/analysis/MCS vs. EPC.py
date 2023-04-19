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

from epc_data_analysis.pipeline import data_agglomeration

from epc_data_analysis.config.kepler.kepler_config import (
    MAPS_CONFIG_PATH,
    MAPS_OUTPUT_PATH,
)

from keplergl import KeplerGl
from epc_data_analysis.getters import epc_data, util_data
from epc_data_analysis.pipeline import data_agglomeration

import pandas as pd

from datetime import timedelta
import matplotlib as mpl

mpl.rcParams.update(mpl.rcParamsDefault)

import collections
import datetime
from ipywidgets import interact


# %%
populations = pd.read_csv(str(PROJECT_DIR) + "/inputs/population.csv")[
    ["laname21", "population_2020"]
]
populations.columns = ["LOCAL_AUTHORITY_LABEL", "POPULATION"]

# %%
epc_df = pd.read_csv(
    str(PROJECT_DIR) + "/inputs/epc_df_complete_preprocessed.csv",
    parse_dates=["INSPECTION_DATE", "HP_INSTALL_DATE", "FIRST_HP_MENTION"],
)

# %%
epc_df.columns

# %%
dedupl_epc_df = feature_engineering.filter_by_year(
    epc_df, "UPRN", year=2021, up_to=True, selection="latest entry"
)
print(dedupl_epc_df.shape)
print(epc_df.shape)
print(dedupl_epc_df.columns)


# %%
dedupl_epc_df = feature_engineering.get_coordinates(dedupl_epc_df)
dedupl_epc_df = data_agglomeration.add_hex_id(dedupl_epc_df, resolution=7.5)

hex_to_LA = data_agglomeration.map_hex_to_feature(
    dedupl_epc_df, "LOCAL_AUTHORITY_LABEL"
)

# %%
dedupl_epc_df["HP_LOST_ALL"] = ~dedupl_epc_df["HP_AT_LAST"] & dedupl_epc_df["ANY_HP"]
dedupl_epc_df["HP_ADDED_ALL"] = ~dedupl_epc_df["HP_AT_FIRST"] & dedupl_epc_df["ANY_HP"]
dedupl_epc_df["HP_IN_MIDDLE"] = (
    ~dedupl_epc_df["HP_AT_FIRST"]
    & ~dedupl_epc_df["HP_AT_LAST"]
    & dedupl_epc_df["ANY_HP"]
)

# %%
mcs_available = dedupl_epc_df["MCS_AVAILABLE"] == True
epc_hp_mention = ~(dedupl_epc_df["FIRST_HP_MENTION"].isna())

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
print(dedupl_epc_df["EPC HP entry before MCS"].value_counts(dropna=False))

epc_entry_before_mcs = dedupl_epc_df.loc[dedupl_epc_df["EPC HP entry before MCS"]]

epc_entry_before_mcs["> 1 YEAR DIFF"] = (
    epc_entry_before_mcs["HP_INSTALL_DATE"] - epc_entry_before_mcs["INSPECTION_DATE"]
) > timedelta(days=356)

epc_entry_before_mcs["> 1 YEAR DIFF"].value_counts(normalize=True)

# %%
print(dedupl_epc_df["No EPC HP entry after MCS"].value_counts(dropna=False))

no_epc_entry_after_mcs = dedupl_epc_df.loc[dedupl_epc_df["No EPC HP entry after MCS"]]

no_epc_entry_after_mcs["> 1 YEAR DIFF"] = (
    no_epc_entry_after_mcs["INSPECTION_DATE"]
    - no_epc_entry_after_mcs["HP_INSTALL_DATE"]
) > timedelta(days=365)

no_epc_entry_after_mcs["> 1 YEAR DIFF"].value_counts(normalize=True)

# %%
print(no_epc_entry_after_mcs["HP_LOST_ALL"].value_counts(normalize=True))

no_epc_entry_after_mcs.loc[no_epc_entry_after_mcs["> 1 YEAR DIFF"]][
    "HP_LOST_ALL"
].value_counts(normalize=True)

# %%
print(epc_entry_before_mcs["HP_LOST_ALL"].value_counts(normalize=True))

epc_entry_before_mcs.loc[epc_entry_before_mcs["> 1 YEAR DIFF"]][
    "HP_LOST_ALL"
].value_counts(normalize=True)

# %%
none = dedupl_epc_df.loc[~epc_hp_mention & ~mcs_available]
none["HP set"] = "No HP"

either = dedupl_epc_df.loc[epc_hp_mention | mcs_available]
either["HP set"] = "EPC | MCS"

# both = dedupl_epc_df.loc[epc_hp_mention & mcs_available]
both = dedupl_epc_df.loc[epc_hp_mention & mcs_available]
both["HP set"] = "EPC & MCS"

epc_only = dedupl_epc_df.loc[epc_hp_mention & ~mcs_available]
epc_only["HP set"] = "EPC only"

mcs_only = dedupl_epc_df.loc[~epc_hp_mention & mcs_available]
mcs_only["HP set"] = "MCS only"

epc_all = dedupl_epc_df.loc[epc_hp_mention]
epc_all["HP set"] = "EPC all"

mcs_all = dedupl_epc_df.loc[mcs_available]
mcs_all["HP set"] = "MCS all"

epc_hp_before_mcs = dedupl_epc_df.loc[dedupl_epc_df["EPC HP entry before MCS"]]
epc_hp_before_mcs["HP set"] = "Conflict A"


no_epc_after_mcs = dedupl_epc_df.loc[dedupl_epc_df["No EPC HP entry after MCS"]]
no_epc_after_mcs["HP set"] = "Conflict B"

conflict_a_short = epc_entry_before_mcs.loc[epc_entry_before_mcs["> 1 YEAR DIFF"]]
conflict_a_short["HP set"] = "Conflict A > 1 year"

conflict_b_short = no_epc_entry_after_mcs.loc[no_epc_entry_after_mcs["> 1 YEAR DIFF"]]
conflict_b_short["HP set"] = "Conflict B > 1 year"


any_hp = pd.concat(
    [
        none,
        either,
        both,
        epc_only,
        mcs_only,
        epc_all,
        mcs_all,
        no_epc_after_mcs,
        epc_hp_before_mcs,
    ]
)
feature_1_order = [
    "No HP",
    "EPC | MCS",
    "EPC & MCS",
    "EPC all",
    "EPC only",
    "MCS all",
    "MCS only",
]

# eature_1_order = ['EPC | MCS', 'EPC & MCS', 'EPC all', 'EPC only', 'MCS all', 'MCS only']
# feature_1_order = [ 'Conflict A',  'Conflict A > 1 year',  'Conflict B',  'Conflict B > 1 year']
any_hp["HP set"].value_counts(dropna=False)

# %%
# feature_1_order = [
##    'EPC | MCS',
#    "EPC & MCS",
#    "EPC all",
#    "EPC only",
#    "MCS all",
#    "MCS only",
# ]

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
    plot_title="HP Subsets split by Tenure",
    figsize=(15, 5),
    with_labels=True,
    legend_loc="middle",
    width=0.65,
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "BUILT_FORM",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by Built Form",
    figsize=(15, 5),
    width=0.7,
    # legend_loc='outside',
    with_labels=True,
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "HP_LOST_ALL",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by Built Form",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "HP_TYPE",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by HP Type",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "HAS_HP_AT_SOME_POINT",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by 'has HP at some point'",
)

# %%
any_hp["test"] = (any_hp["HP_INSTALL_DATE"] < any_hp["INSPECTION_DATE"]) & (
    any_hp["HP_INSTALLED"]
)
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "test",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by 'has HP at some point'",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "HP_INSTALLED",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.built_form_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by 'has HP at some point'",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "PROPERTY_TYPE",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.prop_type_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by Property Type",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "CURRENT_ENERGY_RATING",
    feature_1_order=feature_1_order,
    feature_2_order=["A", "B", "C", "D", "E", "F", "G"],
    plotting_colors="RdYlGn_r",
    plot_title="HP Subsets split by EPC Rating",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%
any_hp["installation_type"] = any_hp["installation_type"].str.strip()
any_hp["installation_type"].value_counts()

# %%
any_hp["installation_type"].fillna("unknown", inplace=True)
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "installation_type",
    feature_1_order=feature_1_order,
    feature_2_order=[
        "Domestic",
        "Non-Domestic",
        "Commercial",
        "Unspecified",
        "unknown",
    ],
    plotting_colors="viridis",
    plot_title="HP Subsets split by Installation Type",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%
any_hp["# records"].fillna("unknown", inplace=True).astype("int")
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "# records",
    feature_1_order=feature_1_order,
    # feature_2_order=["Domestic", "Non-Domestic", "Commercial", "Unspecified", "unknown"],
    plotting_colors="RdYlGn_r",
    plot_title="HP Subsets split by # Records",
)

# %%
any_hp["ENTRY_YEAR"] = any_hp["INSPECTION_DATE"].dt.year
any_hp["ENTRY_YEAR"].fillna(0, inplace=True)
any_hp["ENTRY_YEAR"] = any_hp["ENTRY_YEAR"].astype("int")


easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "ENTRY_YEAR",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    legend_loc="outside",
    plot_title="HP Subsets split by Entry Year",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%
any_hp.loc[any_hp["HP set"] == "EPC & MCS"]["IMD Decile"].mean()

# %%
print(any_hp.loc[any_hp["HP set"] == "EPC all"]["IMD Decile"].median())
print(any_hp.loc[any_hp["HP set"] == "EPC only"]["IMD Decile"].median())
print(any_hp.loc[any_hp["HP set"] == "MCS all"]["IMD Decile"].median())
print(any_hp.loc[any_hp["HP set"] == "MCS only"]["IMD Decile"].median())
print(any_hp.loc[any_hp["HP set"] == "Conflict A"]["IMD Decile"].median())
print(any_hp.loc[any_hp["HP set"] == "Conflict B"]["IMD Decile"].median())


# %%
print(any_hp.loc[any_hp["HP set"] == "EPC | MCS"]["IMD Decile"].median())
print(any_hp.loc[any_hp["HP set"] == "No HP"]["IMD Decile"].median())

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "IMD Decile",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="RdYlGn",
    # legend_loc="outside",
    plot_title="HP Subsets split by IMD Decile",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%
any_hp["Country"].value_counts()

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "Country",
    "HP set",
    feature_1_order=["England", "Wales", "Scotland"],
    feature_2_order=feature_1_order,
    # feature_2_order=["England", "Wales", 'Scotland'],
    plotting_colors="viridis",
    plot_title="HP Subsets split by Country (split)",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%
any_hp["alt_type"].fillna("unknown", inplace=True)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "alt_type",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    legend_loc="outside",
    plot_title="HP Subsets split by alt_type",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    # with_labels=True
)

# %%
any_hp["version"].fillna("unknown", inplace=True)

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "version",
    feature_1_order=feature_1_order,
    feature_2_order=[1.0, 2.0, 3.0, 4.0, 5.0],
    # feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by n_certificates",
)

# %%
any_hp["version"].value_counts(dropna=False)

# %%
any_hp["n_certificates"].fillna("unknown", inplace=True)

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "n_certificates",
    feature_1_order=feature_1_order,
    feature_2_order=[0.0, 1.0, 2.0],
    # feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by # Associated Certificates",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    # with_labels=True
)

# %%
any_hp["new"].fillna("unknown", inplace=True)

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "new",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by New Installation",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    # with_labels=True
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "CONSTRUCTION_AGE_BAND",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.const_year_order_merged,
    plotting_colors="viridis",
    plot_title="HP Subsets split by Construction Age Band",
    # legend_loc="outside",
    figsize=(15, 5),
    width=0.9,
    # legend_loc='outside',
    with_labels=True,
)

# %%
# any_hp['FIRST_HP_MENTION'] = pd.to_datetime(any_hp['FIRST_HP_MENTION'])
any_hp["FIRST_HP_MENTION_YEAR"] = any_hp["FIRST_HP_MENTION"].dt.year
any_hp["FIRST_HP_MENTION_YEAR"].fillna(any_hp["HP_INSTALL_DATE"].dt.year, inplace=True)
# any_hp["FIRST_HP_MENTION_YEAR"] = any_hp["FIRST_HP_MENTION_YEAR"].astype('int')


easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "FIRST_HP_MENTION_YEAR",
    feature_1_order=feature_1_order,
    feature_2_order=[
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
    legend_loc="outside",
    plot_title="HP Subsets split by First HP Mention",
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
    plot_title="HP Subsets split by # Entries",
)

# %%

# any_hp['N_ENTRIES'] = any_hp['N_ENTRIES'].astype('str')
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "X",
    feature_1_order=feature_1_order,
    # feature_2_order=["1", "2", "3", "4"],
    plotting_colors="viridis",
    plot_title="HP Subsets split by # Entries",
)

# %%
any_hp["HP_INSTALL_YEAR"] = any_hp["HP_INSTALL_DATE"].dt.year
any_hp["HP_INSTALL_YEAR"].fillna(0, inplace=True)
any_hp["HP_INSTALL_YEAR"] = any_hp["HP_INSTALL_YEAR"].astype("int")


easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "HP_INSTALL_YEAR",
    feature_2_order=[
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
    feature_1_order=feature_1_order[1:],
    plotting_colors="viridis",
    plot_title="HP Subsetse split by HP Install Year",
    figsize=(15, 5),
    width=0.8,
    # legend_loc='outside',
    with_labels=True,
)

# %%

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "TRANSACTION_TYPE",
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
    feature_1_order=feature_1_order,
    plot_title="HP Subsets split by Transaction Type",
    # legend_loc="outside",
    figsize=(15, 5),
    width=0.9,
    # legend_loc='outside',
    with_labels=True,
)

# %%
any_hp.loc[
    any_hp["CONSTRUCTION_AGE_BAND"].isin(
        ["1991-1998", "1996-2002", "2003-2007", "2007 onwards", "unknown"]
    )
    & (any_hp["HP set"] == "EPC & MCS")
]["N_ENTRIES"].value_counts(dropna=False, normalize=True)

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

any_hp["YEAR_DIFF_INSTALL"] = any_hp["ENTRY_YEAR"] - any_hp["HP_INSTALL_YEAR"]
any_hp["YEAR_DIFF_INSTALL"].value_counts(dropna=False, normalize=True)

# %%

any_hp["DIFF"] = any_hp["INSPECTION_DATE"] - any_hp["HP_INSTALL_DATE"]
any_hp["DIFF"].value_counts(dropna=False, normalize=True)

# %%
any_hp["BLUB"] = any_hp["INSPECTION_DATE"] - any_hp["HP_INSTALL_DATE"] > timedelta(
    days=0
)
any_hp["DIFF BIGGER"] = (any_hp["DIFF"] > timedelta(days=356)) & (
    any_hp["INSPECTION_DATE"] > any_hp["HP_INSTALL_DATE"]
)
any_hp["DIFF 2 years+"] = (any_hp["DIFF"] > timedelta(days=356)) & (
    any_hp["DIFF"] < timedelta(days=712)
)

# %%
any_hp.loc[any_hp["HP set"] == "MCS all"]["BLUB"].value_counts() / any_hp.loc[
    any_hp["HP set"] == "MCS all"
].shape[0]

# %%
any_hp.loc[any_hp["HP set"] == "MCS all"]["DIFF BIGGER"].value_counts() / any_hp.loc[
    any_hp["HP set"] == "MCS all"
].shape[0]

# %%
any_hp.loc[any_hp["HP set"] == "EPC | MCS"]["DIFF BIGGER"].value_counts() / any_hp.loc[
    any_hp["HP set"] == "EPC | MCS"
].shape[0]

# %%
any_hp.loc[any_hp["HP set"] == "EPC & MCS"]["DIFF BIGGER"].value_counts() / any_hp.loc[
    any_hp["HP set"] == "EPC & MCS"
].shape[0]

# %%
any_hp.loc[any_hp["HP set"] == "EPC & MCS"]["DIFF BIGGER"].value_counts()

# %%
any_hp.loc[(any_hp["HP set"] == "EPC & MCS") & ~(any_hp["DIFF BIGGER"])][
    "TRANSACTION_TYPE"
].value_counts(normalize=True)

# %%
any_hp.loc[(any_hp["HP set"] == "EPC & MCS") & ~(any_hp["DIFF BIGGER"])][
    "TRANSACTION_TYPE"
].value_counts(normalize=True)

# %%
any_hp.loc[
    (any_hp["HP set"] == "EPC & MCS")
    & (any_hp["TRANSACTION_TYPE"] == "RHI application")
]["DIFF BIGGER"].value_counts(normalize=True)

# %%
any_hp["YEAR_DIFF_INSTALL"] = any_hp["ENTRY_YEAR"] - any_hp["HP_INSTALL_YEAR"]
any_hp.loc[any_hp["HP set"] == "MCS all"][
    "YEAR_DIFF_INSTALL"
].value_counts() / any_hp.loc[any_hp["HP set"] == "MCS all"].shape[0]

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
any_hp["YEAR_DIFF_INSTALL"] = any_hp["YEAR_DIFF_INSTALL"].astype("int")

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
# Get HEX ID to Local Authority Mapping
# hex_to_LA = data_agglomeration.map_hex_to_feature(epc_df, "LOCAL_AUTHORITY_LABEL")

la_df = pd.DataFrame(dedupl_epc_df.LOCAL_AUTHORITY_LABEL.unique())
la_df.columns = ["LOCAL_AUTHORITY_LABEL"]
data_by_year = []

for year in [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]:

    year_df = feature_engineering.filter_by_year(
        dedupl_epc_df, "BUILDING_ID", year, up_to=True
    ).copy()
    print(year_df.shape)
    year_df["HP_INSTALLED"] = year_df["HP_INSTALL_DATE"].dt.year <= year

    year_stamp = str(year) + "/01/01 00:00"
    year_agglo_df = data_agglomeration.get_agglomerated_features(
        year_df,
        feature=None,
        agglo_feature="LOCAL_AUTHORITY_LABEL",
        year_stamp=year_stamp,
    )
    # year_agglo_df = pd.merge(year_agglo_df,hex_df, on='hex_id', how='right')
    year_agglo_df = pd.merge(
        year_agglo_df, la_df, on="LOCAL_AUTHORITY_LABEL", how="right"
    )
    year_agglo_df["YEAR_STAMP"] = str(year) + "/01/01 00:00"

    year_agglo_df = pd.merge(hex_to_LA, year_agglo_df, on=["LOCAL_AUTHORITY_LABEL"])
    data_by_year.append(year_agglo_df)

time_data = pd.concat(data_by_year)

time_data["HP_PERC"] = time_data["HP_PERC"].fillna(0.0)
time_data["HP_CAT"] = time_data["HP_CAT"].fillna("0.0 %")


# %%
from keplergl import KeplerGl
from epc_data_analysis.config.kepler import kepler_config
from epc_data_analysis.config.kepler.kepler_config import (
    MAPS_CONFIG_PATH,
    MAPS_OUTPUT_PATH,
)


time_data["EPC_CAT"] = time_data["EPC_CAT"].fillna("C-D")
time_data["HP_PERCENTAGE"] = time_data["HP_CAT"]

version_tag = "HP_time_course"

config_file = MAPS_CONFIG_PATH + "{}_config.txt".format(version_tag)
config = kepler_config.get_config(config_file)

time_map = KeplerGl(height=500)  # , config=config)

time_map.add_data(
    data=time_data[
        [
            "HP_PERC",
            "HP_PERCENTAGE",
            "EPC_CAT",
            "hex_id",
            "LOCAL_AUTHORITY_LABEL",  # "CO2_EMISSIONS_CURRENT",
            "YEAR_STAMP",
        ]
    ],
    name="Year Data",
)

time_map

# %%
kepler_config.save_config(time_map, config_file)

time_map.save_to_html(file_name=MAPS_OUTPUT_PATH + "Time_Course_Heat_Pumps_final.html")

# %%
from epc_data_analysis.config.kepler import kepler_config as kepler

config = kepler.get_config(kepler.MAPS_CONFIG_PATH + "epc_mcs_comparison_short.txt")

epc_mcs_comp = KeplerGl(height=500, config=config)

epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "EPC & MCS"][["LONGITUDE", "LATITUDE"]],
    name="EPC & MCS",
)

epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "MCS only"][["LONGITUDE", "LATITUDE"]],
    name="MCS only",
)

epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "EPC only"][["LONGITUDE", "LATITUDE"]],
    name="EPC only",
)

epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "EPC all"][["LONGITUDE", "LATITUDE"]],
    name="EPC all",
)

epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "MCS all"][["LONGITUDE", "LATITUDE"]],
    name="MCS all",
)


epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "Conflict A"][["LONGITUDE", "LATITUDE"]],
    name="Conflict A",
)

epc_mcs_comp.add_data(
    data=any_hp.loc[any_hp["HP set"] == "Conflict B"][["LONGITUDE", "LATITUDE"]],
    name="Conflict B",
)
epc_mcs_comp

# %%
kepler.save_config(epc_mcs_comp, MAPS_CONFIG_PATH + "epc_mcs_comparison_dots.txt")

epc_mcs_comp.save_to_html(file_name=MAPS_OUTPUT_PATH + "EPC_vs_MCS_dots.html")

# %%
agglo_feature = ["LOCAL_AUTHORITY_LABEL", "hex_id"][0]

# %%
interval = 10
values = [
    [str(i) + "-" + str(i + interval) + "%"] * interval for i in range(0, 100, interval)
]

values = [item for sublist in values for item in sublist]
keys = [i / 100 for i in range(0, 100)]

cat_dict_10 = dict(zip(keys, values))

interval = 2
values = [
    [str(i) + "-" + str(i + interval) + "%"] * interval for i in range(0, 100, interval)
]

values = [item for sublist in values for item in sublist]
keys = [i / 100 for i in range(0, 100)]

cat_dict_1 = dict(zip(keys, values))


all_props = dedupl_epc_df.copy()
all_hps = dedupl_epc_df[~((~epc_hp_mention) & (~mcs_available))]

all_props_total = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    all_props, "TENURE", agglo_feature=agglo_feature
)
all_props_total.rename(columns={agglo_feature + "_TOTAL": "TOTAL"}, inplace=True)


all_hps_total = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    all_hps, "TENURE", agglo_feature=agglo_feature
)
all_hps_total.rename(columns={agglo_feature + "_TOTAL": "TOTAL"}, inplace=True)


all_datasets = collections.defaultdict(dict)

norm_dict = {"all properties": all_props_total, "all HP properties": all_hps_total}


for dataset_name in [
    "EPC & MCS",
    "MCS only",
    "EPC only",
    "MCS all",
    "EPC all",
    "Conflict A",
    "Conflict B",
]:

    for normset_name in ["all properties", "all HP properties"]:

        dataset = any_hp.loc[any_hp["HP set"] == dataset_name][
            ["LONGITUDE", "LATITUDE", "hex_id", "TENURE", "LOCAL_AUTHORITY_LABEL"]
        ]

        aggl_dataset = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
            dataset, "TENURE", agglo_feature=agglo_feature
        )

        if agglo_feature == "LOCAL_AUTHORITY_LABEL":

            aggl_dataset = pd.merge(
                aggl_dataset, populations, on=agglo_feature, how="right"
            )

            aggl_dataset["perc"] = round(
                (
                    aggl_dataset[agglo_feature + "_TOTAL"] / aggl_dataset["POPULATION"]
                ).astype(float),
                2,
            )

        else:

            norm_set = norm_dict[normset_name]
            aggl_dataset = pd.merge(
                aggl_dataset, norm_set, on=agglo_feature, how="right"
            )

            aggl_dataset["perc"] = round(
                (aggl_dataset[agglo_feature + "_TOTAL"] / aggl_dataset["TOTAL"]).astype(
                    float
                ),
                2,
            )

        aggl_dataset["perc"].fillna(0.0, inplace=True)

        # aggl_dataset["perc"] = aggl_dataset["perc"].map(cat_dict_1)

        if agglo_feature == "LOCAL_AUTHORITY_LABEL":
            aggl_dataset = pd.merge(
                aggl_dataset, hex_to_LA, on=agglo_feature, how="right"
            )

        all_datasets[dataset_name][normset_name] = aggl_dataset


# %%
norm_set = "all properties"

# %%
from epc_data_analysis.config.kepler import kepler_config as kepler

config_file_tag = (
    "norm_by_HP_prop" if norm_set == "all HP properties" else "norm_by_population"
)

# config_file_tag = "norm_by_HP_prop"

if agglo_feature == "LOCAL_AUTHORITY_LABEL":
    config_file_tag += "_LA"

config = kepler.get_config(
    kepler.MAPS_CONFIG_PATH + "epc_mcs_comparison_{}.txt".format(config_file_tag)
)


epc_mcs_comp = KeplerGl(height=500, config=config)

epc_mcs_comp.add_data(
    data=all_datasets["EPC & MCS"][norm_set][
        ["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="EPC & MCS",
)

epc_mcs_comp.add_data(
    data=all_datasets["MCS only"][norm_set][
        ["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="MCS only",
)

epc_mcs_comp.add_data(
    data=all_datasets["EPC only"][norm_set][
        ["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="EPC only",
)

epc_mcs_comp.add_data(
    data=all_datasets["EPC all"][norm_set][["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]],
    name="EPC (all)",
)


epc_mcs_comp.add_data(
    data=all_datasets["MCS all"][norm_set][["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]],
    name="MCS (all)",
)

epc_mcs_comp.add_data(
    data=all_datasets["Conflict A"][norm_set][
        ["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="Conflict A",
)

epc_mcs_comp.add_data(
    data=all_datasets["Conflict B"][norm_set][
        ["hex_id", "perc", "LOCAL_AUTHORITY_LABEL"]
    ],
    name="Conflict B",
)


display(epc_mcs_comp)

# %%
"epc_mcs_comparison{}.txt".format(config_file_tag)

config_file_tag = "norm_by_population_LA"

# %%
kepler.save_config(
    epc_mcs_comp, MAPS_CONFIG_PATH + "epc_mcs_comparison_{}.txt".format(config_file_tag)
)

epc_mcs_comp.save_to_html(
    file_name=MAPS_OUTPUT_PATH + "EPC_vs_MCS_{}.html".format(config_file_tag)
)

# %%
import numpy as np

latest_wales = pd.read_csv("latest_wales")

# %%
list(latest_wales.columns)

# %%
# Get HEX ID to Local Authority Mapping
# hex_to_LA = data_agglomeration.map_hex_to_feature(epc_df, "LOCAL_AUTHORITY_LABEL")

latest_wales = pd.read_csv("latest_wales")

latest_wales = feature_engineering.get_coordinates(latest_wales)
latest_wales = data_agglomeration.add_hex_id(latest_wales, resolution=7.5)

hex_to_LA = data_agglomeration.map_hex_to_feature(latest_wales, "LOCAL_AUTHORITY_LABEL")

la_df = pd.DataFrame(latest_wales.LOCAL_AUTHORITY_LABEL.unique())
la_df.columns = ["LOCAL_AUTHORITY_LABEL"]
all_data = []

all_data.append(latest_wales[["LOCAL_AUTHORITY_LABEL", "hex_id"]])

for eff in [
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
]:

    agglo_df = data_agglomeration.get_agglomerated_features(
        latest_wales, feature=eff + "_DIFFS", agglo_feature="LOCAL_AUTHORITY_LABEL"
    )
    agglo_df = pd.merge(agglo_df, la_df, on="LOCAL_AUTHORITY_LABEL", how="right")

    agglo_df.rename(
        columns={
            1.0: "1.0_" + eff,
            2.0: "2.0_" + eff,
            3.0: "3.0_" + eff,
            4.0: "4.0_" + eff,
            5.0: "5.0_" + eff,
            0.0: "0.0_" + eff,
        },
        inplace=True,
    )

    cols = [
        "1.0_" + eff,
        "2.0_" + eff,
        "3.0_" + eff,
        "4.0_" + eff,
        "5.0_" + eff,
        "0.0_" + eff,
    ]
    cols = [c for c in cols if c in agglo_df.columns]
    all_data.append(agglo_df[cols])


agglo_df = pd.concat(all_data, axis=1)


agglo_df = pd.merge(hex_to_LA, agglo_df, on=["LOCAL_AUTHORITY_LABEL"])
agglo_df.columns

# %%
agglo_f = "hex_id"

latest_wales = pd.read_csv("latest_wales")
latest_wales = feature_engineering.get_coordinates(latest_wales)
latest_wales = data_agglomeration.add_hex_id(latest_wales, resolution=7.5)

hex_to_LA = data_agglomeration.map_hex_to_feature(latest_wales, "LOCAL_AUTHORITY_LABEL")

la_df = pd.DataFrame(latest_wales.LOCAL_AUTHORITY_LABEL.unique())
la_df.columns = ["LOCAL_AUTHORITY_LABEL"]

agglo_df = data_agglomeration.get_agglomerated_features(
    latest_wales, feature=None, agglo_feature=agglo_f
)
# agglo_df = pd.merge(agglo_df,la_df, on='LOCAL_AUTHORITY_LABEL', how='right')
# = pd.merge(hex_to_LA, agglo_df, on=["LOCAL_AUTHORITY_LABEL"])

for eff in [
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
]:

    eff_dict = latest_wales.groupby([agglo_f]).mean()[eff + "_DIFF"].to_dict()
    agglo_df[eff + "_DIFF"] = agglo_df[agglo_f].map(eff_dict)

agglo_df = feature_engineering.get_coordinates(agglo_df)
agglo_df.head()

# %%
from epc_data_analysis.config.kepler import kepler_config as kepler

# config = kepler.get_config(
# kepler.MAPS_CONFIG_PATH + "epc_mcs_comparison_{}.txt".format(config_file_tag))


epc_mcs_comp = KeplerGl(height=500)

for eff in [
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
]:

    epc_mcs_comp.add_data(
        data=agglo_df.loc[agglo_df[eff + "_DIFF"] > 0.0][
            ["hex_id", agglo_f, eff + "_DIFF"]
        ],
        name=eff,
    )


display(epc_mcs_comp)

# %%
epc_mcs_comp.save_to_html(file_name=MAPS_OUTPUT_PATH + "Updates.html")

# %%
latest_wales = pd.read_csv("latest_wales")
latest_wales = feature_engineering.get_coordinates(latest_wales)
latest_wales = data_agglomeration.add_hex_id(latest_wales, resolution=7.5)


from epc_data_analysis.config.kepler import kepler_config as kepler

# config = kepler.get_config(
# kepler.MAPS_CONFIG_PATH + "epc_mcs_comparison_{}.txt".format(config_file_tag))


epc_mcs_comp = KeplerGl(height=500)

for eff in [
    "MAINHEAT_ENERGY_EFF",
    "HOT_WATER_ENERGY_EFF",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_ENERGY_EFF",
    "WALLS_ENERGY_EFF",
    "ROOF_ENERGY_EFF",
    "LIGHTING_ENERGY_EFF",
]:

    epc_mcs_comp.add_data(
        data=latest_wales.loc[latest_wales[eff + "_DIFF"] > 0.0][
            ["LONGITUDE", "LATITUDE", eff + "_DIFF"]
        ],
        name=eff,
    )


display(epc_mcs_comp)

# %%
