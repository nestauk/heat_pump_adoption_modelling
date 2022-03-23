# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
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
epc_df = pd.read_csv(
    str(PROJECT_DIR) + "/inputs/epc_df_complete_preprocessed.csv",
    parse_dates=["INSPECTION_DATE", "HP_INSTALL_DATE", "FIRST_HP_MENTION"],
)

# %%
dedupl_epc_df = feature_engineering.filter_by_year(
    epc_df, "BUILDING_ID", year=2021, up_to=True, selection="latest entry"
)
print(dedupl_epc_df.shape)
print(epc_df.shape)
print(dedupl_epc_df.columns)


# %%
# dedupl_epc_df.loc[~(dedupl_epc_df["FIRST_HP_MENTION"].isna())].shape

# %%
~(dedupl_epc_df["FIRST_HP_MENTION"].isna())

# %%
dedupl_epc_df.columns

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
dedupl_epc_df["EPC HP entry before MCS"].value_counts(dropna=False)


# %%
epc_entry_before_mcs = dedupl_epc_df.loc[dedupl_epc_df["EPC HP entry before MCS"]]

epc_entry_before_mcs["> 1 YEAR DIFF"] = (
    epc_entry_before_mcs["HP_INSTALL_DATE"] - epc_entry_before_mcs["INSPECTION_DATE"]
) > timedelta(days=356)

epc_entry_before_mcs["> 1 YEAR DIFF"].value_counts(normalize=True)

# %%
dedupl_epc_df["No EPC HP entry after MCS"].value_counts(dropna=False)

# %%
no_epc_entry_after_mcs = dedupl_epc_df.loc[dedupl_epc_df["No EPC HP entry after MCS"]]

no_epc_entry_after_mcs["> 1 YEAR DIFF"] = (
    no_epc_entry_after_mcs["INSPECTION_DATE"]
    - no_epc_entry_after_mcs["HP_INSTALL_DATE"]
) > timedelta(days=365)

no_epc_entry_after_mcs["> 1 YEAR DIFF"].value_counts(normalize=True)

# %%
no_epc_entry_after_mcs["HP_LOST"].value_counts(normalize=True)

# %%
no_epc_entry_after_mcs.loc[no_epc_entry_after_mcs["> 1 YEAR DIFF"]][
    "HP_LOST"
].value_counts(normalize=True)

# %%
epc_entry_before_mcs["HP_LOST"].value_counts(normalize=True)

# %%
epc_entry_before_mcs.loc[epc_entry_before_mcs["> 1 YEAR DIFF"]]["HP_LOST"].value_counts(
    normalize=True
)

# %%
dedupl_epc_df = feature_engineering.get_coordinates(dedupl_epc_df)
dedupl_epc_df = data_agglomeration.add_hex_id(dedupl_epc_df, resolution=7.5)

# %%
# none = dedupl_epc_df.loc[~epc_hp_mention & ~mcs_available]
# none['HP set'] = 'none'

# either = dedupl_epc_df.loc[epc_hp_mention | mcs_available]
# either['HP set'] = 'EPC | MCS'

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
        both,
        epc_only,
        mcs_only,
        epc_all,
        mcs_all,
        no_epc_after_mcs,
        epc_hp_before_mcs,
        conflict_a_short,
        conflict_b_short,
    ]
)
feature_1_order = [
    "EPC & MCS",
    "EPC all",
    "EPC only",
    "MCS all",
    "MCS only",
    "Conflict A",
    "Conflict B",
]

# feature_1_order = ['EPC & MCS', 'EPC all', 'EPC only', 'MCS all', 'MCS only']
# feature_1_order = [ 'Conflict A',  'Conflict A > 1 year',  'Conflict B',  'Conflict B > 1 year']
any_hp["HP set"].value_counts(dropna=False)

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
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "NOT_THEN_HP",
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
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "CURRENT_ENERGY_RATING",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.rating_order,
    plotting_colors="RdYlGn_r",
    plot_title="HP Subsets split by EPC Rating",
)

# %%
any_hp["ENTRY_YEAR"] = any_hp["INSPECTION_DATE"].dt.year

easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "ENTRY_YEAR",
    feature_1_order=feature_1_order,
    feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    legend_loc="outside",
    plot_title="HP Subsets split by Entry Year",
)

# %%
any_hp.loc[any_hp["HP set"] == "EPC & MCS"]["IMD Decile"].mean()

# %%
print(any_hp.loc[any_hp["HP set"] == "EPC all"]["IMD Decile"].mean())
print(any_hp.loc[any_hp["HP set"] == "EPC only"]["IMD Decile"].mean())
print(any_hp.loc[any_hp["HP set"] == "MCS all"]["IMD Decile"].mean())
print(any_hp.loc[any_hp["HP set"] == "MCS only"]["IMD Decile"].mean())
print(any_hp.loc[any_hp["HP set"] == "Conflict A"]["IMD Decile"].mean())
print(any_hp.loc[any_hp["HP set"] == "Conflict B"]["IMD Decile"].mean())


# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "HP set",
    "IMD Decile",
    feature_1_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="RdYlGn",
    legend_loc="outside",
    plot_title="HP Subsets split by IMD Decile",
)

# %%
easy_plotting.plot_subcats_by_other_subcats(
    any_hp,
    "Country",
    "HP set",
    feature_2_order=feature_1_order,
    # feature_2_order=feature_settings.year_order,
    plotting_colors="viridis",
    plot_title="HP Subsets split by Country",
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
    legend_loc="outside",
)

# %%
# any_hp['FIRST_HP_MENTION'] = pd.to_datetime(any_hp['FIRST_HP_MENTION'])
any_hp["FIRST_HP_MENTION_YEAR"] = any_hp["FIRST_HP_MENTION"].dt.year
any_hp["FIRST_HP_MENTION_YEAR"].fillna(any_hp["HP_INSTALL_DATE"].dt.year)

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
    "POSTTOWN",
    feature_1_order=feature_1_order,
    # feature_2_order=["1", "2", "3", "4"],
    plotting_colors="viridis",
    plot_title="HP Subsets split by # Entries",
)

# %%
any_hp["HP_INSTALL_YEAR"] = any_hp["HP_INSTALL_DATE"].dt.year


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
    feature_1_order=feature_1_order,
    plotting_colors="viridis",
    plot_title="HP Subsetse split by HP Install Year",
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
    legend_loc="outside",
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
epc_mcs_df.loc[~epc_mcs_df["epc_address"].isna()][
    "BUILDING_REFERENCE_NUMBER"
].unique().shape

# %%

# %%

# any_hp = feature_engineering.get_coordinates(any_hp)

from epc_data_analysis.config.kepler import kepler_config as kepler

config = kepler.get_config(kepler.MAPS_CONFIG_PATH + "epc_mcs_comparison_short.txt")

epc_mcs_comp = KeplerGl(height=500)  # , config=config)

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
all_props = dedupl_epc_df.copy()
all_hps = dedupl_epc_df[~((~epc_hp_mention) & (~mcs_available))]

all_props_total = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    all_props, "TENURE", agglo_feature="hex_id"
)
all_props_total.rename(columns={"hex_id_TOTAL": "TOTAL"}, inplace=True)


all_hps_total = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
    all_hps, "TENURE", agglo_feature="hex_id"
)
all_hps_total.rename(columns={"hex_id_TOTAL": "TOTAL"}, inplace=True)


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
            ["LONGITUDE", "LATITUDE", "hex_id", "TENURE"]
        ]

        norm_set = norm_dict[normset_name]

        aggl_dataset = data_agglomeration.get_cat_distr_grouped_by_agglo_f(
            dataset, "TENURE", agglo_feature="hex_id"
        )

        aggl_dataset = pd.merge(aggl_dataset, norm_set, on="hex_id", how="right")

        aggl_dataset["perc"] = round(
            (aggl_dataset["hex_id_TOTAL"] / aggl_dataset["TOTAL"]).astype(float), 2
        )

        aggl_dataset["perc"].fillna(0.0, inplace=True)
        all_datasets[dataset_name][normset_name] = aggl_dataset


# %%
norm_set = "all properties"

# %%
from epc_data_analysis.config.kepler import kepler_config as kepler

config_file_tag = "" if norm_set == "all HP properties" else "_norm_by_pop"

config = kepler.get_config(
    kepler.MAPS_CONFIG_PATH + "epc_mcs_comparison{}.txt".format(config_file_tag)
)

epc_mcs_comp = KeplerGl(height=500, config=config)

epc_mcs_comp.add_data(
    data=all_datasets["EPC & MCS"][norm_set][["hex_id", "perc"]],
    name="EPC & MCS",
)

epc_mcs_comp.add_data(
    data=all_datasets["MCS only"][norm_set][["hex_id", "perc"]],
    name="MCS only",
)

epc_mcs_comp.add_data(
    data=all_datasets["EPC only"][norm_set][["hex_id", "perc"]],
    name="EPC only",
)

epc_mcs_comp.add_data(
    data=all_datasets["EPC all"][norm_set][["hex_id", "perc"]],
    name="EPC (all)",
)


epc_mcs_comp.add_data(
    data=all_datasets["MCS all"][norm_set][["hex_id", "perc"]],
    name="MCS (all)",
)

epc_mcs_comp.add_data(
    data=all_datasets["Conflict A"][norm_set][["hex_id", "perc"]],
    name="Conflict A",
)

epc_mcs_comp.add_data(
    data=all_datasets["Conflict B"][norm_set][["hex_id", "perc"]],
    name="Conflict B",
)


display(epc_mcs_comp)

# %%
kepler.save_config(
    epc_mcs_comp, MAPS_CONFIG_PATH + "epc_mcs_comparison_norm_by_pop.txt"
)

epc_mcs_comp.save_to_html(file_name=MAPS_OUTPUT_PATH + "EPC_vs_MCS_norm_by_pop.html")

# %%
