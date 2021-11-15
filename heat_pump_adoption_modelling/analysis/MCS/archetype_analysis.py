import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.MCS.load_mcs import load_inflation
from heat_pump_adoption_modelling.pipeline.MCS.mcs_epc_joining import join_mcs_epc_data

merged = join_mcs_epc_data(all_records=True)


inflation = load_inflation()

merged["TOTAL_FLOOR_AREA"] = merged["TOTAL_FLOOR_AREA"].astype(float)

merged = merged.merge(inflation, how="left", on="year")
merged["inflated_cost"] = merged["cost"] * merged["multiplier"]


test_epcs = pd.read_csv("inputs/domestic-E06000026-Plymouth/certificates.csv")


ages_dict = {
    "England and Wales: before 1900": False,
    "1900-1929": False,
    "1930-1949": False,
    "1950-1966": True,
    "1965-1975": True,
    "1976-1983": True,
    "1983-1991": True,
    "1991-1998": True,
    "1996-2002": True,
    "2003-2007": True,
    "2007 onwards": True,
    "unknown": pd.NA,
    np.nan: pd.NA,
    None: pd.NA,
}

merged["built_post_1950"] = merged["CONSTRUCTION_AGE_BAND"].apply(
    lambda x: ages_dict[x]
)

# temporary fix for weird efficiencies
for column in ["WALLS_ENERGY_EFF", "FLOOR_ENERGY_EFF"]:
    merged.loc[["|" in str(field) for field in merged[column]], column] = np.nan
    merged[column] = merged[column].replace("unknown", np.nan)

merged["INSPECTION_DATE"] = pd.to_datetime(
    merged["INSPECTION_DATE"].replace("unknown", np.nan)
)

merged["epc_after_mcs"] = merged["date"] < merged["INSPECTION_DATE"]

merged["poor_wall_post_inst"] = merged["WALLS_ENERGY_EFF"].isin(
    ["Poor", "Very Poor"]
) & (merged["epc_after_mcs"])
merged["poor_floor_post_inst"] = merged["FLOOR_ENERGY_EFF"].isin(
    ["Poor", "Very Poor"]
) & (merged["epc_after_mcs"])

merged["good_wall_pre_inst"] = merged["WALLS_ENERGY_EFF"].isin(
    ["Good", "Very Good"]
) & (~merged["epc_after_mcs"])
merged["good_floor_pre_inst"] = merged["FLOOR_ENERGY_EFF"].isin(
    ["Good", "Very Good"]
) & (~merged["epc_after_mcs"])

# merged["ave_wall_pre_inst"] = (merged["WALLS_ENERGY_EFF"] == "Average") & (~merged["epc_after_mcs"])
# merged["ave_wall_post_inst"] = (merged["WALLS_ENERGY_EFF"] == "Average") & (merged["epc_after_mcs"])

# merged["ave_floor_pre_inst"] = (merged["FLOOR_ENERGY_EFF"] == "Average") & (~merged["epc_after_mcs"])
# merged["ave_floor_post_inst"] = (merged["FLOOR_ENERGY_EFF"] == "Average") & (merged["epc_after_mcs"])


merged["any_poorwallpost"] = merged.groupby("index_x")["poor_wall_post_inst"].transform(
    lambda x: x.any()
)
merged["any_poorfloorpost"] = merged.groupby("index_x")[
    "poor_floor_post_inst"
].transform(lambda x: x.any())

merged["any_goodwallpre"] = merged.groupby("index_x")["good_wall_pre_inst"].transform(
    lambda x: x.any()
)
merged["any_goodfloorpre"] = merged.groupby("index_x")["good_floor_pre_inst"].transform(
    lambda x: x.any()
)

# merged["any_avewallpre"] = merged.groupby("index_x")["ave_wall_pre_inst"].transform(lambda x: x.any())
# merged["any_avewallpost"] = merged.groupby("index_x")["ave_wall_post_inst"].transform(lambda x: x.any())

# merged["any_avefloorpre"] = merged.groupby("index_x")["ave_floor_pre_inst"].transform(lambda x: x.any())
# merged["any_avefloorpost"] = merged.groupby("index_x")["ave_floor_post_inst"].transform(lambda x: x.any())

merged = merged.groupby("index_x").first()


# cavity -> average / good / very good
# solid -> poor / very poor


def bool_not(series):
    result = pd.Series(np.repeat(pd.NA, len(series)))
    result.loc[series == True] = False
    result.loc[series == False] = True
    return result


archetypes = {
    # flat, any wall, any floor, any area, post-1950
    "archetype_1": ((merged.PROPERTY_TYPE == "Flat") & (merged.built_post_1950)),
    # flat, any wall, any floor, any area, pre-1950
    "archetype_2": (
        (merged.PROPERTY_TYPE == "Flat") & bool_not(merged.built_post_1950)
    ),
    # bungalow, good wall, [any floor], any area, post-1950
    "archetype_3": (
        (merged.PROPERTY_TYPE == "Bungalow")
        & (merged.any_goodwallpre)
        & (merged.built_post_1950)
    ),
    # bungalow, bad wall, [any floor], <=85 area, pre-1950
    "archetype_4": (
        (merged.PROPERTY_TYPE == "Bungalow")
        & (merged.any_poorwallpost)
        & (merged.TOTAL_FLOOR_AREA <= 85)
        & bool_not(merged.built_post_1950)
    ),
    # mid-terrace or maisonette, any wall, any floor, any area, any age
    "archetype_5": (
        (merged.PROPERTY_TYPE == "Maisonette")
        | ((merged.PROPERTY_TYPE == "House") & (merged.BUILT_FORM == "Mid-Terrace"))
    ),
    # semi-detached or end-terrace, good wall, [any floor], any area, post-1950
    "archetype_6": (
        (merged.PROPERTY_TYPE == "House")
        & (merged.BUILT_FORM.isin(["Semi-Detached", "End-Terrace"]))
        & (merged.any_goodwallpre)
        & (merged.built_post_1950)
    ),
    # detached, good wall, [any floor], any area, post-1950
    "archetype_7": (
        (merged.PROPERTY_TYPE == "House")
        & (merged.BUILT_FORM == "Detached")
        & (merged.any_goodwallpre)
        & (merged.built_post_1950)
    ),
    # detached, bad wall, [any floor], any area, pre-1950
    "archetype_8": (
        (merged.PROPERTY_TYPE == "House")
        & (merged.BUILT_FORM == "Detached")
        & (merged.any_poorwallpost)
        & bool_not(merged.built_post_1950)
    ),
    # semi-detached or end-terrace, bad wall, [any floor], any area, pre-1950
    "archetype_9": (
        (merged.PROPERTY_TYPE == "House")
        & (merged.BUILT_FORM.isin(["Semi-Detached", "End-Terrace"]))
        & (merged.any_poorwallpost)
        & bool_not(merged.built_post_1950)
    ),
    # bungalow, bad wall, [any floor], >85 area, pre-1950
    "archetype_10": (
        (merged.PROPERTY_TYPE == "Bungalow")
        & (merged.any_poorwallpost)
        & (merged.TOTAL_FLOOR_AREA > 85)
        & bool_not(merged.built_post_1950)
    ),
}

for key, value in archetypes.items():
    merged[key] = value


for i in archetypes.values():
    print(i.sum())


exp_dict = {
    1: [5, 4],
    2: [8, 6],
    3: [8, 6],
    4: [10, 8],
    5: [9, 7],
    6: [10, 8],
    7: [12, 10],
    8: [14, 11],
    9: [12, 10],
    10: [12, 10],
}


def plot_archetype_scatter(data, i, arc_dict, arc_exp):
    for key, value in arc_dict.items():
        data[key] = value
    colname = "archetype_" + str(i)
    archetype_data = data.loc[
        data[colname], ["capacity", "inflated_cost", "tech_type"]
    ].dropna(how="any")
    ad_ashp = archetype_data.loc[archetype_data.tech_type == "Air Source Heat Pump"]
    ad_gshp = archetype_data.loc[
        archetype_data.tech_type == "Ground/Water Source Heat Pump"
    ]

    fig, ax = plt.subplots()

    ax.set_xlim(0, 30)
    ax.set_ylim(0, 50000)
    ax.set_xlabel("Capacity (kW)")
    ax.set_ylabel("Cost (£2021)")
    ax.set_title(colname + " (n = " + str(archetype_data.shape[0]) + ")")

    ax.scatter(ad_ashp.capacity, ad_ashp.inflated_cost, s=3, c="blue", alpha=0.1)
    ax.scatter(
        ad_ashp.capacity.median(),
        ad_ashp.inflated_cost.median(),
        s=20,
        c="blue",
        edgecolors="red",
    )
    ax.scatter([], [], s=10, c="blue", label="ASHP (n = " + str(ad_ashp.shape[0]) + ")")

    ax.scatter(ad_gshp.capacity, ad_gshp.inflated_cost, s=3, c="orange", alpha=0.1)
    ax.scatter(
        ad_gshp.capacity.median(),
        ad_gshp.inflated_cost.median(),
        s=20,
        c="orange",
        edgecolors="red",
    )
    ax.scatter(
        [], [], s=10, c="orange", label="GSHP (n = " + str(ad_gshp.shape[0]) + ")"
    )

    ax.axvline(arc_exp[i][0], linestyle="--", color="blue")
    ax.axvline(arc_exp[i][1], linestyle="--", color="orange")

    ax.legend(loc="upper right")

    plt.savefig("outputs/figures/archetypes/" + colname + ".png")


for i in range(1, 11):
    plot_archetype_scatter(merged, i, archetypes, exp_dict)


merged_nona = merged[["capacity", "cost"]].dropna(how="any")

fig, ax = plt.subplots()
ax.set_ylim(0, 100000)
ax.scatter(merged_nona.capacity, merged_nona.cost, s=3, alpha=0.1)


merged_nona_2 = merged[["TOTAL_FLOOR_AREA", "cost"]].dropna(how="any")

fig, ax = plt.subplots()
ax.set_ylim(0, 100000)
ax.scatter(merged_nona_2.TOTAL_FLOOR_AREA, merged_nona_2.cost, s=3, alpha=0.1)


fig, ax = plt.subplots()

archetype_1_data = merged.loc[merged.archetype_1, ["capacity", "inflated_cost"]].dropna(
    how="any"
)
archetype_2_data = merged.loc[merged.archetype_2, ["capacity", "inflated_cost"]].dropna(
    how="any"
)
archetype_3_data = merged.loc[merged.archetype_3, ["capacity", "inflated_cost"]].dropna(
    how="any"
)
archetype_9_data = merged.loc[merged.archetype_9, ["capacity", "inflated_cost"]].dropna(
    how="any"
)

ax.set_xlim(0, 30)
ax.set_ylim(0, 30000)

ax.scatter(
    archetype_1_data.capacity, archetype_1_data.inflated_cost, s=3, c="blue", alpha=0.1
)
ax.scatter(
    archetype_2_data.capacity,
    archetype_2_data.inflated_cost,
    s=3,
    c="orange",
    alpha=0.3,
)
ax.scatter(
    archetype_3_data.capacity, archetype_3_data.inflated_cost, s=3, c="green", alpha=0.1
)
ax.scatter(
    archetype_9_data.capacity,
    archetype_9_data.inflated_cost,
    s=3,
    c="purple",
    alpha=0.1,
)

plt.clf()


# new archetypes
def new_archetypes(df):
    new_archetypes = {
        "archetype_1": (df.PROPERTY_TYPE == "Flat") & (df.built_post_1950),
        "archetype_2": (df.PROPERTY_TYPE == "Flat") & bool_not(df.built_post_1950),
        "archetype_3": (df.PROPERTY_TYPE == "Bungalow") & (df.built_post_1950),
        "archetype_4": (
            (df.PROPERTY_TYPE == "Maisonette")
            | (
                (df.PROPERTY_TYPE == "House")
                & (df.BUILT_FORM.isin(["Semi-Detached", "Mid-Terrace", "End-Terrace"]))
            )  # enclosed??
        )
        & (df.built_post_1950),
        "archetype_5": (
            (df.PROPERTY_TYPE == "House")
            & (df.BUILT_FORM == "Detached")
            & (df.built_post_1950)
        ),
        "archetype_6": (
            (df.PROPERTY_TYPE == "Maisonette")
            | (
                (df.PROPERTY_TYPE == "House")
                & (df.BUILT_FORM.isin(["Semi-Detached", "Mid-Terrace", "End-Terrace"]))
            )
        )
        & bool_not(df.built_post_1950),
        "archetype_7": (df.PROPERTY_TYPE == "Bungalow") & bool_not(df.built_post_1950),
        "archetype_8": (
            (df.PROPERTY_TYPE == "House")
            & (df.BUILT_FORM == "Detached")
            & bool_not(df.built_post_1950)
        ),
    }
    return new_archetypes


new_exp_dict = {
    1: [5, 4],
    2: [8, 6],
    3: [8, 6],
    4: [10, 8],
    5: [12, 10],
    6: [12, 10],
    7: [12, 10],
    8: [14, 11],
}


for i in range(1, 9):
    plot_archetype_scatter(merged, i, new_archetypes, new_exp_dict)

merged_recent = merged.loc[merged.year >= 2019].reset_index(drop=True)
new_dict = new_archetypes(merged_recent)
for i in range(1, 9):
    plot_archetype_scatter(merged_recent, i, new_dict, new_exp_dict)

for key, value in new_dict.items():
    merged_recent[key] = value

archetype_cost_quartiles = pd.DataFrame()
archetype_capacity_quartiles = pd.DataFrame()

for archetype in new_dict.keys():
    filtered_df = merged_recent.loc[merged_recent[archetype]]
    cost_quantiles = filtered_df.inflated_cost.quantile([0.25, 0.5, 0.75])
    cost_quantiles.name = archetype
    capacity_quantiles = filtered_df.capacity.quantile([0.25, 0.5, 0.75])
    capacity_quantiles.name = archetype
    archetype_cost_quartiles = archetype_cost_quartiles.append(cost_quantiles)
    archetype_capacity_quartiles = archetype_capacity_quartiles.append(
        capacity_quantiles
    )


# ASHP cost

ashp_cost_data_list = []
for archetype in new_dict.keys():
    filtered_df = merged_recent.loc[
        merged_recent[archetype] & (merged_recent.tech_type == "Air Source Heat Pump")
    ]
    costs = list(filtered_df.inflated_cost.dropna())
    ashp_cost_data_list.append(costs)

ashp_cost_data_list = ashp_cost_data_list[::-1]

fig, ax = plt.subplots()
ax.set_xlim(0, 30000)
ax.set_xlabel("Inflation-adjusted cost of installation (£2021)")
ax.set_ylabel("Property type")
ax.set_title("ASHP installation costs by property type (2019-21)")
ax.boxplot(
    ashp_cost_data_list,
    vert=False,
    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "alpha": 0.3,
    },
)
ytickNames = plt.setp(
    ax,
    yticklabels=[
        "Post-1950 flats",
        "Pre-1950 flats",
        "Post-1950 bungalows",
        "Post-1950 semi-detached,\nterraced and maisonettes",
        "Post-1950 detached",
        "Pre-1950 semi-detached,\nterraced and maisonettes",
        "Pre-1950 bungalows",
        "Pre-1950 detached",
    ][::-1],
)
plt.tight_layout()
plt.savefig("outputs/figures/ashp_cost_archetype_boxplots.png")


# ASHP capacity
ashp_capacity_data_list = []
for archetype in new_dict.keys():
    filtered_df = merged_recent.loc[
        merged_recent[archetype] & (merged_recent.tech_type == "Air Source Heat Pump")
    ]
    capacities = list(filtered_df.capacity.dropna())
    ashp_capacity_data_list.append(capacities)

ashp_capacity_data_list = ashp_capacity_data_list[::-1]

fig, ax = plt.subplots()
ax.set_xlim(0, 30)
ax.set_xlabel("ASHP capacity")
ax.set_ylabel("Property type")
ax.set_title("ASHP capacities by property type (2019-21)")
ax.boxplot(
    ashp_capacity_data_list,
    vert=False,
    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "alpha": 0.3,
    },
)
ytickNames = plt.setp(
    ax,
    yticklabels=[
        "Post-1950 flats",
        "Pre-1950 flats",
        "Post-1950 bungalows",
        "Post-1950 semi-detached,\nterraced and maisonettes",
        "Post-1950 detached",
        "Pre-1950 semi-detached,\nterraced and maisonettes",
        "Pre-1950 bungalows",
        "Pre-1950 detached",
    ][::-1],
)
plt.tight_layout()
plt.savefig("outputs/figures/ashp_capacity_archetype_boxplots.png")

# GSHP cost

gshp_cost_data_list = []
for archetype in ["archetype_3", "archetype_5", "archetype_8"]:
    filtered_df = merged_recent.loc[
        merged_recent[archetype]
        & (merged_recent.tech_type == "Ground/Water Source Heat Pump")
    ]
    costs = list(filtered_df.inflated_cost.dropna())
    gshp_cost_data_list.append(costs)

gshp_cost_data_list = gshp_cost_data_list[::-1]

fig, ax = plt.subplots()
ax.set_xlim(0, 80000)
ax.set_xlabel("Inflation-adjusted cost of installation (£2021)")
ax.set_ylabel("Property type")
ax.set_title("G/WSHP installation costs by property type (2019-21)")
ax.boxplot(
    gshp_cost_data_list,
    vert=False,
    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "alpha": 0.3,
    },
)
ytickNames = plt.setp(
    ax,
    yticklabels=["Post-1950 bungalows", "Post-1950 detached", "Pre-1950 detached"][
        ::-1
    ],
)
plt.tight_layout()
plt.savefig("outputs/figures/gshp_cost_archetype_boxplots.png")


# GSHP capacity

gshp_capacity_data_list = []
for archetype in ["archetype_3", "archetype_5", "archetype_8"]:
    filtered_df = merged_recent.loc[
        merged_recent[archetype]
        & (merged_recent.tech_type == "Ground/Water Source Heat Pump")
    ]
    capacities = list(filtered_df.capacity.dropna())
    gshp_capacity_data_list.append(capacities)

gshp_capacity_data_list = gshp_capacity_data_list[::-1]

fig, ax = plt.subplots()
ax.set_xlim(0, 50)
ax.set_xlabel("G/WSHP capacity")
ax.set_ylabel("Property type")
ax.set_title("G/WSHP capacities by property type (2019-21)")
ax.boxplot(
    gshp_capacity_data_list,
    vert=False,
    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "alpha": 0.3,
    },
)
ytickNames = plt.setp(
    ax,
    yticklabels=["Post-1950 bungalows", "Post-1950 detached", "Pre-1950 detached"][
        ::-1
    ],
)
plt.tight_layout()
plt.savefig("outputs/figures/gshp_capacity_archetype_boxplots.png")


# overall scatter by floor area
fig, ax = plt.subplots()
merged_recent_ashp = merged_recent.loc[
    merged_recent.tech_type == "Air Source Heat Pump"
]
ax.set_xlim(0, 400)
ax.set_ylim(0, 30000)
ax.set_xlabel("Property floor area (m$^2$)")
ax.set_ylabel("Inflation-adjusted cost of installation (£2021)")
ax.set_title("ASHP installation costs against floor area (2019-21)")
ax.scatter(merged_recent.TOTAL_FLOOR_AREA, merged_recent.inflated_cost, s=2, alpha=0.2)
plt.savefig("outputs/figures/area_cost_scatter.png")


# cost/capacity ASHP boxplots
merged_recent.loc[[v.is_integer() for v in merged_recent.capacity]].groupby(
    "capacity"
).size()
capacity_list = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16]

data_list = []
for capacity in capacity_list:
    filtered_df = merged_recent.loc[merged_recent.capacity == capacity]
    costs = list(filtered_df.inflated_cost.dropna())
    data_list.append(costs)

fig, ax = plt.subplots()
ax.set_ylim(0, 50000)
ax.set_xlabel("HP capacity")
ax.set_ylabel("Inflation-adjusted cost (£2021)")
ax.set_title("HP installation costs by capacity (2019-21)")
ax.boxplot(
    data_list,
    flierprops={
        "marker": "o",
        "markersize": 3,
        "markerfacecolor": "black",
        "alpha": 0.3,
    },
)
xtickNames = plt.setp(ax, xticklabels=capacity_list)
plt.tight_layout()
plt.savefig("outputs/figures/capacity_cost_boxplots.png")


# ATA/ATW

just_products = merged[["manufacturer", "product_name", "tech_type"]]

just_products.loc[
    just_products.manufacturer.str.lower().str.contains("panasonic")
    & just_products.product_name.str.lower().str.contains("s-")
]

aahps = {
    "a shade warmer": [
        "wz-",
    ],
    "carbon reduction products": [
        "ecohp",
    ],
    "bosch": [
        "climate 5000",
        "mini vrf",
        "mdci",
        # "single",  <- this returns a bunch of GSHPs
        "multi split",
    ],
    "ivt": [
        "nordic inverter",
        "pr-n",
    ],
    "daikin": [
        "siesta",
        "amxm",
        "atxm",
        "ftxm",
    ],
    "hitachi": [
        "ivx",
        "ras-",
        "ram-",
        "rak-",
        "rasc-",
        "rad-",
        "mono",
        "primairy",  # not a typo
        "utopia",
    ],
    "airwell": [
        "flow logic",
        "mini vrf",
        "cdm",
        "ddm",
        "dlse",
        "fdm",
        "hdh",
        "hdlw",
        "hkd",
        "hrd",
        "xbd",
        "xdl",
        "xdm",
    ],
    "carrier": [
        "xct",
        "x power mini vrf",
        "42q",
    ],
    "ciat": [
        "42hv",
    ],
    "clivet": [
        "mini vrf",
        "vrf m",
    ],
    "gree": [  # dodgy
        "gmv",
        "home",
        "slim",
        "mini",
    ],
    "haier": [
        "multisplit",
        "monosplit",
        "super match",
        "tide",
        "jade",
        "flexis plus",
        "pearl",
        "tundra",
        "nebula",
        "tower",
        "column",
    ],
    "hokkaido": [
        "hk",
        "hf",
        "inverter",
        "monosplit",
    ],
    "kaysun": [
        "suite",
        "amazon",
        "kam",
        "suite",
        "prodigy",
        "onnix",
        "double flow",
    ],
    "lennox": [
        "mpb",
        "mla",
        "mha",
    ],
    "lg": [  # dodgy
        "multi v",
        "artcool",
        "deluxe",
        "standard plus",
        "standard 2",
        "u24",
        "u4",
        "ua3",
        "ul2",
        "uu",
    ],
    "mdv": [
        "mini vrf",
        "mdv-v",
        "all-easy-",
        "blanc",
        "mission",
    ],
    "midea": [
        "mini vrf",
        "mdv-v",
        "all-easy-",
        "blanc",
        "mission",
    ],
    "mitsubishi": [
        "micro kxz",
        "scm",
        "fde",
        "srf",
        "srk",
        "skm",
        "srr",
        "inverter",
        "vna",
        "vsa",
        "vnp",
        "vnx",
        "vsx",
        "zmx",
        "zr",
        "zs",
        "zsx",
    ],
    "samsung": [
        "dvm s eco",
        "rac",
    ],
    "toshiba": [
        "mini-smmse",
        "inverter",
        "multisplit",
        "u2avg",
        "multi-split",
        "ras",
        "e2kv",
        "uzfv",
        "pkvsg",
        "seiya",
        "edge",
    ],
    "fujitsu": [
        "eco",
        "designer",
        "standard",
        "aoyg",
        "asyb",
        "abyg",
        "agyg",
        "auxg",
        "auyg",
    ],
    "equation": [
        "tide plus",
        "s-ac",
    ],
    "euroinverter": [
        "arkemix",
    ],
    "gss": [
        "gsx",
    ],
    "panasonic": [
        "cu-",
        "cs-",
        "u-",
        "s-",
    ],
}


merged["aahp"] = False

for man in aahps.keys():
    for prod in aahps[man]:
        merged["aahp"] = merged["aahp"] | (
            merged["manufacturer"].str.lower().str.contains(man)
            & merged["product_name"].str.lower().str.contains(prod)
        )


# unused plots, 15/11/21

# # numbers of installations each year
# def plot_installation_numbers(df):

#     numbers_data = df.groupby(["year", "tech_type"]).size().unstack().loc[2014:2021]

#     fig, ax = plt.subplots()

#     numbers_data.plot(kind="bar", ax=ax, zorder=3, color=["#1AC9E6", "#EB548C"])
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Installations")
#     ax.grid(axis="y", color="0.8", zorder=0)
#     ax.legend(title="Heat pump type")
#     ax.set_title("Numbers of MCS certified heat pumps installed")
#     plt.xticks(rotation=0)
#     plt.tight_layout()

#     plt.savefig(PROJECT_DIR / "outputs/figures/installation_numbers.png")


# def add_property_category(df):
#     df["property_category"] = np.nan
#     df.loc[
#         [string in ["Flat", "Maisonette"] for string in df.PROPERTY_TYPE],
#         "property_category",
#     ] = "Flat/Maisonette"
#     df.loc[
#         [
#             (string1 in ["House", "Bungalow"])
#             & (
#                 string2
#                 in [
#                     "Enclosed Mid-Terrace",
#                     "Mid-Terrace",
#                     "Enclosed End-Terrace",
#                     "End-Terrace",
#                 ]
#             )
#             for string1, string2 in zip(df.PROPERTY_TYPE, df.BUILT_FORM)
#         ],
#         "property_category",
#     ] = "Terraced House"
#     df.loc[
#         [
#             (string1 in ["House", "Bungalow"])
#             & (string2 in ["Detached", "Semi-Detached"])
#             for string1, string2 in zip(df.PROPERTY_TYPE, df.BUILT_FORM)
#         ],
#         "property_category",
#     ] = "Detached/Semi-Detached House"
#     return df


# c1 = "#142459"
# c2 = "#ee9a3a"
# c3 = "#820401"
# c4 = "#a9a9a9"


# def plot_property_medians(df, hp_type="Air Source Heat Pump", inflate=True):
#     if hp_type not in ["Air Source Heat Pump", "Ground/Water Source Heat Pump"]:
#         print("invalid hp type")
#     else:
#         df = df[df.tech_type == hp_type]
#         if inflate:
#             variable = "inflated_cost"
#             y_label = "Median inflation-adjusted cost of installation (£2021)"
#         else:
#             variable = "cost"
#             y_label = "Median cost of installation (£)"
#         property_category_medians = (
#             df.groupby(["year", "property_category"])["cost"]
#             .median()
#             .unstack()
#             .loc[2016:2021]
#         )

#         fig, ax = plt.subplots()

#         ax.plot(
#             property_category_medians.index,
#             property_category_medians["Detached/Semi-Detached House"].tolist(),
#             label="Detached/Semi-Detached House",
#             c=c1,
#             marker="o",
#             markersize=4,
#         )
#         ax.plot(
#             property_category_medians.index,
#             property_category_medians["Flat/Maisonette"].tolist(),
#             label="Flat/Maisonette",
#             c=c2,
#             marker="o",
#             markersize=4,
#         )
#         ax.plot(
#             property_category_medians.index,
#             property_category_medians["Terraced House"].tolist(),
#             label="Terraced House",
#             c=c3,
#             marker="o",
#             markersize=4,
#         )
#         ax.set_xticks(list(range(2016, 2022)))
#         ax.grid(color="0.8")
#         ax.set_ylim(bottom=0)
#         ax.set_xlabel("Year")
#         ax.set_ylabel(y_label)
#         ax.legend(title="Dwelling type", loc="lower right")
#         short_dict = {
#             "Air Source Heat Pump": "ASHP",
#             "Ground/Water Source Heat Pump": "G/WSHP",
#         }
#         if inflate:
#             ax.set_title(
#                 "Median inflation-adjusted cost of MCS certified\n"
#                 + short_dict[hp_type]
#                 + " installations by dwelling type"
#             )
#         else:
#             ax.set_title(
#                 "Median cost of MCS certified "
#                 + short_dict[hp_type]
#                 + " installations by dwelling type"
#             )

#         plt.savefig(
#             "outputs/figures/property_medians_"
#             + short_dict[hp_type].lower().replace("/", "").replace(" ", "")
#             + ".png"
#         )


# def plot_property_counts(df, hp_type="Air Source Heat Pump"):
#     if hp_type not in ["Air Source Heat Pump", "Ground/Water Source Heat Pump", "any"]:
#         print("invalid hp type")
#     else:
#         if hp_type in ["Air Source Heat Pump", "Ground/Water Source Heat Pump"]:
#             df = df[df.tech_type == hp_type]

#         df["property_category"] = df["property_category"].fillna("Other/Unknown")

#         property_category_counts = (
#             df.groupby(["year", "property_category"])["cost"]
#             .count()
#             .unstack()
#             .loc[2014:2021]
#         )

#         fig, ax = plt.subplots()

#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Detached/Semi-Detached House"].tolist(),
#             label="Detached/Semi-Detached House",
#             marker="o",
#             markersize=4,
#             c=c1,
#         )
#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Flat/Maisonette"].tolist(),
#             label="Flat/Maisonette",
#             marker="o",
#             markersize=4,
#             c=c2,
#         )
#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Terraced House"].tolist(),
#             label="Terraced House",
#             marker="o",
#             markersize=4,
#             c=c3,
#         )
#         ax.plot(
#             property_category_counts.index,
#             property_category_counts["Other/Unknown"].tolist(),
#             label="Other/Unknown",
#             marker="o",
#             markersize=4,
#             c=c4,
#         )
#         ax.set_xticks(list(range(2014, 2022)))
#         ax.set_ylim(bottom=0)
#         ax.grid(color="0.8")
#         ax.set_xlabel("Year")
#         ax.set_ylabel("Number of installations")
#         ax.legend(title="Dwelling type")
#         short_dict = {
#             "Air Source Heat Pump": "ASHP",
#             "Ground/Water Source Heat Pump": "G/WSHP",
#             "any": "heat pump",
#         }
#         ax.set_title(
#             "Number of MCS certified "
#             + short_dict[hp_type]
#             + " installations by dwelling type"
#         )
#         plt.tight_layout()

#         plt.savefig(
#             "outputs/figures/property_counts_"
#             + short_dict[hp_type].lower().replace("/", "").replace(" ", "")
#             + ".png"
#         )


# #### Geographic distribution - local authorities
# def regional_plot(df):
#     df.groupby("local_authority").agg(
#         number=("cost", lambda col: col.count()),
#         median_cost=("cost", lambda col: col.median()),
#     ).sort_values("median_cost", ascending=False)
#     # tbc...


# # overall plots


# df=merged
# col_list = ['red', 'green', 'blue']
# fig, ax = plt.subplots()
# ax.set_ylim(0, 100000)
# for year in range(2019, 2022):
#     df_filtered = df.loc[(df.year == year) & (df.capacity < 30) & (df.inflated_cost < 100000), ["capacity", "inflated_cost"]].dropna()
#     ax.scatter(df_filtered.capacity, df_filtered.inflated_cost, c=col_list[year-2019], s=2, alpha=0.01)
#     ax.plot(np.unique(df_filtered.capacity), np.poly1d(np.polyfit(df_filtered.capacity, df_filtered.inflated_cost, 1))(np.unique(df_filtered.capacity)), c=col_list[year-2019])

# fig, ax = plt.subplots()
# ax.set_ylim(0, 30000)
# for year in range(2019, 2022):
#     df_filtered = df.loc[(df.year == year) & (df.TOTAL_FLOOR_AREA < 400) & (df.inflated_cost < 30000), ["TOTAL_FLOOR_AREA", "inflated_cost"]].dropna()
#     ax.scatter(df_filtered.TOTAL_FLOOR_AREA, df_filtered.inflated_cost, c=col_list[year-2019], s=2, alpha=0.01)
#     ax.plot(np.unique(df_filtered.TOTAL_FLOOR_AREA), np.poly1d(np.polyfit(df_filtered.TOTAL_FLOOR_AREA, df_filtered.inflated_cost, 1))(np.unique(df_filtered.TOTAL_FLOOR_AREA)), c=col_list[year-2019])


# indiv_sums = dhps.groupby(['product_id', 'flow_temp', 'scop']).size().reset_index().rename(columns={0:'indiv_counts'})
# product_sums = dhps.groupby('product_id').size().reset_index().rename(columns={0:'total'})

# test = indiv_sums.merge(product_sums, on='product_id')
# test = test.set_index(['product_id', 'flow_temp', 'scop'])
# test.sort_values('total', ascending=False)


# some LAs with not many installations
# some non-standard local authority names -
# e.g. "Belfast City Council, Lisburn & Castlereagh..."
