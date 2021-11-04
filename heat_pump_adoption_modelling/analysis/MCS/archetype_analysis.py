import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

from heat_pump_adoption_modelling import PROJECT_DIR
from heat_pump_adoption_modelling.pipeline.MCS.mcs_epc_joining import join_mcs_epc_data

merged = join_mcs_epc_data(all_records=True)

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


merged["TOTAL_FLOOR_AREA"] = merged["TOTAL_FLOOR_AREA"].astype(float)


inflation = pd.read_csv("inputs/data/inflation.csv").rename(
    columns={
        "Year": "year",
        "Multiplier to Use for 2021\n [Combined] Overall Index": "multiplier",
    }
)

merged["year"] = merged.date.dt.year
merged = merged.merge(inflation, how="left", on="year")
merged["inflated_cost"] = merged["cost"] * merged["multiplier"]


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


def plot_archetype_scatter(i):
    colname = "archetype_" + str(i)
    archetype_data = merged.loc[
        merged[colname], ["capacity", "inflated_cost", "tech_type"]
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

    ax.axvline(exp_dict[i][0], linestyle="--", color="blue")
    ax.axvline(exp_dict[i][1], linestyle="--", color="orange")

    ax.legend(loc="upper right")

    plt.savefig("outputs/figures/archetypes/" + colname + ".png")


for i in range(1, 11):
    plot_archetype_scatter(i)


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
