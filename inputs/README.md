# Input Data

Last updated: 6 October 2021 by Julia Suter

## Overview

In the `inputs/EPC_data` folder you will find the following versions of the EPC data.

- Raw EPC data for England, Wales and Scotland.
- Cleansed EPC data for Wales and England provided by EST.
- Preprocessed (cleaned, deduplicated and added features) EPC data for Wales, England and Scotland.

## Raw data

In `inputs/EPC_data/raw_data` you can find the raw EPC data. The current version holds the data up to the second quarter of 2021.

The data for England and Wales can be found in `inputs/EPC_data/raw_data/england_wales` in a zipped file named _all-domestic-certificates.zip_.

The data for Scotland can be found in `inputs/EPC_data/raw_data/scotland` in a zipped file named _D_EPC_data.csv_.

| Selection (Q2_2021) | Samples    |
| ------------------- | ---------- |
| GB                  | 22 840 162 |
| England             | 20 318 501 |
| Wales               | 1 111 383  |
| Scotland            | 1 410 278  |

## Cleansed EPC (EST)

**IMPORTANT: We are not allowed to share this dataset without explicit permission by EST.**

In `inputs/EPC_data/EST_cleansed_versions` you can find a cleansed version of the EPC data for England and Wales provided by EST.

EST selected a set of relevant features, cleaned them and got rid of erroneous values. They also identified duplicates. Both versions with and without deduplication are available as zipped files:

- EPC_England_Wales_cleansed.csv.zip
- EPC_England_Wales_cleansed_and_deduplicated.csv.zip (14.4 million samples)

This version does not include the most recent EPC data as it only contains entries until the first quarter of 2020. It also does not include any data on Scotland.

The cleansed set includs the following 45 features:

```
ROW_NUM
LMK_KEY
ADDRESS1
ADDRESS2
ADDRESS3
POSTCODE
BUILDING_REFERENCE_NUMBER
LOCAL_AUTHORITY
LOCAL_AUTHORITY_LABEL
CONSTITUENCY
COUNTY
LODGEMENT_DATE
FINAL_PROPERTY_TYPE
FINAL_PROP_TENURE
FINAL_PROPERTY_AGE
FINAL_HAB_ROOMS
FINAL_FLOOR_AREA
FINAL_WALL_TYPE
FINAL_WALL_INS
FINAL_RIR
FINAL_LOFT_INS
FINAL_ROOF_TYPE
FINAL_MAIN_FUEL
FINAL_SEC_SYSTEM
FINAL_SEC_FUEL_TYPE
FINAL_GLAZ_TYPE
FINAL_ENERGY_CONSUMPTION
FINAL_EPC_BAND
FINAL_EPC_SCORE
FINAL_CO2_EMISSIONS
FINAL_FUEL_BILL
FINAL_METER_TYPE
FINAL_FLOOR_TYPE
FINAL_FLOOR_INS
FINAL_HEAT_CONTROL
FINAL_LOW_ENERGY_LIGHTING
FINAL_FIREPLACES
FINAL_WIND_FLAG
FINAL_PV_FLAG
FINAL_SOLAR_THERMAL_FLAG
FINAL_MAIN_FUEL_NEW
FINAL_HEATING_SYSTEM
```

## Preprocessed EPC Dataset

In `inputs/EPC_data/preprocessed_data/Q2_2021` you can find three different versions of preprocessed EPC data.

**Since the preprocessing depends on data cleaning and feature engineering algorithms that may change over time, the data in this folder should be considered a snapshot of the current status in September 2021. Ideally, you should always work with the output of the most recent preprocessing version.**

You can generate the preprocessed datasets from the raw data by executing the script in _preprocess_epc_data.py_ in `heat_pump_adoption_modelling/pipeline/preprocessing`.

It will generate three versions of the data in `outputs/EPC_data/preprocessed_data/Q[quarter]_[YEAR]`. They will be written out as regular CSV-files.

| Filename                                 | Version                                         | Samples    |
| ---------------------------------------- | ----------------------------------------------- | ---------- |
| EPC_GB_raw.csv                           | Original raw data                               | 22 840 162 |
| EPC_GB_preprocessed.csv                  | Cleaned and added features, includes duplicates | 22 839 568 |
| EPC_GB_preprocessed_and_deduplicated.csv | Cleaneda and added features, without duplicates | 18 179 719 |

_EPC_GB_raw.csv_ merges the data for England, Wales and Scotland in one file, yet leaves the data unaltered.

The preprocessing steps include cleaning and standardising the data and adding additional features. For a detailed description of the proprocessing consult this documentation [work in progress].

We also identified duplicates, i.e. samples referring to the same property yet often at different points in times. A preprocessed version of the EPC data without duplicates can be found in _EPC_GB_preprocessed_and_deduplicated.csv_.

Since duplicates can be interesting for some research questions, we also save the version with duplicates included as _EPC_GB_preprocessed.csv_.

**Note**: In order make up/downloading more efficient, we zipped the files in `inputs/EPC_data/preprocessed_data` - so the filenames all end in _.zip_. The data loading script in `heat_pump_adoption_modelling/pipeline/getters/epc_data.py` will handle the unzipping if necessary.
