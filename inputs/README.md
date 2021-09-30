# Input Data

Last updated: 30 September 2021

Julia Suter

### Overview

In the `inputs/EPC_data` folder you will find the following versions of the EPC data.

- Raw EPC data for England, Wales and Scotland.
- Cleansed EPC data for Wales and England provided by EST.
- Preprocessed (cleaned, deduplicated and added features) EPC data for Wales, England and Scotland.

### Raw data

In `inputs/EPC_data/Raw_data` you can find the raw EPC data. The current version holds the data up to the second quarter of 2021.

The data for England and Wales can be found in `inputs/EPC*data/Raw_data/England_Wales in a zipped file named \_all-domestic-certificates.zip*.

The data for Scotland can be found in `inputs/EPC*data/Raw_data/Scotland in a zipped file named \_D_EPC_data.csv*.

### Cleansed EPC (EST)

**IMPORTANT: We are not allowed to share this dataset without explicit permission by EST.**

In `/inputs/EPC_data/EST_cleansed_versions` you can find a cleansed version of the EPC data for England and Wales provided by EST.

EST selected a set of relevant features, cleaned them and got rid of erroneous values. They also identified duplicates. Both versions with and without deduplication are available as zipped files:

- EPC_England_Wales_cleansed.csv.zip
- EPC_England_Wales_cleansed_and_deduplicated.csv.zip

This version does not include the most recent EPC data as it only contains entries until the first quarter of 2020. It also does not include any data on Scotland.

The cleansed set includs the following features:

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

### Preprocessed EPC Dataset

In `inputs/EPC_data/Preprocessed_data/Q2_2021` you can find three different versions of preprocessed EPC data.

**Since the preprocessing depends on data cleaning and feature engineering algorithms that may change over time, the data in this folder should be considered a snapshot of the current status in September 2021. Ideally, you should always work with the output of the most recent preprocessing version. **

You can generate the preprocessed datasets from the raw data by executing the script in _preprocess_data.py_ in `/heat_pump_adoption_modelling/pipeline/preprocessing`.

It will generate three versions of the data in `/outputs/EPC_data/Preprocessed_data/Q[quarter]_[YEAR]`. They will be written out as regular CSV-files.

- _EPC_GB_raw.csv_ : original data

- _EPC_GB_preprocessed.csv_: cleaned and added features, includes duplicates

- _EPC_GB_preprocessed_and_deduplicated.csv_: cleaned, added features and without duplicates

_EPC_GB_raw.csv_ merges the data for England, Wales and Scotland in one file, yet leaves the data unaltered.

The preprocessing includes cleaning and standardising the data and adding additional features. For a detailed description of the proprocessing consult the documentation [work in progress, link follows].

We also identified duplicates, i.e. samples referring to the same property yet often at different times. A preprocessed version of the EPC data without duplicates can be found in _EPC_GB_preprocessed_and_deduplicated.csv_.

Since duplicates can be interesting for some research questions, we also save the version with duplicates included as _EPC_GB_preprocessed.csv_.

**Note**: In order make up/downloading more efficient, we zipped the files in `/inputs/EPC_data/Preprocessed_data` - so the filenames all end in _.zip_. The data loading script in `/heat_pump_adoption_modelling/pipeline/getters/epc_data.py` will handle the unzipping if necessary.
