# EPC Dataset

**This documentation describes the different EPC data versions for England, Wales and Scotland and how the the data was preprocessed in order to facilitate data analysis.**

### Original EPC Data

##### Download

The original data was for England and Wales is freely available [here](https://epc.opendatacommunities.org/https://epc.opendatacommunities.org/). The data for Scotland can be downloaded from [here](https://statistics.gov.scot/resource?uri=http%3A%2F%2Fstatistics.gov.scot%2Fdata%2Fdomestic-energy-performance-certificates).

Note that the data is updated every quarter so in order to work with the most recent data, follow the steps under Data Updates.

| Version                   | # Samples  | # Features |
| ------------------------- | ---------- | ---------- |
| Original raw data         | 22 840 162 | 40         |
| After cleaning            | 22 840 162 | 40         |
| After adding features     | 22 839 568 | 54         |
| After removing duplicates | 181 797 19 | 54         |

##### Data Updates

The EPC registry is updated 4 times a year, so every quarter. In order to work with the most recent data, the EPC data needs to be downloaded and preprocessed.

**Current version: Q2_2021**

Below we describe the necessary steps to update the data. It's actually less complicated than it looks:

- Download most recent England/Wales data from [here](https://epc.opendatacommunities.org/https://epc.opendatacommunities.org/). The filename is _all-domestic-certificates.zip_.
- Download most recent Scotland data from [here](https://statistics.gov.scot/resource?uri=http%3A%2F%2Fstatistics.gov.scot%2Fdata%2Fdomestic-energy-performance-certificates). The filename is is of the format `D_EPC_data_2012-[year]Q[quarter]_extract_[month][year].zip`, for example _D_EPC_data_2012-2021Q2_extract_0721.zip_.
- Clear the folder `inputs/EPC_data/raw_data`. If there is raw EPC data from previous quarters/years, delete it or move it to a subfolder. Note that the most recently downlaoded data will include all the data from previous downloads so there should be no data loss when overwriting previous data.
- Move both downloaded files to the folder `inputs/EPC_data/raw_data`.
- Shorten the Scotland filename to _D_EPC_data.zip_.
- Note: The zipped Scotland data may not be processable by the Python package _ZipFile_ because of an unsupported compression method. This problem can be solved easily solved by unzipping and zipping the data manually. Make sure the filename remains _D_EPC_data.zip_.

- Create a new folder in `outpus/EPC_data/preprocessed_data/` with the name pattern `Q[quarter]_[year]`, indicating when the data was updated last. This information will be displayed when downloading the data and is reflected in the original filename for Scotland. For example, _Q2_2021_ includes the data up to June 2021.

- Open the config file `config/base.yaml` and update all the paths including `outputs/EPC_data/preprocessed_data/Q[quarter]_[year]` to match the folder created in the last step.

- Execute the script XYZ which generates the preprocessed data in folder `outputs/EPC_data/preprocessed_data/Q[quarter]_[year]`.

  Note: The data preprocessing is optimised for the data collected in 2021 (Q2_2021). More recent EPC data may include feature values not yet covered by the current preprocessing algorithm (for example new construction age bands), possibly causing errors when excuting the script.
  These can usually be fixed easily so feel free to open an issue or submit a pull request.

- Enjoy your updated EPC data.
