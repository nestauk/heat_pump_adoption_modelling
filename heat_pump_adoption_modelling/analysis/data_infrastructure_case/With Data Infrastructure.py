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
# # Version with Data Infrastructure
#
# ### Load Wales EPC Data and Plot Construction Age by Property Type
#
# ![](sun.jpg)

# %%
# %load_ext autoreload
# %autoreload 2

import asf_core_data

from asf_core_data.getters.epc import epc_data, data_batches
from asf_core_data.pipeline.preprocessing import preprocess_epc_data
from asf_core_data.utils.visualisation import easy_plotting

from asf_core_data import Path

# %%
LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

# %%
data_batches.check_for_newest_batch(data_path=LOCAL_DATA_DIR, verbose=True)

# %%
wales_epc = preprocess_epc_data.load_and_preprocess_epc_data(
    data_path=LOCAL_DATA_DIR, batch="newest", n_samples=5000, subset="Wales"
)

# %%
from asf_core_data.utils.visualisation import easy_plotting

easy_plotting.plot_subcats_by_other_subcats(
    wales_epc,
    "PROPERTY_TYPE",
    "CONSTRUCTION_AGE_BAND",
    plotting_colors="viridis",
    legend_loc="outside",
    plot_title="Construction Age by Property Type",
    fig_save_path=Path(LOCAL_DATA_DIR) / "outputs/figures/",
)

# %% [markdown]
# ## Success!
#
# ![party.png](images.png)
#
# - Clean and readable code
# - You can get started with the actual work

# %%
data_batches.get_all_batch_names(data_path="S3")

# %%
data_batches.get_most_recent_batch(data_path="S3")

# %%
data_batches.get_all_batch_names(data_path=LOCAL_DATA_DIR)

# %%
data_batches.get_most_recent_batch(data_path=LOCAL_DATA_DIR)

# %%
wales_epc = epc_data.load_england_wales_data(
    data_path=LOCAL_DATA_DIR, subset="Wales", batch="2021_Q2_0721"
)
wales_epc.shape

# %%
wales_epc = epc_data.load_raw_epc_data(
    data_path=LOCAL_DATA_DIR, subset="Wales", batch="newest"
)
wales_epc.shape

# %%
wales_epc = epc_data.load_preprocessed_epc_data(
    data_path=LOCAL_DATA_DIR, batch="newest"
)
wales_epc.shape

# %%
