# File: heat_pump_adoption_modelling/pipeline/supervised_model/utils/kepler.py
"""
Generate Kepler maps for supervised model outputs.
"""

# ----------------------------------------------------------------------------------


import yaml
from heat_pump_adoption_modelling import get_yaml_config, Path, PROJECT_DIR


# Load config file
config = get_yaml_config(
    Path(str(PROJECT_DIR) + "/heat_pump_adoption_modelling/config/base.yaml")
)

KEPLER_PATH = str(PROJECT_DIR) + config["SUPERVISED_MODEL_OUTPUT"] + "/kepler/"


def get_config(path):
    """Return Kepler config in yaml format.

    Parameters
    ----------
    path: str
        Path to config files.

    Return
    ---------
    config: str
        Return Kepler configuration content."""

    with open(path, "r") as infile:
        config = infile.read()
        config = yaml.load(config, Loader=yaml.FullLoader)

    return config


def save_config(map, config_path):
    """Save Kepler map configruations.

    Parameters
    ----------
    map: Kepler.map
        Kepler map after modifications.

    config_path: str
        Path to config files.

    Return: None"""

    with open(config_path, "w") as outfile:
        outfile.write(str(map.config))
