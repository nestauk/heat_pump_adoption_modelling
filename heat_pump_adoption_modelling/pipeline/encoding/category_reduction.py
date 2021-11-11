# File: heat_pump_adoption_modelling/pipeline/encoding/category_reduction.py
"""
Reduce or merge categories in order to reduce the feature space.
"""

# ----------------------------------------------------------------------------------

# Import
import numpy as np

# ----------------------------------------------------------------------------------


transaction_type_red = {
    "rental": "rental",
    "marketed sale": "marketed saled",
    "unknown": "unknown",
    np.nan: "unknown",
    "not recorded": "unknown",
    "assessment for green deal": "green deal related",
    "following green deal": "green deal related",
    "RHI application": "green deal related",
    "FiT application": "green deal related",
    "non marketed sale": "non marketed sale",
    "new dwelling": "new dwelling",
    "ECO assessment": "ECO assessment",
    "rental (social)": "rental (social)",
    "rental (private)": "rental (private)",
    "rental (private) - this is for backwards compatibility only and should not be used": "rental (private)",
    "Stock Condition Survey": "Stock Condition Survey",
    "Stock condition survey": "Stock Condition Survey",
    "not sale or rental": "not sale or rental",
    "rental (social) - this is for backwards compatibility only and should not be used": "rental (social)",
    "not recorded - this is for backwards compatibility only and should not be used": "unknown",
}


glazing_type_red = {
    "double glazing installed before 2002": "double glazing",
    "double glazing installed during or after 2002": "double glazing",
    "unknown": "unknown",
    "secondary glazing": "double glazing",
    "double glazing, unknown install date": "double glazing",
    "single glazing": "single glazing",
    "triple glazing": "triple glazing",
    "double, known data": "double glazing",
    "triple, known data": "triple glazing",
    np.nan: "unknown",
}

energy_tarrif_red = {
    "Single": "single",
    "dual": "dual",
    "Unknown": "unknown",
    "dual (24 hour)": "dual",
    "off-peak 18 hour": "off-peak",
    np.nan: "unknown",
    "standard tariff": "standard",
    "off-peak 7 hour": "off-peak",
    "unknown": "unknown",
    "off-peak 10 hour": "off-peak",
    "24 hour": "unknown",
    "5": "unknown",
}

mech_ventilation_red = {
    "natural": "natural",
    "mechanical, extract only": "mechanical",
    "mechanical, supply and extract": "mechanical",
    np.nan: "unknown",
    "unknown": "unknown",
}

feature_red_dict = {
    "GLAZED_TYPE": glazing_type_red,
    "MECHANICAL_VENTILATION": mech_ventilation_red,
    "ENERGY_TARIFF": energy_tarrif_red,
    "TRANSACTION_TYPE": transaction_type_red,
}


def category_reduction(df, feature_red_dict=feature_red_dict):
    """Reduce the number of categories for categorical features
    by mapping some values to the same category.
    For example for GLAZED_TYPE or ENERGY_TARIFF.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe with features for which to reduce number of categories.

    feature_red_dict : dict
        Feature and category reduction dict for mapping values to fewer categories.

    Return
    ---------
    df : pandas.DataFrame
        Dataframe with fewer categories."""

    for feat in feature_red_dict.keys():
        df[feat] = df[feat].map(feature_red_dict[feat])

    return df
