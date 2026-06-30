"""
Chargement du dataset.
"""

import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Charge le dataset Telco Churn depuis un chemin CSV.

    Args
    ----
    path : chemin vers le fichier .csv

    Return
    ------
    DataFrame brut, tel que chargé.
    """
    return pd.read_csv(path)
