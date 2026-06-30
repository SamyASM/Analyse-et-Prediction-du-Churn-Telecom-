"""
Fonctions d'HARMONISATION.

Supprime les doublons sémantiques des variables catégorielles
(ex : 'Yes', 'yes', 'Y', 'True' qui signifient tous "oui").

A mettre à jour au fur et à mesure que de nouvelles incohérences
sont détectées dans les données.
"""

import pandas as pd

# Colonnes binaires oui/non à harmoniser vers 'y' / NaN inchangé sinon
COLONNES_OUI_NON = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def harmoniser_texte(data: pd.DataFrame) -> pd.DataFrame:
    """
    Met en minuscule et retire les espaces superflus de toutes les
    colonnes de type texte (object).
    """
    data = data.copy()
    col = data.select_dtypes(include=["object"]).columns
    for x in col:
        data[x] = data[x].str.lower().str.strip()
    return data


def harmoniser_gender(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["gender"] = data["gender"].replace({"male": "m", "man": "m", "female": "f"})
    return data


def harmoniser_senior_citizen(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["SeniorCitizen"] = data["SeniorCitizen"].replace(
        {"no": "0", "yes": "1", "not senior": "0"}
    )
    return data


def harmoniser_options_oui_non(data: pd.DataFrame, colonnes: list = None) -> pd.DataFrame:
    """
    Harmonise les colonnes d'options (OnlineSecurity, TechSupport, ...)
    où 'yes'/'true' doivent devenir 'y'.
    """
    data = data.copy()
    colonnes = colonnes or COLONNES_OUI_NON
    for col in colonnes:
        data[col] = data[col].replace({"yes": "y", "true": "y"})
    return data


def harmoniser_contract(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Contract"] = data["Contract"].replace({"month-to-month": "m-m"})
    return data


def harmoniser_churn(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["Churn"] = data["Churn"].replace(
        {"yes": "y", "churned": "y", "no": "n", "no churn": "n"}
    )
    return data


def harmoniser_payment_method(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["PaymentMethod"] = data["PaymentMethod"].replace(
        {"bank transfer (automatic)": "bank transfer"}
    )
    return data


def harmoniser_tout(data: pd.DataFrame) -> pd.DataFrame:
    """
    Applique toutes les harmonisations dans l'ordre, en partant
    d'un DataFrame brut (texte non normalisé).

    Args
    ----
    data : DataFrame

    Return
    ------
    DataFrame harmonisé.
    """
    data = harmoniser_texte(data)
    data = harmoniser_gender(data)
    data = harmoniser_senior_citizen(data)
    data = harmoniser_options_oui_non(data)
    data = harmoniser_contract(data)
    data = harmoniser_churn(data)
    data = harmoniser_payment_method(data)
    return data
