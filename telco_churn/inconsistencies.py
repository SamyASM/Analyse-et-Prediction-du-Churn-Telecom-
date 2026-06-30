"""
Fonctions de remplacement par valeurs manquantes pour les INCOHERENCES.

Détecte les valeurs aberrantes (tenure négatif/999, TotalCharges
incompatible avec MonthlyCharges et tenure, etc.) et les remplace
par NaN, pour qu'elles soient gérées proprement à l'étape d'imputation.
"""

import numpy as np
import pandas as pd


def replace_na_tenure(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace par NaN les incohérences sur 'tenure' :
    valeurs négatives ou nulles, et la valeur sentinelle 999.
    """
    data = data.copy()
    data.loc[(data["tenure"] <= 0) | (data["tenure"] == 999), "tenure"] = np.nan
    return data


def replace_na_total_charges(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remplace par NaN les incohérences sur 'TotalCharges' :
    - valeurs négatives
    - TotalCharges < MonthlyCharges (impossible)
    - TotalCharges très inférieur à tenure * MonthlyCharges (sous-paiement
      au-delà d'une tolérance d'1/3 des mois, pouvant être une promo)
    - TotalCharges très supérieur à tenure * MonthlyCharges (sur-paiement
      au-delà d'une tolérance d'1/5 des mois)
    """
    data = data.copy()

    # Valeurs négatives
    data.loc[data["TotalCharges"] < 0, "TotalCharges"] = np.nan

    # TotalCharges inférieur à la charge mensuelle : impossible
    data.loc[data["TotalCharges"] < data["MonthlyCharges"], "TotalCharges"] = np.nan

    # Sous-paiement incohérent (tolérance: 1/3 des mois peut être une promo)
    tolerance_bas = (data["tenure"] // 3) * data["MonthlyCharges"]
    incoherent_bas = data["TotalCharges"] < ((data["tenure"] * data["MonthlyCharges"]) - tolerance_bas)
    data.loc[incoherent_bas, "TotalCharges"] = np.nan

    # Sur-paiement incohérent (tolérance: 1/5 des mois)
    tolerance_haut = (data["tenure"] // 5) * data["MonthlyCharges"]
    incoherent_haut = data["TotalCharges"] > ((data["tenure"] * data["MonthlyCharges"]) + tolerance_haut)
    data.loc[incoherent_haut, "TotalCharges"] = np.nan

    return data


def replace_na_churn_unknown(data: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes où Churn vaut 'unknown' : sans étiquette,
    ce n'est plus de l'apprentissage supervisé exploitable.
    """
    return data[data["Churn"] != "unknown"].copy()


def gerer_incoherences(data: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline complet de gestion des incohérences : tenure, TotalCharges,
    puis suppression des lignes Churn='unknown'.

    Args
    ----
    data : DataFrame harmonisé (texte en minuscules) avec TotalCharges
           déjà converti en numérique.

    Return
    ------
    DataFrame nettoyé des incohérences (valeurs aberrantes -> NaN).
    """
    data = replace_na_tenure(data)
    data = replace_na_total_charges(data)
    data = replace_na_churn_unknown(data)
    return data
