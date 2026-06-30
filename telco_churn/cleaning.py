"""
Pipeline de NETTOYAGE complet.

Assemble, dans le bon ordre, les étapes :
1. Correction du format de TotalCharges
2. Harmonisation des doublons sémantiques
3. Gestion des incohérences (-> NaN)

Ce module ne fait QUE du nettoyage déterministe, appliqué AVANT le
split train/test (aucune fuite de données possible ici puisqu'on ne
calcule aucune statistique globale dépendante du split, type
moyenne/médiane d'imputation).
"""

import pandas as pd

from .formatting import total_charges_to_numeric
from .harmonization import harmoniser_tout
from .inconsistencies import gerer_incoherences


def nettoyer_dataset(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Applique le pipeline de nettoyage complet sur le DataFrame brut.

    Args
    ----
    data    : DataFrame brut tel que chargé depuis le CSV
    verbose : affiche des messages d'avancement

    Return
    ------
    DataFrame nettoyé (format corrigé, texte harmonisé, incohérences -> NaN).
    """
    if verbose:
        print("1/3 - Correction du format de TotalCharges...")
    data = total_charges_to_numeric(data, verbose=verbose)

    if verbose:
        print("2/3 - Harmonisation des catégories...")
    data = harmoniser_tout(data)

    if verbose:
        print("3/3 - Gestion des incohérences (valeurs aberrantes -> NaN)...")
    data = gerer_incoherences(data)

    if verbose:
        print("Nettoyage terminé. Shape finale :", data.shape)

    return data
