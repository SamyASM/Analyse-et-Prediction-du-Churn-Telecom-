"""
Fonctions de FORMAT.

Corrige le format de la colonne TotalCharges, qui est censée être
numérique mais contient des symboles monétaires / virgules.
"""

import pandas as pd


def mauvais_format_total_charges(data: pd.DataFrame, col: str = "TotalCharges") -> pd.DataFrame:
    """
    Renvoie les lignes dont `col` est non convertissable en numérique
    (hors vrais NaN).

    Args
    ----
    data : DataFrame
    col  : nom de la colonne à vérifier (défaut: TotalCharges)

    Return
    ------
    DataFrame des lignes problématiques.
    """
    non_null = data[col].notna()
    apres_conversion = pd.to_numeric(data[col], errors="coerce")

    # Les nouveaux NaN (après conversion) qui n'étaient pas NaN avant = problèmes
    return data[non_null & apres_conversion.isna()]


def total_charges_to_numeric(data: pd.DataFrame, col: str = "TotalCharges", verbose: bool = True) -> pd.DataFrame:
    """
    Convertit `col` en numérique en nettoyant les caractères non numériques
    (symboles monétaires, virgules décimales, etc.)

    /!\\ A utiliser si toutes les valeurs sont dans la même unité (ex: dollars)

    Args
    ----
    data    : DataFrame
    col     : colonne à convertir
    verbose : affiche un message si aucune valeur problématique n'est trouvée

    Return
    ------
    DataFrame avec `col` convertie en numérique.
    """
    data = data.copy()
    bad_lines = mauvais_format_total_charges(data, col)

    if len(bad_lines) == 0:
        if verbose:
            print("Aucune valeur problématique identifiée dans", col)
        return data

    # Virgule -> point
    data.loc[bad_lines.index, col] = bad_lines[col].str.replace(",", ".", regex=False)

    # On garde uniquement chiffres et points
    data.loc[bad_lines.index, col] = data.loc[bad_lines.index, col].str.replace(r"[^0-9.]", "", regex=True)

    data[col] = pd.to_numeric(data[col], errors="coerce")

    return data
