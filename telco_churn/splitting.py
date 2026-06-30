"""
Découpage TRAIN / TEST.

A appliquer juste après le nettoyage déterministe (formatting,
harmonisation, incohérences) et AVANT toute imputation/scaling/encodage,
pour éviter toute fuite de données.
"""

from sklearn.model_selection import train_test_split


def split_train_test(data, target_col: str = "Churn", test_size: float = 0.20, random_state: int = None):
    """
    Sépare un DataFrame nettoyé en X_train, X_test, y_train, y_test,
    avec stratification sur la variable cible pour conserver
    le déséquilibre de classes initial.

    Args
    ----
    data         : DataFrame nettoyé (avant imputation/encodage)
    target_col   : nom de la colonne cible (défaut: Churn)
    test_size    : proportion du jeu de test
    random_state : graine aléatoire (None = non reproductible, comme l'original ;
                   fixez une valeur entière pour des résultats reproductibles)

    Return
    ------
    X_train, X_test, y_train, y_test
    """
    X = data.drop(columns=[target_col])
    y = data[target_col]

    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
