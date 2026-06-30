"""
MODELISATION : entraînement et comparaison de modèles avec coût métier.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from .business_metrics import COST_FN, COST_FP, cout_moyen, get_business_scorer


def encoder_cible(y_train, y_test):
    """
    Encode la variable cible (texte -> 0/1) avec un LabelEncoder
    fitté sur y_train uniquement.

    Return
    ------
    y_train_enc, y_test_enc, encoder (utile pour inverse_transform)
    """
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)
    return y_train_enc, y_test_enc, encoder


def get_modeles(random_state: int = 42) -> dict:
    """
    Renvoie les 3 modèles candidats du projet :
    régression logistique classique, pondérée par les coûts, et Gradient Boosting.
    """
    return {
        "Logistique classique": LogisticRegression(),
        "Logistique pondérée": LogisticRegression(class_weight={0: COST_FP, 1: COST_FN}),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }


def evaluer_modeles_test(modeles: dict, X_train, y_train, X_test, y_test, seuil: float = 0.5) -> dict:
    """
    Entraîne chaque modèle sur (X_train, y_train) et évalue le coût
    moyen sur (X_test, y_test) avec un seuil de décision donné.

    Return
    ------
    dict {nom_modele: resultats (dict de cout_moyen)}
    """
    resultats = {}
    for nom, modele in modeles.items():
        modele.fit(X_train, y_train)
        proba = modele.predict_proba(X_test)[:, 1]
        y_pred = (proba > seuil).astype(int)
        resultats[nom] = cout_moyen(y_test, y_pred)
    return resultats


def comparer_modeles_cv(modeles: dict, X_train, y_train, n_splits: int = 5, random_state: int = 42) -> dict:
    """
    Compare les modèles par Cross-Validation stratifiée sur le coût métier.

    Return
    ------
    dict {nom_modele: cout_moyen_cv}
    """
    scorer = get_business_scorer()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    scores_cv = {}
    for nom, modele in modeles.items():
        scores = cross_val_score(modele, X_train, y_train, cv=cv, scoring=scorer)
        couts = -scores
        scores_cv[nom] = {"moyenne": couts.mean(), "std": couts.std(), "details": couts}
        print(f"{nom}: coût moyen CV = {couts.mean():.4f} € (+/- {couts.std():.4f})")

    return scores_cv
