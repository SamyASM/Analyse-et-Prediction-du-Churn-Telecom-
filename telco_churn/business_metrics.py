"""
Métrique métier : coût moyen asymétrique (Faux Positif / Faux Négatif).

Le but n'est pas de maximiser l'accuracy/F1, mais de MINIMISER
le coût moyen des erreurs de classification :
- Fausse alerte (FP)     : coûte COST_FP €
- Churn non détecté (FN) : coûte COST_FN €
"""

from sklearn.metrics import confusion_matrix, make_scorer

COST_FP = 25
COST_FN = 200
SEUIL_BAYES = COST_FP / (COST_FP + COST_FN)  # seuil de décision optimal théorique


def cout_moyen(y_true, y_pred, cout_fp: float = COST_FP, cout_fn: float = COST_FN) -> dict:
    """
    Calcule le coût moyen empirique R_hat(g) en euros pour un classifieur.

    Args
    ----
    y_true : labels réels (0/1)
    y_pred : labels prédits (0/1)
    cout_fp, cout_fn : coûts unitaires des erreurs

    Return
    ------
    dict avec cout_total, cout_moyen, TN, FP, FN, TP, n
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    cout_total = fp * cout_fp + fn * cout_fn
    n = len(y_true)

    return {
        "cout_total": cout_total,
        "cout_moyen": cout_total / n,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "n": n,
    }


def afficher_resultats(nom: str, resultats: dict) -> None:
    """Affiche joliment le résultat d'un appel à cout_moyen()."""
    print(f"\n--- {nom} ---")
    print(f"  TN={resultats['TN']}  FP={resultats['FP']}  "
          f"FN={resultats['FN']}  TP={resultats['TP']}")
    print(f"  Coût total      : {resultats['cout_total']:.2f} €")
    print(f"  Coût moyen/indiv: {resultats['cout_moyen']:.4f} €")


def _scorer_cout(y_true, y_pred):
    """Score à MAXIMISER pour sklearn (donc le coût négatif)."""
    return -cout_moyen(y_true, y_pred)["cout_moyen"]


def get_business_scorer():
    """
    Renvoie un scorer sklearn compatible cross_val_score / GridSearchCV,
    basé sur le coût métier (à maximiser = coût minimal).
    """
    return make_scorer(_scorer_cout)
