"""
telco_churn : package modulaire pour le projet de prédiction de churn télécom.

Modules :
- loading            : chargement du CSV
- formatting         : correction du format de TotalCharges
- harmonization      : harmonisation des doublons sémantiques
- inconsistencies    : valeurs aberrantes -> NaN
- cleaning           : pipeline de nettoyage complet (assemble les 3 ci-dessus)
- splitting          : découpage train/test stratifié
- preprocessing      : ColumnTransformer (imputation, scaling, encodage)
- business_metrics   : coût métier asymétrique FP/FN
- modeling           : entraînement et comparaison de modèles

Exemple d'utilisation rapide :

    from telco_churn import nettoyer_dataset, split_train_test, build_preprocessor
    from telco_churn import encoder_cible, get_modeles, evaluer_modeles_test

    df = load_data("telco_customer_data_v2.csv")
    df_clean = nettoyer_dataset(df)
    X_train, X_test, y_train, y_test = split_train_test(df_clean, random_state=42)

    y_train, y_test, _ = encoder_cible(y_train, y_test)

    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    modeles = get_modeles()
    resultats = evaluer_modeles_test(modeles, X_train_t, y_train, X_test_t, y_test)
"""

from .loading import load_data
from .formatting import mauvais_format_total_charges, total_charges_to_numeric
from .harmonization import harmoniser_tout
from .inconsistencies import gerer_incoherences
from .cleaning import nettoyer_dataset
from .splitting import split_train_test
from .preprocessing import build_preprocessor
from .business_metrics import COST_FP, COST_FN, SEUIL_BAYES, cout_moyen, afficher_resultats, get_business_scorer
from .modeling import encoder_cible, get_modeles, evaluer_modeles_test, comparer_modeles_cv

__all__ = [
    "load_data",
    "mauvais_format_total_charges",
    "total_charges_to_numeric",
    "harmoniser_tout",
    "gerer_incoherences",
    "nettoyer_dataset",
    "split_train_test",
    "build_preprocessor",
    "COST_FP",
    "COST_FN",
    "SEUIL_BAYES",
    "cout_moyen",
    "afficher_resultats",
    "get_business_scorer",
    "encoder_cible",
    "get_modeles",
    "evaluer_modeles_test",
    "comparer_modeles_cv",
]
