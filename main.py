"""
main.py — Pipeline complet de bout en bout, reproductible.

Reprend exactement les étapes du notebook PROJET.ipynb mais en
s'appuyant sur les fonctions modulaires du package telco_churn.

Usage :
    python main.py /chemin/vers/telco_customer_data_v2.csv
"""

import sys

from telco_churn import (
    load_data,
    nettoyer_dataset,
    split_train_test,
    build_preprocessor,
    encoder_cible,
    get_modeles,
    evaluer_modeles_test,
    comparer_modeles_cv,
    afficher_resultats,
    SEUIL_BAYES,
)


def main(csv_path: str, random_state: int = 42):
    # 1. Chargement
    print(">>> Chargement des données...")
    df = load_data(csv_path)
    print("Shape brute :", df.shape)

    # 2. Nettoyage (format + harmonisation + incohérences -> NaN)
    print("\n>>> Nettoyage des données...")
    df_clean = nettoyer_dataset(df)

    # 3. Split train/test AVANT toute transformation non déterministe
    print("\n>>> Split train/test...")
    X_train, X_test, y_train, y_test = split_train_test(
        df_clean, target_col="Churn", test_size=0.20, random_state=random_state
    )
    print("Train :", X_train.shape, " Test :", X_test.shape)

    # 4. Encodage de la cible
    y_train_enc, y_test_enc, target_encoder = encoder_cible(y_train, y_test)

    # 5. Préprocessing (imputation + scaling + encodage), fit sur train uniquement
    print("\n>>> Préprocessing (imputation, scaling, encodage)...")
    preprocessor = build_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    # 6. Modélisation : comparaison sur le jeu de test
    print(f"\n>>> Seuil de Bayes théorique : {SEUIL_BAYES:.4f}")
    modeles = get_modeles(random_state=random_state)

    print("\n>>> Évaluation sur le jeu de test (seuil 0.5)...")
    resultats_test = evaluer_modeles_test(modeles, X_train_t, y_train_enc, X_test_t, y_test_enc)
    for nom, res in resultats_test.items():
        afficher_resultats(nom, res)

    # 7. Validation croisée pour confirmer la robustesse des résultats
    print("\n>>> Cross-validation (5 folds) sur le coût métier...")
    modeles_cv = get_modeles(random_state=random_state)  # nouvelles instances non fittées
    comparer_modeles_cv(modeles_cv, X_train_t, y_train_enc, random_state=random_state)

    return {
        "preprocessor": preprocessor,
        "modeles": modeles,
        "resultats_test": resultats_test,
        "target_encoder": target_encoder,
    }


if __name__ == "__main__":
    chemin = sys.argv[1] if len(sys.argv) > 1 else "telco_customer_data_v2.csv"
    main(chemin)
