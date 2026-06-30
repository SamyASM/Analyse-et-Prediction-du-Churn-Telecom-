"""
Construction du PREPROCESSOR (imputation, mise à l'échelle, encodage).

Ce module ne fait que CONSTRUIRE le ColumnTransformer : il ne fait
jamais de fit/transform lui-même. C'est à l'appelant de faire
.fit_transform() sur X_train et .transform() sur X_test, pour éviter
toute fuite de données.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

COL_TENURE = ["tenure"]
COL_ROBUST = ["MonthlyCharges", "TotalCharges"]
COL_DROP = ["customerID"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construit le ColumnTransformer du projet :
    - tenure        : imputation moyenne + StandardScaler
    - MonthlyCharges/TotalCharges : imputation médiane + RobustScaler (valeurs extrêmes)
    - catégorielles : imputation valeur la plus fréquente + OneHotEncoder

    Args
    ----
    X : DataFrame des features (utilisé uniquement pour détecter
        automatiquement les colonnes catégorielles)

    Return
    ------
    ColumnTransformer non fitté.
    """
    col_cat = (
        X.select_dtypes(include="object")
        .drop(columns=[c for c in COL_DROP if c in X.columns], errors="ignore")
        .columns.tolist()
    )

    pipe_tenure = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    pipe_robust = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler()),
    ])

    pipe_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        [
            ("tenure", pipe_tenure, COL_TENURE),
            ("robust", pipe_robust, COL_ROBUST),
            ("cat", pipe_cat, col_cat),
        ],
        remainder="drop",  # drop customerID explicitement
    )

    return preprocessor
