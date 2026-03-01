# Analyse et Prédiction du Churn Telecom

Projet de machine learning visant à prédire la résiliation client (churn) dans un contexte télécom, à partir d’un dataset volontairement dégradé (valeurs manquantes, incohérences, erreurs de typage).

## Contexte métier

Le churn représente une perte directe de revenu.  
L’objectif est d’identifier les clients à risque afin de prioriser des actions de rétention sous contrainte opérationnelle (capacité limitée d’appels ou d’offres).

Le coût d’erreur est asymétrique :
- Prédire qu’un client reste alors qu’il part = perte business critique
- La priorité est donc le rappel sur la classe churn

## Structure du projet

- `Buisness_Understanding.ipynb` : cadrage métier et réflexion sur la priorisation
- `PROJET.ipynb` : nettoyage des données, exploration, modélisation

## Données

- ~70 000 observations
- 21 variables
- Variables démographiques, contractuelles, services, facturation
- Variable cible : `Churn`

## Data Preparation

Problèmes traités :
- Valeurs manquantes
- Types incohérents (TotalCharges importé en object)
- Valeurs négatives sur tenure
- Outliers extrêmes
- Incohérences entre MonthlyCharges et TotalCharges

Nettoyage et corrections appliqués avant modélisation.

## Modélisation

Baseline :
- RandomForestClassifier (scikit-learn)
- Train / test split
- Évaluation via accuracy, precision, recall

Orientation métier :
- Optimisation du rappel churn
- Logique de priorisation top-k

## Stack technique

- Python
- pandas
- numpy
- scikit-learn
- matplotlib / seaborn

## Améliorations possibles

- Pipeline sklearn complet (ColumnTransformer)
- Gestion du déséquilibre de classe
- Optimisation du seuil métier
- Validation croisée
- Industrialisation (joblib, Docker, API)

## Auteur

Samy ASMA
