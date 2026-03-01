# Analyse et Prédiction du Churn Telecom

Projet de machine learning visant à anticiper la résiliation (churn) clients à partir d’un dataset volontairement très dégradé (valeurs manquantes, incohérences métier, formats hétérogènes, variables mal typées). :contentReference[oaicite:0]{index=0}

## Contexte métier

- La problématique posée est la réduction d’un churn élevé et l’identification de profils à risque pour prioriser des actions marketing sous contrainte de capacité (ex: appels sortants limités). :contentReference[oaicite:1]{index=1}
- Coût d’erreur asymétrique
- L’erreur la plus critique est de prédire “reste” alors que le client va partir (perte de revenu + coût d’acquisition)
- Objectif implicite orienté rappel/coverage des churneurs dans le haut du ranking (logique top-k / lift). :contentReference[oaicite:2]{index=2}

## Contenu du dépôt

- Buisness_Understanding.ipynb
  - Formalisation métier du churn
  - Priorisation des erreurs (coût business)
  - Intuition lift/top 5% (capacité opérationnelle) :contentReference[oaicite:3]{index=3}
- PROJET.ipynb
  - Data understanding
  - Diagnostic qualité des données
  - Nettoyage et corrections d’incohérences
  - Baseline de modélisation scikit-learn (RandomForestClassifier) :contentReference[oaicite:4]{index=4}

## Données

Le notebook charge un fichier CSV local nommé telco_customer_data_v2.csv (chemin Windows présent dans le code). :contentReference[oaicite:5]{index=5}

- Dimensions observées au chargement: 70 000 lignes, 21 colonnes :contentReference[oaicite:6]{index=6}
- Variables
  - Démographiques: gender, SeniorCitizen, Partner, Dependents
  - Compte / contrat: tenure, Contract, PaperlessBilling, PaymentMethod
  - Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - Facturation: MonthlyCharges, TotalCharges
  - Cible: Churn :contentReference[oaicite:7]{index=7}

## Problèmes de qualité identifiés

- Valeurs manquantes sur plusieurs variables (ex: gender, Partner, Dependents, PaymentMethod, TotalCharges, etc.) :contentReference[oaicite:8]{index=8}
- Types incohérents
  - TotalCharges est importé en type object alors qu’il devrait être numérique comme MonthlyCharges :contentReference[oaicite:9]{index=9}
- Valeurs métier incohérentes
  - tenure négatif (ex: -5, -10) repéré dans des sous-ensembles :contentReference[oaicite:10]{index=10}
  - TotalCharges à 0 ou manquant malgré des MonthlyCharges non nuls (cas extrêmes à traiter)
  - Présence de chaînes parasites (ex: “USD” dans TotalCharges) :contentReference[oaicite:11]{index=11}
- Outliers massifs
  - TotalCharges max très élevé (ordre de 1e6), indiquant erreurs ou cas extrêmes :contentReference[oaicite:12]{index=12}

## Nettoyage effectué (exemples visibles)

- Contrôle des statistiques après nettoyage et cohérence moyenne
  - tenure moyen ~ 22 mois
  - MonthlyCharges moyen ~ 60
  - TotalCharges moyen ~ 1800, cohérent avec 60 x 22 :contentReference[oaicite:13]{index=13}
- Gestion de cas incohérents sur TotalCharges (min à 0)
  - Identification de 2 lignes sans données financières exploitables (MonthlyCharges manquant et TotalCharges = 0)
  - Suppression de ces lignes (indices 7313, 36810) :contentReference[oaicite:14]{index=14}

## Modélisation

Le notebook importe et utilise scikit-learn avec une base RandomForestClassifier. :contentReference[oaicite:15]{index=15}

- Split train/test: train_test_split
- Modèle: RandomForestClassifier
- Mesures importées: accuracy_score, classification_report :contentReference[oaicite:16]{index=16}

Note: l’orientation métier exposée dans Business Understanding met l’accent sur la minimisation de l’erreur “prédit reste alors que part”, ce qui correspond en pratique à une recherche de rappel élevé sur la classe churn (ou à un choix de seuil / top-k orienté churn). :contentReference[oaicite:17]{index=17}

## Installation

- Python 3.x
- Dépendances principales
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn :contentReference[oaicite:18]{index=18}

Exemple (environnement local):

- pip install numpy pandas matplotlib seaborn scikit-learn jupyter

## Exécution

- Ouvrir les notebooks dans Jupyter / VS Code
  - Buisness_Understanding.ipynb
  - PROJET.ipynb :contentReference[oaicite:19]{index=19}
- Adapter le chemin du CSV dans PROJET.ipynb
  - Remplacer le chemin Windows local par le chemin de votre machine (ou placer le CSV dans le repo et utiliser un chemin relatif). :contentReference[oaicite:20]{index=20}

## Résultats attendus

- Un diagnostic clair des défauts de qualité (missing, types, incohérences)
- Un dataset corrigé permettant un entraînement sans erreurs de parsing
- Une baseline de classification (Random Forest) et des métriques standards
- Un cadrage business expliquant pourquoi la priorisation des churneurs (lift/top-k) est centrale, au-delà d’une simple accuracy. :contentReference[oaicite:21]{index=21}

## Pistes d’amélioration (logique produit)

- Encodage robuste + pipeline sklearn (ColumnTransformer, OneHotEncoder, SimpleImputer)
- Gestion explicite du déséquilibre de classe (class_weight, métriques PR-AUC, rappel churn)
- Optimisation du seuil en fonction d’une capacité opérationnelle (top 5%, top 10%) et suivi lift/gains chart
- Validation croisée + tuning hyperparamètres
- Export modèle + reproductibilité (requirements.txt, random_state, sauvegarde joblib)

## Samy ASMA

- SamyASM :contentReference[oaicite:22]{index=22}
