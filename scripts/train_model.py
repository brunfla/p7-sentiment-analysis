#!/usr/bin/env python
# coding: utf-8

# # Modèle simple : Régression Logistique

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Initialiser Hydra avec une nouvelle configuration
initialize(config_path="../notebook/config", version_base=None)
cfg = compose(config_name="simple-model-ci-github")

# Afficher la configuration globale
print("Configuration globale :")
print(cfg)

# Charger les paramètres du modèle
model_config = cfg.model
print(f"\nModèle sélectionné : {model_config.name}")
print(f"Paramètres du modèle : {model_config.parameters}")

# Charger le dataset
dataset_path = cfg.dataset.path  # Utiliser la clé correcte
df = pd.read_csv(dataset_path)
print(f"\nDataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")


# #### Charger MLflow

# In[22]:
import requests
import subprocess
import mlflow
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra


def is_mlflow_running(host="127.0.0.1", port=5000):
    """
    Vérifie si le serveur MLflow est en cours d'exécution.
    """
    url = f"http://{host}:{port}"
    try:
        response = requests.get(url)
        return response.status_code == 200  # Vérifie le code de réponse HTTP
    except requests.ConnectionError:
        return False


# Vérifier si MLflow est en cours d'exécution
mlflow_host = cfg.model.mlflow.trackingUri.split("://")[1].split(":")[0]
mlflow_port = int(cfg.model.mlflow.trackingUri.split(":")[-1])

if not is_mlflow_running(host=mlflow_host, port=mlflow_port):
    subprocess.Popen(["mlflow", "server", "--host", mlflow_host, "--port", str(mlflow_port)])
    print(f"MLflow server started on http://{mlflow_host}:{mlflow_port}.")
else:
    print(f"MLflow server is already running on http://{mlflow_host}:{mlflow_port}.")
    # Vérifier si un run est déjà actif
    if mlflow.active_run() is not None:
        print(f"Ending the active run with ID: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

# Configurer MLflow
print("Démarrer un run.")
mlflow.set_tracking_uri(cfg.model.mlflow.trackingUri)
mlflow.set_experiment(cfg.model.mlflow.experiment.name)
mlflow.start_run(run_name=cfg.model.mlflow.experiment.run.name)
print("MLflow run started.")


# ## Etape 2 : Prétraitement 

# #### Suppression des lignes vides

# In[23]:


# Suppression des lignes vides
df[df["tweet"].isna() | (df["tweet"] == "")]
df = df[~(df['tweet'].isna() | (df['tweet'] == ""))]


# #### Vectorization TF-IDF

# In[24]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Vérifier le type de vectorizer configuré
if cfg.vectorizer._target_ == "tfidfVectorizer":
    vectorizer = TfidfVectorizer(
        stop_words=cfg.vectorizer.stopWords,
        max_features=cfg.vectorizer.maxFeatures,
        ngram_range=tuple(cfg.vectorizer.ngramRange)
    )
else:
    raise KeyError("La configuration 'tfidVectorizer' est absente ou mal définie dans 'vectorizer'.")
    
# Appliquer fit_transform sur les tweets
X = vectorizer.fit_transform(df['tweet'])

print(f"TF-IDF vectorisation terminée avec {X.shape[1]} caractéristiques.")


# ## Etape 3 : Entraînement et Validation

# #### Découpage des données (`dataSplitting`)

# In[25]:


from sklearn.model_selection import train_test_split, KFold

# Encodage binaire de la cible
y = df['id'].apply(lambda x: 1 if x == 4 else 0)

# Vérifier la stratégie sélectionnée
if cfg.strategy._target_ == "trainValTest":
    # Charger les paramètres de la stratégie
    params = cfg.strategy
    # Découpage Train/Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=params.testSize, random_state=params.randomSeed
    )
    # Découpage Train/Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=params.validationSize, random_state=params.randomSeed
    )
    print(f"Données découpées avec la stratégie 'trainValTest':")
    print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

elif cfg.strategy._target_ == "trainTest":
    # Charger les paramètres de la stratégie
    params = cfg.strategy
    # Découpage Train/Test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=params.testSize, random_state=params.randomSeed
    )
    X_val, y_val = None, None  # Pas de validation pour cette stratégie
    print(f"Données découpées avec la stratégie 'trainTest':")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

elif cfg.strategy._target_ == "crossValidation":
    # Charger les paramètres de la stratégie
    params = cfg.strategy
    kfold = KFold(n_splits=params.folds, shuffle=True, random_state=params.randomSeed)
    folds = list(kfold.split(X, y))
    print(f"Données découpées avec la stratégie 'crossValidation':")
    print(f"Nombre de folds: {len(folds)}")
    # Exemple d'accès au premier fold
    train_idx, val_idx = folds[0]
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_test, y_test = None, None  # Pas de test explicite pour cette stratégie
    print(f"Premier fold - Train: {X_train.shape}, Validation: {X_val.shape}")

else:
    raise ValueError(f"Stratégie de découpage des données '{cfg.strategy._target_}' non reconnue.")


# #### Optimisation des Hyperparamètres

# - Méthodes d'optimisation :
#     - GridSearchCV :
#         - Explore toutes les combinaisons d’hyperparamètres définis dans une grille.
#         - Utilise la validation croisée pour évaluer chaque combinaison.
#         - Retourne :
#             - Les meilleurs hyperparamètres (best_params_).
#             - Le modèle entraîné avec ces hyperparamètres (best_estimator_).
# 
#     - Optuna :
#         - Optimisation bayésienne avec espace de recherche dynamique.
#         - Fonction objectif :
#             - Entraîne un modèle avec des hyperparamètres suggérés.
#             - Évalue la performance via validation croisée (cross_val_score).
#         - Retourne :
#             - Les meilleurs hyperparamètres (best_params).
#             - Un modèle final configuré avec ces paramètres.

# In[26]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
import optuna

# Paramètres pour l'optimisation
if cfg.hyperparameterOptimization._target_ == "gridSearch":
    param_grid = dict(cfg.hyperparameterOptimization.paramGrid)

    # Définir les folds ou l'ensemble de validation
    cv = (
        cfg.validation.crossValidation.folds
        if cfg.strategy._target_ == "crossValidation"
        else [(list(range(X_train.shape[0])), list(range(X_val.shape[0])))]
    )

    # Création de GridSearchCV
    grid_search = GridSearchCV(
        estimator=LogisticRegression(),
        param_grid=param_grid,
        cv=cv,
        verbose=cfg.hyperparameterOptimization.verbosityLevel,
        n_jobs=cfg.hyperparameterOptimization.parallelJobs,
    )
    grid_search.fit(X_train, y_train)

    # Récupération des meilleurs paramètres
    best_params = grid_search.best_params_
    model = grid_search.best_estimator_
    print(f"Best Parameters Found by GridSearch: {best_params}")

    # Évaluation finale sur l'ensemble de test
    if X_test is not None and y_test is not None:
        test_score = model.score(X_test, y_test)
        print(f"Test Accuracy: {test_score}")

elif cfg.hyperparameterOptimization._target_ == "optuna":
    def objective(trial):
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        C = trial.suggest_float("C", 0.1, 10, log=True)
        solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
        max_iter = trial.suggest_int("max_iter", 100, 1000)

        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=max_iter)

        if cfg.strategy._target_ == "crossValidation":
            scores = cross_val_score(model, X_train, y_train, cv=cfg.validation.crossValidation.folds)
            return scores.mean()
        elif cfg.strategy._target_ == "trainValTest":
            model.fit(X_train, y_train)
            return model.score(X_val, y_val)

    # Lancer l'optimisation avec Optuna
    study = optuna.create_study(direction=cfg.hyperparameterOptimization.optuna.optimizationDirection)
    study.optimize(objective, n_trials=cfg.hyperparameterOptimization.optuna.trialCount, timeout=cfg.hyperparameterOptimization.optuna.timeLimitSeconds)

    # Récupération des meilleurs paramètres
    best_params = study.best_params
    model = LogisticRegression(**best_params)
    print(f"Best Parameters Found by Optuna: {best_params}")

    # Évaluation finale sur l'ensemble de test
    if X_test is not None and y_test is not None:
        model.fit(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f"Test Accuracy: {test_score}")

else:
    raise ValueError("Unsupported hyperparameter optimization method")


