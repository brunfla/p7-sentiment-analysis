#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import argparse
import logging
import psutil
import requests
import subprocess
import pickle
import optuna
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# -------------------------
# Configurer le logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Fonction pour afficher les métriques système
# ------------------------------------------------
def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

# ------------------------------------------------
# Fonction pour gérer MLflow localement
# ------------------------------------------------
def handle_mlflow_local(cfg):
    def is_mlflow_running(host="127.0.0.1", port=5000):
        """
        Vérifie si le serveur MLflow est en cours d'exécution.
        """
        url = f"http://{host}:{port}"
        try:
            response = requests.get(url)
            return response.status_code == 200
        except requests.ConnectionError:
            return False

    mlflow_host = cfg.mlflow.trackingUri.split("://")[1].split(":")[0]
    mlflow_port = int(cfg.mlflow.trackingUri.split(":")[-1])

    # Démarrer MLflow si pas déjà lancé
    if not is_mlflow_running(host=mlflow_host, port=mlflow_port):
        subprocess.Popen([
            "mlflow", "server",
            "--host", mlflow_host,
            "--port", str(mlflow_port),
            "--backend-store-uri", cfg.mlflow.storePath if hasattr(cfg.mlflow, 'storePath') else './mlruns'
        ])
        logger.info(
            f"MLflow server started on http://{mlflow_host}:{mlflow_port}. "
            f"Backend store: {cfg.mlflow.storePath if hasattr(cfg.mlflow, 'storePath') else './mlruns'}"
        )

    # Finir les runs MLflow actifs (s'il y en a)
    if mlflow.active_run() is not None:
        logger.info(f"Ending the active run with ID: {mlflow.active_run().info.run_id}")
        mlflow.end_run()

# ------------------------------------------------
# Gestion des targets pour MLflow
# ------------------------------------------------
def handle_mlflow(cfg):
    """
    Configure MLflow en fonction du contexte (local ou Kubernetes) 
    avec gestion des informations de connexion supplémentaires comme l'utilisateur et le mot de passe.
    """
    import os
    import mlflow

    if cfg.mlflow._target_ == "kubernetes":
        logger.info("MLflow configuré pour Kubernetes. Aucun démarrage local requis.")
        mlflow.set_tracking_uri(cfg.mlflow.trackingUri)
        
        # Ajout des informations de connexion
        if hasattr(cfg.mlflow, "username") and hasattr(cfg.mlflow, "password"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow.username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow.password
            logger.info("Ajout des informations d'authentification pour Kubernetes.")
    elif cfg.mlflow._target_ == "local":
        logger.info("Gestion de MLflow local.")
        handle_mlflow_local(cfg)
    else:
        raise ValueError(f"Configuration MLflow inconnue: {cfg.mlflow._target_}")


# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    # --- Parsing d'arguments ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id-file",
        type=str,
        default="mlflow_runid.json",
        help="Chemin du fichier JSON où sauvegarder le run_id MLflow."
    )
    args = parser.parse_args()

    # 1) Afficher les métriques système au démarrage
    log_system_metrics()

    # 2) Récupérer la config Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', '../notebooks/config')
    strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

    # Réinitialiser Hydra si déjà initialisé
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialiser Hydra avec la stratégie choisie
    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # 3) Initialiser MLflow
    handle_mlflow(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment.name)

    # 4) Déterminer le run_name en fonction d'une variable GitHub (si présente) ou de la config
    github_run_id = os.getenv("GITHUB_RUN_ID")  # ou "GITHUB_RUN_NUMBER"
    if github_run_id:
        run_name = f"github_{github_run_id}"
        logger.info(f"Variable GITHUB_RUN_ID détectée, run_name={run_name}")
    else:
        # Fallback: on utilise le runName configuré dans Hydra
        run_name = cfg.mlflow.experiment.get(
            'runName',
            f"{getattr(cfg.model, 'type', 'unknown-model')}-{os.getenv('HYDRA_STRATEGY', 'validation-quick')}"
        )

    # 5) Démarrer un run MLflow
    with mlflow.start_run(run_name=run_name) as run:
        mlflow_run_id = run.info.run_id
        logger.info(f"MLflow run démarré avec ID: {mlflow_run_id}")

        # ------------------------------------------------
        # 0) TIRER (PULL) LES DONNÉES DEPUIS DVC
        # ------------------------------------------------
        # Exécution dvc pull pour être sûr d'avoir la dernière version des données
        # Si votre dataset est géré par DVC, vous pouvez préciser le chemin .dvc ou laisser par défaut
        # subprocess.run(["dvc", "pull", "data/vectorized_data.dvc"], check=True)
        logger.info("DVC pull (optionnel) - Assurez-vous que vos données sont à jour localement.")

        # On peut aussi logger la version DVC (commit hash, etc.) comme paramètre
        try:
            dvc_version = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            mlflow.log_param("dvc_dataset_version", dvc_version)
        except Exception as e:
            logger.warning(f"Impossible de récupérer la version DVC (ou Git). Erreur: {e}")

        # ------------------------------------------------
        # 1) CHARGER LES DONNÉES VECTORISÉES (via un fichier géré par DVC)
        # ------------------------------------------------
        vectorized_data_path = cfg.vectorizer.outputData  # ex: "../data/output/vectorized_data.pkl"
        if not os.path.exists(vectorized_data_path):
            logger.error(
                f"ÉCHEC : le fichier vectorisé {vectorized_data_path} est introuvable. "
                "Veuillez d'abord exécuter l'étape de vectorisation, ou vérifier DVC."
            )
            sys.exit(1)

        logger.info(f"Chargement des données vectorisées depuis {vectorized_data_path}...")
        with open(vectorized_data_path, 'rb') as f:
            X, y = pickle.load(f)
        logger.info(
            f"Données vectorisées chargées avec succès. "
            f"Taille X: {X.shape if hasattr(X, 'shape') else len(X)}, "
            f"Taille y: {len(y)}"
        )

        # ------------------------------------------------
        # 2) PARTITIONNER (si pas déjà fait)
        # ------------------------------------------------
        partition_cfg = cfg.partitioner
        split_path = getattr(partition_cfg, "outputSplit", None)
        existing_split = getattr(partition_cfg, "existingSplitData", False)

        X_train, y_train = None, None
        X_val, y_val = None, None
        X_test, y_test = None, None
        folds = None

        if split_path and existing_split and os.path.exists(split_path):
            logger.info(f"[Partitioner] Chargement d'un split existant depuis {split_path}...")
            with open(split_path, "rb") as f:
                if partition_cfg._target_ == "trainValTest":
                    (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(f)
                    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
                elif partition_cfg._target_ == "trainTest":
                    (X_train, y_train, X_test, y_test) = pickle.load(f)
                    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
                elif partition_cfg._target_ == "crossValidation":
                    (folds, X, y) = pickle.load(f)
                    logger.info(f"Nombre de folds: {len(folds)} (cross-validation)")
                else:
                    raise ValueError(f"Partitioner non reconnu: {partition_cfg._target_}")

        else:
            logger.info("[Partitioner] Pas de split existant. Génération d'un nouveau split...")
            if partition_cfg._target_ == "trainValTest":
                X_train_full, X_test, y_train_full, y_test = train_test_split(
                    X, y,
                    test_size=partition_cfg.testSize,
                    random_state=partition_cfg.randomSeed
                )
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train_full, y_train_full,
                    test_size=partition_cfg.validationSize,
                    random_state=partition_cfg.randomSeed
                )
                logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

                # Sauvegarder le split
                if split_path:
                    with open(split_path, "wb") as f:
                        pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
                    logger.info(f"[Partitioner] Nouveau split trainValTest sauvegardé dans {split_path}")

            elif partition_cfg._target_ == "trainTest":
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=partition_cfg.testSize,
                    random_state=partition_cfg.randomSeed
                )
                logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

                if split_path:
                    with open(split_path, "wb") as f:
                        pickle.dump((X_train, y_train, X_test, y_test), f)
                    logger.info(f"[Partitioner] Nouveau split trainTest sauvegardé dans {split_path}")

            elif partition_cfg._target_ == "crossValidation":
                kfold = KFold(
                    n_splits=partition_cfg.folds,
                    shuffle=True,
                    random_state=partition_cfg.randomSeed
                )
                folds = list(kfold.split(X, y))
                logger.info(f"Nombre de folds: {len(folds)}")

                if split_path:
                    with open(split_path, "wb") as f:
                        pickle.dump((folds, X, y), f)
                    logger.info(f"[Partitioner] Nouveau split crossValidation sauvegardé dans {split_path}")

            else:
                raise ValueError(f"Partition de découpage non reconnue: {partition_cfg._target_}")

        # ------------------------------------------------
        # 3) OPTIMISATION DES HYPERPARAMÈTRES
        # ------------------------------------------------
        if cfg.hyperparameterOptimization._target_ == "gridSearch":
            if partition_cfg._target_ == "trainTest":
                logger.info("Validation non applicable pour 'trainTest'. Hyperparamètres par défaut.")
                best_params = cfg.model.parameters
                model = LogisticRegression(**best_params)
            else:
                if not hasattr(cfg.validation, 'crossValidation'):
                    raise ValueError("La validation croisée est requise pour GridSearch, mais n'est pas définie.")

                param_grid = dict(cfg.hyperparameterOptimization.paramGrid)
                if (partition_cfg._target_ == "crossValidation"
                    and hasattr(cfg.validation, 'crossValidation')
                    and hasattr(cfg.validation.crossValidation, 'folds')):
                    cv = KFold(
                        n_splits=cfg.validation.crossValidation.folds,
                        random_state=partition_cfg.randomSeed,
                        shuffle=True
                    )
                else:
                    cv = 5  # fallback par défaut

                grid_search = GridSearchCV(
                    estimator=LogisticRegression(),
                    param_grid=param_grid,
                    cv=cv,
                    verbose=cfg.hyperparameterOptimization.verbosityLevel,
                    n_jobs=cfg.hyperparameterOptimization.parallelJobs,
                )
                grid_search.fit(X_train, y_train)
                best_params = grid_search.best_params_
                model = grid_search.best_estimator_
                logger.info(f"Meilleurs hyperparamètres trouvés: {best_params}")

        elif cfg.hyperparameterOptimization._target_ == "optuna":
            def objective(trial):
                penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
                C = trial.suggest_float("C", 0.1, 10, log=True)
                solver = trial.suggest_categorical("solver", ["liblinear", "saga"])
                opt_model = LogisticRegression(penalty=penalty, C=C, solver=solver)
                return cross_val_score(opt_model, X_train, y_train, cv=partition_cfg.folds).mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=cfg.hyperparameterOptimization.optuna.trialCount)
            best_params = study.best_params
            model = LogisticRegression(**best_params)
            logger.info(f"Meilleurs hyperparamètres trouvés par Optuna: {best_params}")

        else:
            raise ValueError("Méthode d'optimisation des hyperparamètres non reconnue.")

        # ------------------------------------------------
        # 4) ENTRAÎNER LE MODÈLE
        # ------------------------------------------------
        model.fit(X_train, y_train)
        logger.info("Modèle entraîné avec succès.")

        # Évaluation sur la validation (si disponible)
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            logger.info(f"Score de validation: {val_score}")

        # On NE fait PAS l’évaluation sur X_test ici (séparation train vs test)

        # ------------------------------------------------
        # 5) ENREGISTRER LE MODÈLE DANS MLflow
        # ------------------------------------------------
        mlflow.log_params(best_params)
        mlflow.sklearn.log_model(model, artifact_path="model")
        logger.info("Modèle enregistré dans MLflow.")

        # Afficher les métriques système en fin de run
        log_system_metrics()

    # ------------------------------------------------
    # 6) APRÈS LE with MLflow : SAUVEGARDER LE RUN_ID
    # ------------------------------------------------
    with open(args.run_id_file, "w") as f:
        json.dump({"mlflow_run_id": mlflow_run_id}, f)
    logger.info(f"MLflow run_id sauvegardé dans {args.run_id_file}")


if __name__ == "__main__":
    main()
