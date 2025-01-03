import os
import json
import argparse
import logging
import psutil
import requests
import subprocess
import pickle
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
# Gestion des targets pour MLflow
# ------------------------------------------------
def handle_mlflow(cfg):
    """
    Configure MLflow en fonction du contexte (local ou Kubernetes)
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
        mlflow.set_tracking_uri(cfg.mlflow.trackingUri)
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

    # 4) Charger les partitions générées par generate_split.py
    split_path = cfg.partitioner.outputSplit
    if not os.path.exists(split_path):
        logger.error(f"ÉCHEC : le fichier partitionné {split_path} est introuvable. Exécutez generate_split.py.")
        exit(1)

    logger.info(f"Chargement des données partitionnées depuis {split_path}...")
    with open(split_path, 'rb') as f:
        data = pickle.load(f)

    if cfg.partitioner._target_ == "trainValTest":
        X_train, y_train, X_val, y_val, X_test, y_test = data
    elif cfg.partitioner._target_ == "trainTest":
        X_train, y_train, X_test, y_test = data
    else:
        raise ValueError(f"Partition target inconnue : {cfg.partitioner._target_}")

    # 5) Entraîner le modèle
    model = LogisticRegression(**cfg.model.parameters)
    model.fit(X_train, y_train)
    logger.info("Modèle entraîné avec succès.")

    # Évaluer le modèle sur les données de validation (si disponibles)
    if 'X_val' in locals() and 'y_val' in locals():
        val_score = model.score(X_val, y_val)
        logger.info(f"Score de validation : {val_score}")
        mlflow.log_metric("val_score", val_score)

    # 6) Enregistrer le modèle dans MLflow
    mlflow.log_params(cfg.model.parameters)
    mlflow.sklearn.log_model(model, artifact_path="model")
    logger.info("Modèle enregistré dans MLflow.")

    # 7) Sauvegarder le run_id dans un fichier
    mlflow_run_id = mlflow.active_run().info.run_id
    with open(args.run_id_file, "w") as f:
        json.dump({"mlflow_run_id": mlflow_run_id}, f)
    logger.info(f"MLflow run_id sauvegardé dans {args.run_id_file}")

if __name__ == "__main__":
    main()
