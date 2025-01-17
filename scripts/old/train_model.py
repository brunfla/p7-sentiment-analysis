import os
import json
import logging
import mlflow
import yaml
import subprocess
from partitioned_data import PartitionedData
from training_model import train_lstm, train_logistic_regression, handle_training
from keras.callbacks import Callback
from scikeras.wrappers import KerasClassifier

from train_lstm import train_lstm

# <<-- Import pour la sauvegarde locale
import joblib

# Configurer le logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Récupérer les hashes Git et DVC
# ------------------------------------------------
def get_git_hash():
    """Récupère le hash Git actuel."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode()
        logger.info(f"Git commit hash : {git_commit}")
        return git_commit
    except subprocess.CalledProcessError:
        logger.error("Impossible de récupérer le hash Git.")
        return None

# ------------------------------------------------
# Callback personnalisé pour enregistrer les métriques dans MLflow
# ------------------------------------------------
class MLflowMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            for key, value in logs.items():
                mlflow.log_metric(key, value, step=epoch)

# ------------------------------------------------
# Gestion des entraînements
# ------------------------------------------------
def handle_mlflow_logging(cfg, model):
    """
    Configure et enregistre les paramètres et le modèle dans MLflow,
    puis sauvegarde également le modèle localement dans data/output/model.pkl.
    """
    # Récupérer le hash Git
    git_hash = get_git_hash()

    # Enregistrer les hyperparamètres dans MLflow (si cfg.model.parameters existe)
    if hasattr(cfg.model, "parameters"):
        mlflow.log_params(cfg.model.parameters)

    if git_hash:
        mlflow.set_tag("git_commit", git_hash)

    # Enregistrement du modèle dans MLflow (format scikit-learn)
    mlflow.sklearn.log_model(model, artifact_path="model")
    logger.info("Modèle enregistré dans MLflow.")

    # Sauvegarder le run ID dans un fichier
    run_id_file = "data/output/mlflow_id.json"
    mlflow_run_id = mlflow.active_run().info.run_id
    with open(run_id_file, "w") as f:
        json.dump({"id": mlflow_run_id, "git_commit": git_hash}, f)
    logger.info(f"MLflow run_id sauvegardé dans {run_id_file}")

def handle_training(partitioned_data, cfg):
    """
    Gère le processus d'entraînement en fonction de la configuration fournie.
    """
    model_type = cfg.model.type
    training_cfg = cfg.get("training", {})
    tuning_cfg = cfg.get("tuning", {})
    mlflow_cfg = cfg.get("mlflow", {})

    if model_type == "bidirectional_lstm":
        logger.info("Entraînement d'un modèle LSTM.")
        model = train_lstm(cfg.model, training_cfg, tuning_cfg, mlflow_cfg, partitioned_data)
    elif model_type == "logistic_regression":
        logger.info("Entraînement d'un modèle de régression logistique.")
        model = train_logistic_regression(cfg.model, tuning_cfg, mlflow_cfg, partitioned_data)
    else:
        raise ValueError(f"Type de modèle non supporté : {model_type}")

    return model

# ------------------------------------------------
# Orchestration principale
# ------------------------------------------------
def main():
    # Initialiser Hydra
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra

    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # Charger les données partitionnées
    dataset_path = cfg.training.input
    logger.info(f"Chargement des données depuis {dataset_path}...")
    partitioned_data = PartitionedData.load(dataset_path)

    logger.info(f"Données chargées avec succès. Type de partition : {partitioned_data.partition_type}")

    # Gérer l'entraînement
    model = handle_training(partitioned_data, cfg)

    # Sauvegarde locale du modèle (Pickle/Joblib)
    local_model_path = "data/output/model.pkl"
    joblib.dump(model, local_model_path)
    logger.info(f"Modèle localement sauvegardé dans {local_model_path}")

    # Enregistrer le modèle dans MLflow et localement
    handle_mlflow_logging(cfg, model)

if __name__ == "__main__":
    main()
