import os
import json
import logging
import pickle
import mlflow
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tuning_model import apply_tuning
from training_model import get_callbacks, train_lstm, train_logistic_regression

# Configurer le logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Charger les données partitionnées
# ------------------------------------------------
def load_partitioned_data(split_path, partition_target):
    if not os.path.exists(split_path):
        logger.error(f"ÉCHEC : le fichier partitionné {split_path} est introuvable. Exécutez generate_split.py.")
        exit(1)

    logger.info(f"Chargement des données partitionnées depuis {split_path}...")
    with open(split_path, 'rb') as f:
        data = pickle.load(f)

    if partition_target == "trainValTest":
        return data  # X_train, y_train, X_val, y_val, X_test, y_test
    elif partition_target == "trainTest":
        X_train, y_train, X_test, y_test = data
        return X_train, y_train, None, None, X_test, y_test  # X_val, y_val définis comme None
    else:
        raise ValueError(f"Partition target inconnue : {partition_target}")

# ------------------------------------------------
# Orchestration principale
# ------------------------------------------------
def main():
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)

    # Charger les configurations
    cfg = compose(config_name="model.yaml")
    training_cfg = {
        "earlyStopping": compose(config_name="training/earlyStopping.yaml"),
        "learningRateScheduler": compose(config_name="training/learningRate.yaml"),
    }
    tuning_type_cfg = compose(config_name="tuning/type/optuna.yaml")
    tuning_parameters_cfg = compose(config_name="tuning/parameters.yaml")

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # Charger les données
    data = load_partitioned_data("data/output/train_ready_data.pkl", cfg.partitioner._target_)

    # Entraîner le modèle
    if cfg.model.type == "logistic_regression":
        model = train_logistic_regression(cfg, tuning_type_cfg, tuning_parameters_cfg, data)
    elif cfg.model.type == "bidirectional_lstm":
        model = train_lstm(cfg, training_cfg, tuning_type_cfg, tuning_parameters_cfg, data)
    else:
        raise ValueError(f"Type de modèle non supporté : {cfg.model.type}")

    # Sauvegarder le modèle dans MLflow
    mlflow.log_params(cfg.model)
    mlflow.log_artifact("model")
    logger.info("Modèle enregistré dans MLflow.")

    # Sauvegarder le run ID dans MLflow
    run_id_file = "data/output/mlflow_id.json"
    mlflow_run_id = mlflow.active_run().info.run_id
    with open(run_id_file, "w") as f:
        json.dump({"id": mlflow_run_id}, f)
    logger.info(f"MLflow run_id sauvegardé dans {run_id_file}")


if __name__ == "__main__":
    main()