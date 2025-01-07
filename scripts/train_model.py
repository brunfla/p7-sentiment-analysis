import os
import json
import logging
import pickle
import psutil
import mlflow
from sklearn.linear_model import LogisticRegression
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from sklearn.metrics import accuracy_score, f1_score
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
# MLFLOW
# ------------------------------------------------
def handle_mlflow(cfg):
    if cfg.mlflow._target_ == "kubernetes":
        logger.info("MLflow configuré pour Kubernetes. Aucun démarrage local requis.")
        mlflow.set_tracking_uri(cfg.mlflow.trackingUri)
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
# Fonction pour charger les données partitionnées
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
# Entraîner et enregistrer un modèle Logistic Regression
# ------------------------------------------------
def train_logistic_regression(cfg, X_train, y_train, X_val=None, y_val=None):
    model = LogisticRegression(**cfg.model.parameters)
    model.fit(X_train, y_train)
    logger.info("Modèle Logistic Regression entraîné avec succès.")

    if X_val is not None and y_val is not None:
        val_score = model.score(X_val, y_val)
        logger.info(f"Score de validation : {val_score}")
        mlflow.log_metric("val_score", val_score)

    mlflow.log_params(cfg.model.parameters)
    mlflow.sklearn.log_model(model, artifact_path="model")
    logger.info("Modèle Logistic Regression enregistré dans MLflow.")

# ------------------------------------------------
# Entraîner et enregistrer un modèle Bidirectional LSTM
# ------------------------------------------------
def train_bidirectional_lstm(cfg, X_train_pad, y_train, X_val_pad=None, y_val=None):
    metrics = list(cfg.model.metrics)
    
    model_lstm = Sequential([
        Embedding(cfg.model.vocab_size, cfg.model.embedding_dim, weights=None,
                  input_length=cfg.model.max_length, trainable=False),
        Bidirectional(LSTM(cfg.model.lstm_units[0], return_sequences=True)),
        Dropout(cfg.model.dropout_rate),
        Bidirectional(LSTM(cfg.model.lstm_units[1])),
        Dense(cfg.model.dense_units, activation='relu'),
        Dense(1, activation=cfg.model.output_activation)
    ])
    model_lstm.compile(optimizer=cfg.model.optimizer, loss=cfg.model.loss_function, metrics=metrics)
    model_lstm.fit(X_train_pad, y_train, validation_split=0.2,
                   epochs=cfg.model.epochs, batch_size=cfg.model.batch_size)
    logger.info("Modèle Bidirectional LSTM entraîné avec succès.")

    mlflow.keras.log_model(model_lstm, artifact_path="model")
    logger.info("Modèle Bidirectional LSTM enregistré dans MLflow.")

# ------------------------------------------------
# Fonction pour entraîner un modèle
# ------------------------------------------------
def train_model(cfg, data):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    if cfg.model.type == "logistic_regression":
        train_logistic_regression(cfg, X_train, y_train, X_val, y_val)
    elif cfg.model.type == "bidirectional_lstm":
        train_bidirectional_lstm(cfg, X_train, y_train, X_val, y_val)
    else:
        raise ValueError(f"Type de modèle non supporté : {cfg.model.type}")

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    log_system_metrics()

    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    handle_mlflow(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment.name)

    data = load_partitioned_data("data/output/train_ready_data.pkl", cfg.partitioner._target_)
    train_model(cfg, data)

    run_id_file = "data/output/mlflow_id.json"
    mlflow_run_id = mlflow.active_run().info.run_id
    with open(run_id_file, "w") as f:
        json.dump({"id": mlflow_run_id}, f)
    logger.info(f"MLflow run_id sauvegardé dans {run_id_file}")

if __name__ == "__main__":
    main()