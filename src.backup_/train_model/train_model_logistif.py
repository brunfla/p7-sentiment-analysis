import os
import mlflow
import mlflow.sklearn
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import load_npz
import pandas as pd
import json
import sys

# Charger les paramètres depuis params.yaml
def load_params(params_file, section):
    import yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

# Configurer le logger
def get_logger(name):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Obtenir le logger
logger = get_logger(__name__)

def train_logistic_regression_incremental(input_train_file, input_val_file, output_dir, model_type, tuning_cfg, mlflow_cfg, chunksize=10000):
    """
    Entraîne un modèle de régression logistique de manière incrémentale avec des données TF-IDF prétraitées.

    Args:
        input_train_file (str): Chemin vers le fichier d'entraînement TF-IDF.
        input_val_file (str): Chemin vers le fichier de validation TF-IDF.
        output_dir (str): Répertoire pour sauvegarder le modèle.
        model_type (str): Type de modèle, ici "logistic_regression".
        tuning_cfg (dict): Configuration des hyperparamètres du modèle.
        mlflow_cfg (dict): Configuration pour le suivi avec MLflow.
        chunksize (int): Taille des lots pour l'entraînement incrémental.
    """
    # Configuration initiale
    logger.info(f"Chargement des données d'entraînement depuis : {input_train_file}")
    logger.info(f"Chargement des données de validation depuis : {input_val_file}")
    mlflow.set_tracking_uri(mlflow_cfg["trackingUri"])
    mlflow.set_experiment(mlflow_cfg["experiment"]["name"])

    model = SGDClassifier(loss="log_loss", max_iter=tuning_cfg.get("max_iter", 1000), tol=tuning_cfg.get("tol", 1e-3))

    total_samples = 0
    all_predictions = []
    all_labels = []

    with mlflow.start_run(run_name=mlflow_cfg["experiment"]["run"]["name"], description=mlflow_cfg["experiment"]["run"]["description"]) as run:
        mlflow.set_tags(mlflow_cfg["experiment"]["run"].get("tags", {}))

        # Charger les données d'entraînement
        train_data = pd.read_pickle(input_train_file)
        X_train = train_data['features']
        y_train = train_data['labels']

        # Entraînement incrémental
        for start in range(0, X_train.shape[0], chunksize):
            end = min(start + chunksize, X_train.shape[0])
            X_chunk = X_train[start:end]
            y_chunk = y_train[start:end]

            if total_samples == 0:
                model.partial_fit(X_chunk, y_chunk, classes=pd.unique(y_train))
            else:
                model.partial_fit(X_chunk, y_chunk)

            total_samples += len(y_chunk)

        # Évaluation sur les données de validation
        val_data = pd.read_pickle(input_val_file)
        X_val = val_data['features']
        y_val = val_data['labels']
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        logger.info(f"Accuracy sur les données de validation : {val_accuracy:.4f}")
        mlflow.log_metric("val_accuracy", val_accuracy)

        # Sauvegarde du modèle si l'accuracy atteint le seuil minimal
        min_accuracy = mlflow_cfg["experiment"]["run"].get("min_accuracy", 0.7)
        if val_accuracy >= min_accuracy:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "model.pkl")
            mlflow.sklearn.save_model(model, model_path)
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info(f"Modèle sauvegardé dans : {model_path}")
        else:
            logger.warning(f"Modèle non sauvegardé car l'accuracy {val_accuracy:.4f} est inférieure au seuil minimal {min_accuracy:.4f}")

        # Enregistrer l'ID de la run MLflow
        run_id = run.info.run_id
        mlflow_id_path = os.path.join(output_dir, "mlflow_id.json")
        with open(mlflow_id_path, "w") as f:
            json.dump({"run_id": run_id}, f)
        logger.info(f"ID de la run MLflow sauvegardé dans : {mlflow_id_path}")

if __name__ == "__main__":
    # Charger les paramètres
    params_file = "params.yaml"
    section = "train_logistic_tfidf"

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        input_train_file = params["input_train_file"]
        input_val_file = params["input_val_file"]
        output_dir = params["output_dir"]
        model_type = params["model_type"]
        tuning_cfg = params.get("tuning", {})
        mlflow_cfg = params.get("mlflow", {})
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Entraînement incrémental
    train_logistic_regression_incremental(input_train_file, input_val_file, output_dir, model_type, tuning_cfg, mlflow_cfg)
