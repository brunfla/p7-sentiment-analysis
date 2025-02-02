import os
import sys
import logging
import pickle
import json
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# --------------------------------------------------------------------
# Charger les paramètres depuis params.yaml
# --------------------------------------------------------------------
def load_params(params_file, section):
    import yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

# --------------------------------------------------------------------
# Configurer un logger
# --------------------------------------------------------------------
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# --------------------------------------------------------------------
# Fonction principale d'entraînement
# --------------------------------------------------------------------
def train_logistic_regression_incremental(params):
    """
    Entraîne un modèle de régression logistique de manière incrémentale 
    sur des données TF-IDF pré-traitées et logue dans MLflow.

    Paramètres attendus dans 'params':
      - input_train_vec (str) : chemin .pkl pour x_train (vectors, ids)
      - input_train_labels (str) : chemin .csv pour y_train (id, label)
      - input_val_vec (str) : chemin .pkl pour x_val
      - input_val_labels (str) : chemin .csv pour y_val
      - output_dir (str) : répertoire pour sauvegarder le modèle + mlflow_id.json
      - model_type (str) : type du modèle (pas forcément utilisé ici, ex: 'logistic_sgd')
      - tuning (dict) : hyperparamètres (max_iter, tol, etc.)
      - mlflow (dict) : config MLflow (trackingUri, experiment, etc.)
      - chunk_size (int) : taille des chunks pour partial_fit
    """

    # 1) Récupération des chemins depuis params
    try:
        input_train_vec   = params["input_train_vec"]
        input_train_labels= params["input_train_labels"]
        input_val_vec     = params["input_val_vec"]
        input_val_labels  = params["input_val_labels"]
        output_dir        = params["output_dir"]
        model_type        = params["model_type"]  # ex: "logistic_sgd"
        tuning_cfg        = params.get("tuning", {})
        mlflow_cfg        = params.get("mlflow", {})
        chunk_size        = params.get("chunk_size", 10000)
    except KeyError as e:
        logger.error(f"Clé manquante dans les paramètres : {e}")
        sys.exit(1)

    # 2) Configuration MLflow
    mlflow.set_tracking_uri(mlflow_cfg["trackingUri"])
    mlflow.set_experiment(mlflow_cfg["experiment"]["name"])

    # 3) Initialiser le modèle
    #    SGDClassifier en mode 'log_loss' (régression logistique), partial_fit possible
    model = SGDClassifier(
        loss="log_loss",
        max_iter=tuning_cfg.get("max_iter", 1000),
        tol=tuning_cfg.get("tol", 1e-3)
    )

    logger.info("=== Chargement des données d'entraînement ===")
    # 4) Charger x_train (vecteurs)
    with open(input_train_vec, "rb") as f:
        train_data = pickle.load(f)  # => dict { "vectors": sparse_matrix, "ids": [...] }
    X_train = train_data["vectors"]
    train_ids = np.array(train_data["ids"])  # liste d'IDs correspondant aux lignes

    # 5) Charger y_train (labels) 
    df_y_train = pd.read_csv(input_train_labels)  # => contient colonnes ["id", "label"] (ou autre nom)
    # Synchroniser l'ordre (on suppose que train_ids figure dans df_y_train["id"])
    # On créé un dictionnaire {id: label}
    label_map = dict(zip(df_y_train["id"], df_y_train["label"]))
    
    # Convertir train_ids en label via le dict
    y_train = []
    for tid in train_ids:
        # tid doit exister dans label_map
        y_train.append(label_map[tid])
    y_train = np.array(y_train)

    # On vérifie la cohérence
    if X_train.shape[0] != len(y_train):
        logger.error("Nombre d'échantillons différent entre X_train et y_train.")
        sys.exit(1)

    logger.info(f"Train set: {X_train.shape[0]} échantillons, {X_train.shape[1]} features.")

    # ----------------------------------------------------------------
    # Démarrage du run MLflow
    # ----------------------------------------------------------------
    with mlflow.start_run(run_name=mlflow_cfg["experiment"]["run"]["name"], 
                          description=mlflow_cfg["experiment"]["run"].get("description")) as run:

        mlflow.set_tags(mlflow_cfg["experiment"]["run"].get("tags", {}))

        # 6) Entraînement incrémental par chunks
        total_samples = 0
        n_samples = X_train.shape[0]

        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            X_chunk = X_train[start:end]
            y_chunk = y_train[start:end]

            if total_samples == 0:
                # Première itération : on doit passer la liste complète des classes
                classes_ = np.unique(y_train)
                model.partial_fit(X_chunk, y_chunk, classes=classes_)
            else:
                model.partial_fit(X_chunk, y_chunk)

            total_samples += (end - start)
            logger.info(f"partial_fit sur {end - start} échantillons. Total = {total_samples}")

        # 7) Évaluation sur le jeu de validation
        logger.info("=== Évaluation sur données de validation ===")
        with open(input_val_vec, "rb") as f:
            val_data = pickle.load(f)  # => { "vectors":..., "ids":[...] }
        X_val = val_data["vectors"]
        val_ids = np.array(val_data["ids"])

        df_y_val = pd.read_csv(input_val_labels)  # => ["id","label"]
        # Synchro
        label_map_val = dict(zip(df_y_val["id"], df_y_val["label"]))
        y_val = []
        for vid in val_ids:
            y_val.append(label_map_val[vid])
        y_val = np.array(y_val)

        if X_val.shape[0] != len(y_val):
            logger.error("Nombre d'échantillons différent entre X_val et y_val.")
            sys.exit(1)

        logger.info(f"Val set: {X_val.shape[0]} échantillons, {X_val.shape[1]} features.")

        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        logger.info(f"Accuracy sur Validation : {val_accuracy:.4f}")
        mlflow.log_metric("val_accuracy", val_accuracy)

        # 8) Sauvegarde du modèle si performance OK
        min_accuracy = mlflow_cfg["experiment"]["run"].get("min_accuracy", 0.7)
        if val_accuracy >= min_accuracy:
            os.makedirs(output_dir, exist_ok=True)
            model_path = os.path.join(output_dir, "model.pkl")
            mlflow.sklearn.save_model(model, model_path)
            mlflow.sklearn.log_model(model, artifact_path="model")
            logger.info(f"Modèle sauvegardé dans : {model_path}")
        else:
            logger.warning(f"Modèle non sauvegardé (accuracy={val_accuracy:.4f} < min_accuracy={min_accuracy:.4f})")

        # 9) Enregistrer l'ID du run MLflow
        run_id = run.info.run_id
        mlflow_id_path = os.path.join(output_dir, "mlflow_id.json")
        with open(mlflow_id_path, "w") as f:
            json.dump({"run_id": run_id}, f)
        logger.info(f"MLflow run_id = {run_id} sauvegardé dans : {mlflow_id_path}")


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
if __name__ == "__main__":
    params_file = "params.yaml"
    section = "train_logistic_tfidf"

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    train_logistic_regression_incremental(params)
