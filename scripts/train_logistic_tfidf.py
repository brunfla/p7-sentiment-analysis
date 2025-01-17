import os
import mlflow
import mlflow.sklearn
import logging
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from scipy.sparse import load_npz
from params_utils import load_params
from logging_utils import get_logger
import pandas as pd
import json
import sys

# Obtenir le logger
logger = get_logger(__name__)

def test_initial_setup(input_file, target_file, test_size=10):
    """
    Test initial pour valider le chargement des données et les opérations basiques.

    Args:
        input_file (str): Chemin vers les données vectorisées (matrice sparse).
        target_file (str): Chemin vers les labels.
        test_size (int): Nombre de lignes à tester pour validation.

    Returns:
        bool: True si le test réussit, sinon lève une exception.
    """
    logger.info("Démarrage du test initial...")
    X_sparse = load_npz(input_file)
    y = pd.read_csv(target_file, nrows=test_size)

    if X_sparse.shape[0] < test_size or len(y) < test_size:
        raise ValueError("Les données contiennent moins de lignes que le test_size spécifié.")

    X_test = X_sparse[:test_size]
    y_test = y.squeeze()

    logger.info(f"Données de test chargées avec {test_size} lignes.")

    # Vérification d'entraînement rapide
    model = SGDClassifier(loss="log_loss", max_iter=10, tol=1e-3)
    model.fit(X_test, y_test)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    logger.info(f"Test initial réussi avec une accuracy de {accuracy:.4f}.")

    return True

def is_model_logged(run_id):
    """
    Vérifie si un modèle est déjà enregistré pour une run dans MLflow.

    Args:
        run_id (str): Identifiant de la run MLflow.

    Returns:
        bool: True si un modèle est enregistré, False sinon.
    """
    model_uri = f"runs:/{run_id}/model"
    try:
        mlflow.sklearn.load_model(model_uri)
        logger.info(f"Un modèle existe déjà pour la run_id : {run_id}")
        return True
    except Exception:
        logger.info(f"Aucun modèle trouvé pour la run_id : {run_id}")
        return False

def train_logistic_regression_incremental(input_file, target_file, output_dir, model_type, tuning_cfg, mlflow_cfg, chunksize=10000):
    """
    Entraîne un modèle de régression logistique incrémentalement avec des données déjà vectorisées.

    Args:
        input_file (str): Chemin vers les données vectorisées (matrice sparse).
        target_file (str): Chemin vers les labels.
        output_dir (str): Répertoire pour sauvegarder le modèle.
        model_type (str): Type de modèle, ici "logistic_regression".
        tuning_cfg (dict): Configuration pour le modèle (hyperparamètres).
        mlflow_cfg (dict): Configuration pour le suivi avec MLflow.
        chunksize (int): Nombre de lignes chargées en mémoire par itération.
    """
    # Configuration initiale
    logger.info(f"Chargement incrémental des données vectorisées depuis : {input_file}")
    mlflow.set_tracking_uri(mlflow_cfg["trackingUri"])
    mlflow.set_experiment(mlflow_cfg["experiment"]["name"])

    model = SGDClassifier(loss="log_loss", max_iter=tuning_cfg.get("max_iter", 1000), tol=tuning_cfg.get("tol", 1e-3))

    total_samples = 0
    all_predictions = []
    all_labels = []

    with mlflow.start_run(run_name=mlflow_cfg["experiment"]["run"]["name"], description=mlflow_cfg["experiment"]["run"]["description"]) as run:
        mlflow.set_tags(mlflow_cfg["experiment"]["run"].get("tags", {}))

        # Lecture incrémentale des données vectorisées
        X_sparse = load_npz(input_file)  # Charger la matrice sparse
        y = pd.read_csv(target_file, chunksize=chunksize)  # Charger les labels par lot

        for i, y_chunk in enumerate(y):
            start_idx = i * chunksize
            end_idx = min(start_idx + chunksize, X_sparse.shape[0])

            X_chunk = X_sparse[start_idx:end_idx]
            y_chunk = y_chunk.squeeze()

            if total_samples == 0:
                model.partial_fit(X_chunk, y_chunk, classes=pd.unique(y_chunk))
            else:
                model.partial_fit(X_chunk, y_chunk)

            # Évaluation sur le batch actuel
            predictions = model.predict(X_chunk)
            accuracy = accuracy_score(y_chunk, predictions)
            logger.info(f"Accuracy sur le batch actuel : {accuracy:.4f}")

            all_predictions.extend(predictions)
            all_labels.extend(y_chunk)

            total_samples += len(y_chunk)

        # Évaluation finale
        overall_accuracy = accuracy_score(all_labels, all_predictions)
        mlflow.log_metric("overall_accuracy", overall_accuracy)
        logger.info(f"Accuracy globale : {overall_accuracy:.4f}")

        # Sauvegarde du modèle dans un fichier local
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "model.pkl")
        mlflow.sklearn.save_model(model, model_path)

        # Vérification et enregistrement conditionnel dans MLflow
        run_id = run.info.run_id
        model_already_logged = is_model_logged(run_id)
        threshold_accuracy = mlflow_cfg["experiment"]["run"].get("min_accuracy", 0.7)

        if not model_already_logged:
            logger.info("Aucun modèle existant. Enregistrement du modèle dans MLflow.")
            mlflow.sklearn.log_model(model, artifact_path="model")
        elif overall_accuracy >= threshold_accuracy:
            logger.info(f"Enregistrement du modèle car accuracy {overall_accuracy:.4f} >= seuil {threshold_accuracy:.4f}.")
            mlflow.sklearn.log_model(model, artifact_path="model")
        else:
            logger.warning(f"Modèle non enregistré car accuracy {overall_accuracy:.4f} < seuil {threshold_accuracy:.4f}.")

        # Enregistrer l'ID de la run MLflow
        mlflow_id_path = os.path.join(output_dir, "mlflow_id.json")
        with open(mlflow_id_path, "w") as f:
            json.dump({"run_id": run_id}, f)
        logger.info(f"ID de la run MLflow sauvegardé dans : {mlflow_id_path}")

    return model

if __name__ == "__main__":
    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        input_file = params["input_file"]
        target_file = params["target_file"]
        output_dir = params["output_dir"]
        model_type = params["model_type"]
        tuning_cfg = params.get("tuning", {})
        mlflow_cfg = params.get("mlflow", {})
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Étape de validation initiale
    try:
        test_initial_setup(input_file, target_file, test_size=10)
    except Exception as e:
        logger.error(f"Erreur lors du test initial : {e}")
        sys.exit(1)

    # Entraînement incrémental
    train_logistic_regression_incremental(input_file, target_file, output_dir, model_type, tuning_cfg, mlflow_cfg)
