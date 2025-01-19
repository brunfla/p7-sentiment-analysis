import os
import json
import logging
import mlflow
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from scipy.sparse import load_npz
from params_utils import load_params
from logging_utils import get_logger

# Configurer le logger
logger = get_logger(__name__)

def load_model_run_id(run_id_file):
    """
    Charger l'identifiant du modèle depuis un fichier JSON.
    """
    if not os.path.exists(run_id_file):
        raise FileNotFoundError(f"Fichier {run_id_file} introuvable.")

    with open(run_id_file, "r") as f:
        run_data = json.load(f)

    run_id = run_data.get("run_id")
    if not run_id:
        raise ValueError(f"Clé 'run_id' manquante dans le fichier {run_id_file}.")

    logger.info(f"Identifiant du modèle chargé : {run_id}")
    return run_id

def load_model(run_id, tracking_uri):
    """
    Charger le modèle MLflow depuis un run_id.
    """
    logger.info(f"Chargement du modèle depuis MLflow avec run_id : {run_id}")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"Modèle chargé avec succès depuis : {model_uri}")
        return model
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Erreur MLflow lors du chargement du modèle : {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue lors du chargement du modèle : {e}")
        raise

def compute_metrics(y_test, y_pred, y_proba, metrics_cfg):
    """
    Calculer les métriques demandées.
    """
    metrics = {}
    if "accuracy" in metrics_cfg:
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
    if "precision" in metrics_cfg:
        metrics["precision"] = precision_score(y_test, y_pred)
    if "recall" in metrics_cfg:
        metrics["recall"] = recall_score(y_test, y_pred)
    if "f1" in metrics_cfg:
        metrics["f1"] = f1_score(y_test, y_pred)
    if "roc_auc" in metrics_cfg and y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    if "pr_auc" in metrics_cfg and y_proba is not None:
        metrics["pr_auc"] = average_precision_score(y_test, y_proba)
    return metrics

def log_metrics_to_mlflow(metrics):
    """
    Logger les métriques dans MLflow.
    """
    logger.info("Enregistrement des métriques dans MLflow...")
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

def save_metrics(output_file, metrics):
    """
    Sauvegarder les métriques dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Métriques sauvegardées dans : {output_file}")

def plot_metrics(metrics, output_dir):
    """
    Générer des plots pour les métriques et les sauvegarder.
    """
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    # Exemple de plot: Accuracy, Precision, Recall et F1-Score
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_values = [metrics[key] for key in metric_keys if key in metrics]

    plt.figure(figsize=(8, 6))
    plt.bar(metric_keys, metric_values, color="skyblue")
    plt.title("Performance Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)

    plot_path = os.path.join(output_dir, "metrics_plot.png")
    plt.savefig(plot_path)
    logger.info(f"Plot des métriques sauvegardé dans : {plot_path}")
    plt.close()

def evaluate_model(model, test_data_path, label_file, output_file, plot_dir, threshold, metrics_cfg):
    """
    Évaluer le modèle sur les données de test.
    """
    logger.info(f"Chargement des données de test depuis : {test_data_path}")
    try:
        X_test = load_npz(test_data_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données de test : {e}")
        raise

    logger.info(f"Chargement des labels depuis : {label_file}")
    try:
        y_test = pd.read_csv(label_file)["id"]
    except Exception as e:
        logger.error(f"Erreur lors du chargement des labels : {e}")
        raise

    # Prédictions
    logger.info("Prédictions en cours...")
    try:
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        y_pred = (y_proba >= threshold).astype(int) if y_proba is not None else model.predict(X_test)
    except Exception as e:
        logger.error(f"Erreur lors des prédictions : {e}")
        raise

    # Calcul des métriques
    logger.info("Calcul des métriques...")
    try:
        metrics = compute_metrics(y_test, y_pred, y_proba, metrics_cfg)
    except Exception as e:
        logger.error(f"Erreur lors du calcul des métriques : {e}")
        raise

    # Logger les métriques dans MLflow
    log_metrics_to_mlflow(metrics)

    # Sauvegarder les métriques
    save_metrics(output_file, metrics)

    # Générer et sauvegarder des plots
    plot_metrics(metrics, plot_dir)

def main():
    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]  # Nom du script comme section

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur lors du chargement des paramètres : {e}")
        sys.exit(1)

    # Charger le run_id
    try:
        model_run_id = load_model_run_id(params["model_run_id_file"])
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Erreur lors du chargement de l'identifiant du modèle : {e}")
        sys.exit(1)

    # Charger le modèle
    try:
        model = load_model(model_run_id, params["mlflow"]["trackingUri"])
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # Évaluer le modèle
    try:
        with mlflow.start_run(run_id=model_run_id):
            evaluate_model(
                model=model,
                test_data_path=params["input_file"],
                label_file=params["label_file"],
                output_file=params["output_file"],
                plot_dir=params["plot_dir"],
                threshold=params.get("threshold", 0.5),
                metrics_cfg=params.get("metrics", ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]),
            )
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
