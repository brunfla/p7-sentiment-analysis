import os
import json
import logging
import mlflow
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve
)
from params_utils import load_params
from logging_utils import get_logger
import matplotlib.pyplot as plt

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
    Charger le modèle TensorFlow depuis MLflow via un run_id.
    """
    logger.info(f"Chargement du modèle depuis MLflow avec run_id : {run_id}")
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/model"
    try:
        model = mlflow.keras.load_model(model_uri)
        logger.info(f"Modèle chargé avec succès depuis : {model_uri}")
        return model
    except mlflow.exceptions.MlflowException as e:
        logger.error(f"Erreur MLflow lors du chargement du modèle : {e}")
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue lors du chargement du modèle : {e}")
        raise

def compute_metrics(y_true, y_pred, y_scores, metrics_cfg):
    """
    Calculer les métriques demandées.
    """
    metrics = {}
    if "accuracy" in metrics_cfg:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
    if "precision" in metrics_cfg:
        metrics["precision"] = precision_score(y_true, y_pred)
    if "recall" in metrics_cfg:
        metrics["recall"] = recall_score(y_true, y_pred)
    if "f1" in metrics_cfg:
        metrics["f1"] = f1_score(y_true, y_pred)
    if "roc_auc" in metrics_cfg and y_scores is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_scores)
    if "pr_auc" in metrics_cfg and y_scores is not None:
        metrics["pr_auc"] = average_precision_score(y_true, y_scores)
    return metrics

def save_metrics(output_file, metrics):
    """
    Sauvegarder les métriques dans un fichier JSON.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Métriques sauvegardées dans : {output_file}")

def plot_curves(y_true, y_scores, output_dir):
    """
    Générer et sauvegarder les courbes ROC et PR.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_true, y_scores):.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path)
    logger.info(f"ROC curve saved to: {roc_path}")
    plt.close()

    # Courbe PR
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {average_precision_score(y_true, y_scores):.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    pr_path = os.path.join(output_dir, "pr_curve.png")
    plt.savefig(pr_path)
    logger.info(f"Precision-Recall curve saved to: {pr_path}")
    plt.close()

def evaluate_model(params):
    """
    Évaluer le modèle et générer les résultats.
    """
    # Charger les données de test
    logger.info(f"Chargement des données de test depuis : {params['input_file']}")
    with open(params["input_file"], "rb") as f:
        test_data = pickle.load(f)
    X_test, y_test = test_data["X"], test_data["y"]

    # Construire les chemins dynamiques
    mlflow_id_file = os.path.join(params["output_dir"], "model", "mlflow_id.json")
    metrics_file = os.path.join(params["output_dir"], "metrics", "test_metrics.json")
    plots_dir = os.path.join(params["output_dir"], "plots")

    # Charger le modèle
    model_run_id = load_model_run_id(mlflow_id_file)
    model = load_model(model_run_id, params["mlflow"]["trackingUri"])

    # Effectuer les prédictions
    y_scores = model.predict(X_test).flatten()
    y_pred = (y_scores >= params["threshold"]).astype(int)

    # Calcul des métriques
    metrics = compute_metrics(y_test, y_pred, y_scores, params["metrics"])

    # Sauvegarder les métriques
    save_metrics(metrics_file, metrics)

    # Générer des plots
    plot_curves(y_test, y_scores, plots_dir)

    # Enregistrer dans MLflow
    with mlflow.start_run(run_id=model_run_id):
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_artifact(metrics_file, artifact_path="metrics")
        mlflow.log_artifact(os.path.join(plots_dir, "roc_curve.png"), artifact_path="plots")
        mlflow.log_artifact(os.path.join(plots_dir, "pr_curve.png"), artifact_path="plots")
    logger.info("Métriques et plots enregistrés dans MLflow.")



def main():
    params_file = "params.yaml"
    params = load_params(params_file, "test_lstm_bidirectional_with_glove")
    evaluate_model(params)

if __name__ == "__main__":
    main()
