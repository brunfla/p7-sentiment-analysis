#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import logging
import psutil
import mlflow
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    roc_curve,
    precision_recall_curve
)
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from partitioned_data import PartitionedData

from log_system import log_system_metrics, logger

# Configurer le logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def plot_roc_curve(y_test, y_proba, output_path):
    """Générer et enregistrer la courbe ROC."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def plot_precision_recall_curve(y_test, y_proba, output_path):
    """Générer et enregistrer la courbe Precision-Recall."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def log_graphs_to_mlflow(y_test, y_proba):
    """Génère et enregistre les graphiques ROC et Precision-Recall dans MLflow."""
    roc_path = "roc_curve.png"
    prc_path = "precision_recall_curve.png"
    plot_roc_curve(y_test, y_proba, roc_path)
    plot_precision_recall_curve(y_test, y_proba, prc_path)

    mlflow.log_artifact(roc_path, "plots")
    mlflow.log_artifact(prc_path, "plots")
    logger.info("Graphiques ROC et Precision-Recall logués dans MLflow.")

def handle_mlflow(cfg):
    if cfg.mlflow._target_ == "kubernetes":
        logger.info("MLflow configuré pour Kubernetes.")
        mlflow.set_tracking_uri(cfg.mlflow.trackingUri)
        if hasattr(cfg.mlflow, "username") and hasattr(cfg.mlflow, "password"):
            os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow.username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow.password
            logger.info("Ajout des informations d'authentification.")
    elif cfg.mlflow._target_ == "local":
        logger.info("Gestion de MLflow local.")
        mlflow.set_tracking_uri(cfg.mlflow.trackingUri)
    else:
        raise ValueError(f"Configuration MLflow inconnue: {cfg.mlflow._target_}")

def main():
    # 1) Afficher les ressources système au démarrage
    log_system_metrics()

    # 2) Charger la config Hydra
    config_path = os.getenv("HYDRA_CONFIG_PATH", "./config")
    strategy = os.getenv("HYDRA_STRATEGY", "baseline")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # 3) Initialiser MLflow
    handle_mlflow(cfg)
    mlflow.set_experiment(cfg.mlflow.experiment.name)

    # 4) Charger le run_id depuis le fichier JSON
    run_id_file = "data/output/mlflow_id.json"
    if not os.path.exists(run_id_file):
        logger.error(f"Fichier {run_id_file} introuvable.")
        sys.exit(1)

    with open(run_id_file, "r") as f:
        run_data = json.load(f)

    mlflow_run_id = run_data.get("id")
    if not mlflow_run_id:
        logger.error("Aucune clé 'id' dans le fichier JSON.")
        sys.exit(1)

    logger.info(f"MLflow run_id détecté : {mlflow_run_id}")
    model_uri = f"runs:/{mlflow_run_id}/model"
    logger.info(f"Chargement du modèle depuis {model_uri}")

    # 5) Charger le modèle MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # 6) Charger les données partitionnées
    dataset_path = cfg.training.input
    logger.info(f"Chargement des données depuis {dataset_path}...")
    partitioned_data = PartitionedData.load(dataset_path)

    logger.info(f"Données chargées avec succès. Type de partition : {partitioned_data.partition_type}")
    data = partitioned_data.data
    X_test = data.get("X_test")
    y_test = data.get("y_test")

    if X_test is None or y_test is None:
        logger.error("Les données de test sont manquantes dans le fichier partitionné.")
        sys.exit(1)

    logger.info(f"Test set : X_test.shape={X_test.shape}, y_test.shape={len(y_test)}")

    # 8) Prédictions
    if hasattr(model, "predict_proba"):
        logger.info("Utilisation de predict_proba pour obtenir les probabilités.")
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= cfg.test.threshold).astype(int)
    else:
        logger.warning("Le modèle ne supporte pas predict_proba => utilisation de predict().")
        y_score = model.predict(X_test).flatten()
        y_pred = (y_score >= cfg.test.threshold).astype(int)

    y_pred = np.array(y_pred).flatten()
    logger.info(f"Classes dans y_pred : {set(y_pred)}")

    # Vérification des classes jamais prédites
    missing_classes = set(np.unique(y_test)) - set(y_pred)
    if missing_classes:
        logger.warning(f"Classes jamais prédites : {missing_classes}")
        with mlflow.start_run(run_id=mlflow_run_id):
            mlflow.log_text(f"Classes jamais prédites : {missing_classes}", artifact_file="missing_classes.txt")

    # 9) Calcul des métriques
    computed_metrics = {}
    metrics_to_compute = cfg.test.metrics
    if "accuracy" in metrics_to_compute:
        computed_metrics["test_accuracy"] = accuracy_score(y_test, y_pred)
    if "precision" in metrics_to_compute:
        computed_metrics["test_precision"] = precision_score(y_test, y_pred, average=cfg.test.averageMethod)
    if "recall" in metrics_to_compute:
        computed_metrics["test_recall"] = recall_score(y_test, y_pred, average=cfg.test.averageMethod)
    if "f1" in metrics_to_compute:
        computed_metrics["test_f1"] = f1_score(y_test, y_pred, average=cfg.test.averageMethod)
    if "roc_auc" in metrics_to_compute and hasattr(model, "predict_proba"):
        computed_metrics["test_roc_auc"] = roc_auc_score(y_test, y_proba)
    if "pr_auc" in metrics_to_compute and hasattr(model, "predict_proba"):
        computed_metrics["test_pr_auc"] = average_precision_score(y_test, y_proba)

    # Log classification report
    cls_report = classification_report(y_test, y_pred)
    logger.info("\n" + cls_report)

    # 10) Logger tout dans MLflow
    with mlflow.start_run(run_id=mlflow_run_id):
        for k, v in computed_metrics.items():
            logger.info(f"{k}: {v:.4f}")
            mlflow.log_metric(k, v)
        mlflow.log_text(cls_report, artifact_file="final_classification_report.txt")
        logger.info("Métriques finales loguées dans MLflow.")

        # Ajouter les graphiques ROC et Precision-Recall si disponibles
        if hasattr(model, "predict_proba"):
            log_graphs_to_mlflow(y_test, y_proba)

    # 11) Fin : log ressources
    log_system_metrics()

if __name__ == "__main__":
    main()
