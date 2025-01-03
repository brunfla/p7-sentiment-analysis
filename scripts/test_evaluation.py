#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import argparse
import logging
import pickle

import psutil
import mlflow
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report
)
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-id-file",
        default="mlflow_runid.json",
        help="Chemin du fichier JSON contenant le run_id MLflow généré par train_model.py."
    )
    args = parser.parse_args()

    # 1) Afficher les ressources système au démarrage
    log_system_metrics()

    # 2) Charger la config Hydra
    config_path = os.getenv("HYDRA_CONFIG_PATH", "../notebooks/config")
    strategy = os.getenv("HYDRA_STRATEGY", "validation-quick")

    # Réinitialiser Hydra si déjà initialisé
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # 3) Vérifier si on est en mode "validation-quick" => pas d'évaluation finale
    if strategy == "validation-quick":
        logger.warning("Mode 'validation-quick' détecté => pas d'évaluation finale.")
        sys.exit(0)

    # 4) Vérifier si un bloc evaluator existe et s'il est "none"
    if hasattr(cfg, "evaluator"):
        eval_cfg = cfg.evaluator
        if getattr(eval_cfg, "_target_", None) == "none":
            logger.warning("Bloc evaluator défini sur 'none' => pas d'évaluation.")
            sys.exit(0)
    else:
        logger.warning("Aucun bloc 'evaluator' dans la config => pas d'évaluation.")
        sys.exit(0)

    # 5) Charger le run_id depuis le fichier JSON
    if not os.path.exists(args.run_id_file):
        logger.error(f"Fichier {args.run_id_file} introuvable => impossible de charger mlflow_run_id.")
        sys.exit(1)

    with open(args.run_id_file, "r") as f:
        run_data = json.load(f)
    mlflow_run_id = run_data.get("mlflow_run_id", None)
    if not mlflow_run_id:
        logger.error("Aucune clé 'mlflow_run_id' dans ce JSON => évaluation impossible.")
        sys.exit(1)

    logger.info(f"MLflow run_id détecté : {mlflow_run_id}")
    model_uri = f"runs:/{mlflow_run_id}/model"
    logger.info(f"Chargement du modèle depuis {model_uri}")

    # 6) Charger le modèle MLflow
    model = mlflow.sklearn.load_model(model_uri)

    # 7) Charger le split de partition pour récupérer (X_test, y_test)
    partition_cfg = cfg.partitioner
    split_path = getattr(partition_cfg, "outputSplit", None)
    if not split_path or not os.path.exists(split_path):
        logger.error(f"Impossible de charger le fichier de split partition : {split_path}")
        sys.exit(1)

    logger.info(f"Chargement du split depuis {split_path}")
    with open(split_path, "rb") as f:
        data = pickle.load(f)

    # Selon le type de partition, on récupère le test set
    if partition_cfg._target_ == "trainValTest":
        # data = (X_train, y_train, X_val, y_val, X_test, y_test)
        X_train, y_train, X_val, y_val, X_test, y_test = data
    elif partition_cfg._target_ == "trainTest":
        # data = (X_train, y_train, X_test, y_test)
        X_train, y_train, X_test, y_test = data
    elif partition_cfg._target_ == "crossValidation":
        # data = (folds, X, y)
        # => pas de X_test, y_test unique => on peut lever un avertissement ou gérer autrement
        logger.warning("Partition crossValidation détectée => pas de test set unique.")
        sys.exit(0)
    else:
        logger.error(f"Partition non reconnue : {partition_cfg._target_}")
        sys.exit(1)

    logger.info(f"Test set : X_test.shape={X_test.shape}, y_test.shape={len(y_test)}")

    # 8) Récupérer la config d'évaluation (seuil, métriques...)
    threshold = getattr(eval_cfg, "threshold", 0.5)
    metrics_to_compute = getattr(eval_cfg, "metrics", ["accuracy", "precision", "recall", "f1"])
    average_method = getattr(eval_cfg, "averageMethod", "binary")

    logger.info(f"Evaluation => threshold={threshold}, metrics={metrics_to_compute}, average={average_method}")

    # 9) Prédictions
    #    - Si on a un predict_proba, on applique un threshold
    #    - Sinon, on fait un predict classique
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        # Application du threshold
        import numpy as np
        y_pred = (y_proba >= threshold).astype(int)
    else:
        logger.warning("Le modèle ne supporte pas predict_proba => seuil ignoré")
        y_pred = model.predict(X_test)

    # 10) Calcul des métriques
    computed_metrics = {}
    if "accuracy" in metrics_to_compute:
        acc = accuracy_score(y_test, y_pred)
        computed_metrics["test_accuracy"] = acc

    if "precision" in metrics_to_compute:
        prec = precision_score(y_test, y_pred, average=average_method)
        computed_metrics["test_precision"] = prec

    if "recall" in metrics_to_compute:
        rec = recall_score(y_test, y_pred, average=average_method)
        computed_metrics["test_recall"] = rec

    if "f1" in metrics_to_compute:
        f1 = f1_score(y_test, y_pred, average=average_method)
        computed_metrics["test_f1"] = f1

    if "roc_auc" in metrics_to_compute and hasattr(model, "predict_proba"):
        # On recalcule sur y_proba
        auc_val = roc_auc_score(y_test, y_proba)
        computed_metrics["test_roc_auc"] = auc_val

    if "pr_auc" in metrics_to_compute and hasattr(model, "predict_proba"):
        ap_val = average_precision_score(y_test, y_proba)
        computed_metrics["test_pr_auc"] = ap_val

    # Log report global
    cls_report = classification_report(y_test, y_pred)
    logger.info("\n" + cls_report)

    # 11) Logger tout dans MLflow (nouveau run "final_evaluation")
    with mlflow.start_run(run_name="final_evaluation"):
        # Logger les métriques
        for k, v in computed_metrics.items():
            logger.info(f"{k}: {v:.4f}")
            mlflow.log_metric(k, v)

        # Logger le classification report
        mlflow.log_text(cls_report, artifact_file="final_classification_report.txt")

        logger.info("Métriques finales loguées dans MLflow.")

    # 12) Fin : log ressources
    log_system_metrics()

if __name__ == "__main__":
    main()
