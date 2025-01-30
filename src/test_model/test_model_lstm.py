import os
import sys
import json
import logging
import pickle
import numpy as np
import pandas as pd
import mlflow

# Pour le chargement du modèle Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# Pour les métriques
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------
# PARAMÈTRES & LOGGER
# ---------------------------------------------------
def load_params(params_file, section):
    import yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# ---------------------------------------------------
# FONCTION : Plot confusion matrix
# ---------------------------------------------------
def plot_confusion_matrix_custom(conf_matrix, class_names, output_dir, mlflow_on=True):
    """
    Génère un graphique de matrice de confusion et l'enregistre dans output_dir.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")

    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(confusion_matrix_path)
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée dans : {confusion_matrix_path}")

    # Log artifact dans MLflow si actif
    if mlflow_on:
        mlflow.log_artifact(confusion_matrix_path)

# ---------------------------------------------------
# SCRIPT PRINCIPAL
# ---------------------------------------------------
def main():
    # 1) Charger la section de params "test_lstm_bidirectional_with_glove"
    params_file = "params.yaml"
    section = "test_lstm_bidirectional_with_glove"
    params = load_params(params_file, section)

    # 2) Extraire les infos
    input_test_vec = params["input_test_vec"]
    input_test_labels = params["input_test_labels"]
    output_dir = params["output_dir"]
    threshold = params.get("threshold", 0.5)

    # MLflow
    mlflow_conf = params.get("mlflow", {})
    run_id = mlflow_conf.get("run_id", None)
    tracking_uri = mlflow_conf.get("trackingUri", None)

    # Liste de metrics demandées
    requested_metrics = params.get("metrics", [])  # e.g. ["accuracy","precision","recall","f1","roc_auc","pr_auc"]

    # 3) Charger le modèle Keras (.h5)
    model_path = os.path.join(output_dir, "bilstm_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modèle introuvable: {model_path}")
    logger.info(f"Chargement du modèle Keras depuis: {model_path}")
    model = load_model(model_path)
    logger.info("Modèle BiLSTM chargé avec succès.")

    # 4) Charger X_test (pkl) et y_test (csv)
    if not os.path.exists(input_test_vec):
        raise FileNotFoundError(f"Fichier introuvable: {input_test_vec}")
    with open(input_test_vec, "rb") as f:
        test_data = pickle.load(f)
    X_test = test_data["vectors"]  # shape (N, max_seq_len, embedding_dim)
    test_ids = test_data["ids"]

    if not os.path.exists(input_test_labels):
        raise FileNotFoundError(f"Fichier introuvable: {input_test_labels}")
    df_y_test = pd.read_csv(input_test_labels)

    # Synchroniser l'ordre X_test & y_test via "id"
    label_map = dict(zip(df_y_test["id"], df_y_test["label"]))  # id -> label
    y_test = []
    for tid in test_ids:
        y_test.append(label_map[tid])
    y_test = np.array(y_test)

    if len(y_test) != X_test.shape[0]:
        raise ValueError("Incohérence du nombre d'exemples entre X_test et y_test.")

    logger.info(f"Test samples: {X_test.shape[0]}, Input shape: {X_test.shape[1:]}")

    # 5) Prédictions (probabilités)
    logger.info("Prédiction du modèle sur X_test...")
    y_proba = model.predict(X_test)  # shape (N,1)
    y_proba = y_proba.flatten()      # shape (N,)

    # Conversion en labels binaires
    y_pred = (y_proba >= threshold).astype(int)

    # 6) Lancement d'un run MLflow (si run_id est défini)
    mlflow_on = (run_id is not None and tracking_uri is not None)
    if mlflow_on:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_id=run_id)

    # 7) Calculer les metrics demandées
    metrics_result = {}

    if "accuracy" in requested_metrics:
        acc = accuracy_score(y_test, y_pred)
        metrics_result["accuracy"] = acc
        if mlflow_on: mlflow.log_metric("test_accuracy", acc)

    if "precision" in requested_metrics:
        prec = precision_score(y_test, y_pred, zero_division=0)
        metrics_result["precision"] = prec
        if mlflow_on: mlflow.log_metric("test_precision", prec)

    if "recall" in requested_metrics:
        rec = recall_score(y_test, y_pred, zero_division=0)
        metrics_result["recall"] = rec
        if mlflow_on: mlflow.log_metric("test_recall", rec)

    if "f1" in requested_metrics:
        f1 = f1_score(y_test, y_pred, zero_division=0)
        metrics_result["f1"] = f1
        if mlflow_on: mlflow.log_metric("test_f1", f1)

    # Pour "roc_auc" et "pr_auc", on a besoin des probas
    if "roc_auc" in requested_metrics:
        try:
            rocAuc = roc_auc_score(y_test, y_proba)
            metrics_result["roc_auc"] = rocAuc
            if mlflow_on: mlflow.log_metric("test_roc_auc", rocAuc)
        except ValueError as e:
            logger.warning(f"Impossible de calculer le roc_auc: {e}")

    if "pr_auc" in requested_metrics:
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
            prAuc = average_precision_score(y_test, y_proba)
            metrics_result["pr_auc"] = prAuc
            if mlflow_on: mlflow.log_metric("test_pr_auc", prAuc)
        except ValueError as e:
            logger.warning(f"Impossible de calculer le pr_auc: {e}")

    # 8) Matrice de confusion + classification report
    logger.info("Calcul de la matrice de confusion...")
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix_custom(conf_mat,
                                 class_names=["0","1"],
                                 output_dir=output_dir,
                                 mlflow_on=mlflow_on)

    # (Optionnel) classification_report
    report = classification_report(y_test, y_pred, target_names=["0","1"], zero_division=0)
    logger.info("\n" + report)
    if mlflow_on:
        mlflow.log_text(report, artifact_file="classification_report.txt")

    # 9) Sauvegarder un fichier JSON récap des metrics
    metrics_json_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_result, f, indent=4)
    logger.info(f"Métriques sauvegardées dans {metrics_json_path}")

    # Fin du run MLflow
    if mlflow_on:
        mlflow.end_run()

    logger.info("=== Fin du test ===")

# ---------------------------------------------------
# LAUNCH
# ---------------------------------------------------
if __name__ == "__main__":
    main()
