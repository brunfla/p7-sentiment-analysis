import os
import json
import logging
import mlflow
import pandas as pd
import numpy as np
import sys
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------------------------
# Chargement des params, logger
# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# Charger run_id et modèle MLflow
# --------------------------------------------------------------------
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


# --------------------------------------------------------------------
# Plot de la matrice de confusion
# --------------------------------------------------------------------
def plot_confusion_matrix_custom(conf_matrix, class_names, output_dir):
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
    plt.savefig(confusion_matrix_path)
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée dans : {confusion_matrix_path}")

    # Log artifact dans le run MLflow actif
    mlflow.log_artifact(confusion_matrix_path)


# --------------------------------------------------------------------
# Plot des metrics de base
# --------------------------------------------------------------------
def plot_basic_metrics(metrics, output_dir):
    """
    Générer des plots simples pour accuracy, precision, recall, f1
    et les sauvegarder.
    """
    os.makedirs(output_dir, exist_ok=True)

    # On choisit un sous-ensemble de metrics à tracer
    basic_keys = ["test_accuracy", "test_Negative_0_precision", "test_Negative_0_recall", "test_Negative_0_f1-score",
                  "test_Positive_1_precision", "test_Positive_1_recall", "test_Positive_1_f1-score"]
    plot_data = {k: v for k, v in metrics.items() if k in basic_keys}

    if not plot_data:
        logger.info("Aucun metric de base trouvé pour tracer le barplot.")
        return

    keys = list(plot_data.keys())
    values = [plot_data[k] for k in keys]

    plt.figure(figsize=(8, 6))
    plt.bar(keys, values, color="skyblue")
    plt.title("Performance Metrics (basic)")
    plt.ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.0)

    plot_path = os.path.join(output_dir, "metrics_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path)
    logger.info(f"Plot des métriques sauvegardé dans : {plot_path}")
    plt.close()

# --------------------------------------------------------------------
# Calcul et log des metrics
# --------------------------------------------------------------------
def log_classification_report(y_true, y_pred, output_dir):
    """
    Calculer un classification_report à deux classes
    ['Negative_0','Positive_1'], logger les metrics dans MLflow
    sous forme test_<classe>_<metrique>, tracer confusion_matrix, etc.
    """
    # 1) Générer la matrice de confusion
    conf_matrix = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix_custom(conf_matrix, ["Negative_0","Positive_1"], output_dir)

    # 2) Classification report
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Negative_0", "Positive_1"],
        output_dict=True
    )
    # Exemple: report["Negative_0"] = {"precision":..., "recall":..., "f1-score":..., "support":...}
    #          report["accuracy"] = ...
    #          report["macro avg"] = ...
    #          report["weighted avg"] = ...

    # 3) Logger métriques dans MLflow (avec prefix test_)
    #    => On veut par ex. "test_Negative_0_precision": 0.80
    #    => "test_accuracy": 0.81, etc.
    metrics_dict = {}
    for label_name, metrics_info in report.items():
        if isinstance(metrics_info, dict):
            # label_name peut être "Negative_0", "Positive_1", "macro avg", "weighted avg"
            for metric_name, metric_val in metrics_info.items():
                # On ne logge que si c'est un nombre
                if isinstance(metric_val, (int, float)):
                    mlflow_name = f"test_{label_name}_{metric_name}"
                    mlflow.log_metric(mlflow_name, metric_val)
                    metrics_dict[mlflow_name] = metric_val
        else:
            # ex: "accuracy" => c'est un float direct
            mlflow.log_metric("test_accuracy", metrics_info)
            metrics_dict["test_accuracy"] = metrics_info

    return metrics_dict

# --------------------------------------------------------------------
# Évaluer le modèle
# --------------------------------------------------------------------
def evaluate_model(model, test_data_path, label_csv_path, output_file,
                   plot_dir, threshold):
    """
    Évaluer le modèle sur un fichier .pkl (X_test + ids) et un label CSV (id,label).
    On suppose y_test = label_csv_path (colonnes: "id","label").

    1) Charger X_test, y_test
    2) Faire prédictions
    3) Classification report + confusion matrix
    4) Log metrics dans MLflow
    5) Sauvegarder metrics dans un JSON
    6) Plot basique
    """

    logger.info(f"=== Chargement des données de test TF-IDF depuis : {test_data_path}")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Fichier introuvable : {test_data_path}")
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    X_test = test_data["vectors"]
    test_ids = np.array(test_data["ids"])

    logger.info(f"=== Chargement des labels depuis : {label_csv_path}")
    if not os.path.exists(label_csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {label_csv_path}")
    df_y_test = pd.read_csv(label_csv_path)

    # Synchroniser l'ordre : on suppose un mapping id->label
    label_map_test = dict(zip(df_y_test["id"], df_y_test["label"]))
    y_test = []
    for tid in test_ids:
        y_test.append(label_map_test[tid])
    y_test = np.array(y_test)

    if X_test.shape[0] != len(y_test):
        msg = "Nombre d'échantillons différent entre X_test et y_test."
        logger.error(msg)
        raise ValueError(msg)

    logger.info(f"Test set: {X_test.shape[0]} échantillons, {X_test.shape[1]} features.")

    # 1) Prédictions
    logger.info("Prédictions en cours (test)...")
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        # Pas de probas
        y_pred = model.predict(X_test)

    # 2) Classification report + confusion matrix
    logger.info("Calcul du report de classification et matrice de confusion...")
    metrics_dict = log_classification_report(y_test, y_pred, plot_dir)

    # 3) Sauvegarder les métriques dans un JSON
    logger.info("Sauvegarde des métriques dans un fichier JSON")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(metrics_dict, f, indent=4)

    # 4) Plot de quelques métriques de base (barplot)
    plot_basic_metrics(metrics_dict, plot_dir)

    logger.info("Évaluation terminée.")
    return metrics_dict


# --------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------
def main():
    # 1) Charger les paramètres
    params_file = "params.yaml"
    section = "test_logistic_tfidf"  # ou adapter si vous voulez un autre nom de section

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur lors du chargement des paramètres : {e}")
        sys.exit(1)

    # 2) Charger le run_id
    try:
        model_run_id = load_model_run_id(params["model_run_id_file"])
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Erreur lors du chargement de l'identifiant du modèle : {e}")
        sys.exit(1)

    # 3) Charger le modèle
    try:
        model = load_model(model_run_id, params["mlflow"]["trackingUri"])
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # 4) Évaluer le modèle
    try:
        with mlflow.start_run(run_id=model_run_id):
            evaluate_model(
                model=model,
                test_data_path=params["input_test_vec"],   # e.g. "x_test_vec.pkl"
                label_csv_path=params["input_test_labels"],# e.g. "y_test.csv"
                output_file=params["output_metrics_json"], # e.g. "test_metrics.json"
                plot_dir=params["plot_dir"],               # e.g. "data/plots/test_metrics"
                threshold=params.get("threshold", 0.5)
            )
    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation du modèle : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
