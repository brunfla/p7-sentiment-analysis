import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TF_TRT_DISABLE"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Supprime les logs d'information
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import sys
import json
import mlflow
import ktrain
from ktrain import text
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def get_logger(name):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

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
    return run_id

def to_scalar(value):
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        return np.mean(value)
    if hasattr(value, "item"):
        return value.item()
    return value

def load_params(params_file, section):
    import yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

def plot_confusion_matrix(conf_matrix, class_names, output_dir):
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

def evaluate_single_tweet(predictor, tweet):
    """
    Effectue une prédiction pour un tweet unique.
    Returns 0 pour négatif, 1 pour positif.
    """
    prediction = predictor.predict([tweet])[0]
    return 1 if prediction == "label" else 0

def test_model_single_tweet(params):
    """
    Teste le modèle sur des tweets un par un, simulant un cas de production.
    """
    input_dir = params["input_dir"]
    test_data_path = os.path.join(input_dir, params["input_files"][0])  # x_test.csv
    test_labels_path = os.path.join(input_dir, params["input_files"][1])  # y_test.csv

    if not os.path.exists(test_data_path) or not os.path.exists(test_labels_path):
        logger.error("Fichiers de test introuvables.")
        raise FileNotFoundError("Fichiers de test introuvables.")

    logger.info(f"Chargement des tweets de test depuis : {test_data_path}")
    x_test_df = pd.read_csv(test_data_path)
    
    logger.info(f"Chargement des labels de test depuis : {test_labels_path}")
    y_test_df = pd.read_csv(test_labels_path)

    # Associer les features et labels par ID
    x_test = x_test_df.set_index("id").loc[y_test_df["id"]]["feature"].values
    y_test = y_test_df.set_index("id")["label"].values

    # Récupérer le run_id depuis le fichier JSON
    run_id = load_model_run_id(params["model_run_id_file"])
    logger.info(f"Chargement du modèle depuis MLflow run_id: {run_id}")

    # Configuration de MLflow
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])

    # Téléchargement des artefacts du modèle depuis MLflow
    model_artifacts_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="final_model"
    )
    logger.info(f"Artefacts téléchargés depuis MLflow : {model_artifacts_path}")

    # Chargement du modèle ktrain
    predictor = ktrain.load_predictor(model_artifacts_path)

    # On ouvre un contexte MLflow pour tout logger
    with mlflow.start_run(run_id=run_id):
        predictions = []
        for tweet in x_test:
            prediction = evaluate_single_tweet(predictor, tweet)
            print(f"{tweet} => {prediction}")
            predictions.append(prediction)

        # Calcul des métriques
        accuracy = accuracy_score(y_test, predictions)
        logger.info(f"Accuracy sur les données de test : {accuracy:.4f}")

        # Classification report
        report_dict = classification_report(
            y_test, 
            predictions, 
            target_names=["Negative_0", "Positive_1"],
            output_dict=True
        )
        logger.info(f"\n{report_dict}")

        # Log de l’accuracy (préfixé par test_)
        mlflow.log_metric("test_accuracy", accuracy)

        # Log des autres métriques (precision, recall, f1, etc.)
        for label, values in report_dict.items():
            # labels sont: "Negative_0", "Positive_1", "accuracy", "macro avg", "weighted avg"
            if isinstance(values, dict):  
                # ex: {'precision': 0.81, 'recall': 0.82, 'f1-score': 0.81, 'support': 1000}
                for metric_name, metric_value in values.items():
                    if isinstance(metric_value, (float, int)):
                        mlflow.log_metric(f"test_{label}_{metric_name}", metric_value)
            else:
                # ex: 'accuracy' renvoie un float directement
                if isinstance(values, (float, int)):
                    mlflow.log_metric("test_accuracy_overall", values)

        # Matrice de confusion + log en artifact
        conf_matrix = confusion_matrix(y_test, predictions)
        output_dir = params["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(conf_matrix, ["Negative_0", "Positive_1"], output_dir)

        # Sauvegarde des prédictions en JSON + log en artifact
        output_path = os.path.join(output_dir, "predictions.json")
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=4)
        logger.info(f"Prédictions sauvegardées dans : {output_path}")
        mlflow.log_artifact(output_path)

def main():
    params_file = "params.yaml"
    test_params = load_params(params_file, "test_model_single_tweet")
    test_model_single_tweet(test_params)

if __name__ == "__main__":
    main()
