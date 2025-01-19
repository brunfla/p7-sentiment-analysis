import os
import argparse
import json
import logging
import pickle
import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from params_utils import load_params
from logging_utils import get_logger
import matplotlib.pyplot as plt
import sys

logger = get_logger(__name__)

def build_model(input_dim, embedding_matrix, params):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=input_dim,
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=params["max_length"],
            trainable=False
        )
    )
    for i, units in enumerate(params["lstm_units"]):
        return_sequences = i < len(params["lstm_units"]) - 1  # True sauf pour la dernière couche LSTM
        model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
        model.add(Dropout(params["dropout_rate"]))
    model.add(Dense(params["dense_units"], activation="relu"))
    model.add(Dropout(params["dropout_rate"]))
    model.add(Dense(1, activation="sigmoid"))  # Sortie scalaire pour la classification binaire

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_training_history(history, output_dir):
    """
    Générer des plots pour l'évolution des métriques d'entraînement et de validation.

    Args:
        history: Historique d'entraînement Keras.
        output_dir: Répertoire où sauvegarder les plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot de l'accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Train vs Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    accuracy_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(accuracy_plot_path)
    logger.info(f"Plot d'accuracy sauvegardé dans : {accuracy_plot_path}")
    plt.close()

    # Plot de la perte
    plt.figure(figsize=(10, 6))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Plot de perte sauvegardé dans : {loss_plot_path}")
    plt.close()

def log_plots_to_mlflow(output_dir):
    """
    Logger les plots dans MLflow en tant qu'artefacts.

    Args:
        output_dir: Répertoire où les plots sont sauvegardés.
    """
    accuracy_plot_path = os.path.join(output_dir, "accuracy_plot.png")
    loss_plot_path = os.path.join(output_dir, "loss_plot.png")

    if os.path.exists(accuracy_plot_path):
        mlflow.log_artifact(accuracy_plot_path, artifact_path="plots")
        logger.info(f"Plot de précision loggé dans MLflow : {accuracy_plot_path}")

    if os.path.exists(loss_plot_path):
        mlflow.log_artifact(loss_plot_path, artifact_path="plots")
        logger.info(f"Plot de perte loggé dans MLflow : {loss_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Script d'entraînement ou d'enregistrement MLflow.")
    parser.add_argument(
        "--only-log-to-mlflow",
        action="store_true",
        help="Relancer uniquement l'enregistrement des artefacts dans MLflow, sans entraînement."
    )
    args = parser.parse_args()

    params_file = "params.yaml"
    section = "train_lstm_bidirectional_with_glove"

    # Charger les paramètres
    params = load_params(params_file, section)

    # Répertoires pour les artefacts
    model_dir = os.path.join(params["output_dir"], "model")
    metrics_dir = os.path.join(params["output_dir"], "metrics")
    plots_dir = os.path.join(params["output_dir"], "plots")

    if args.only_log_to_mlflow:
        # Relancer uniquement l'enregistrement des artefacts dans MLflow
        logger.info("Relance uniquement pour l'enregistrement des artefacts MLflow.")
        
        # Vérification des artefacts nécessaires
        model_path = os.path.join(model_dir, "model.h5")
        accuracy_plot_path = os.path.join(plots_dir, "accuracy_plot.png")
        loss_plot_path = os.path.join(plots_dir, "loss_plot.png")
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        
        missing_files = []
        for file_path in [model_path, accuracy_plot_path, loss_plot_path, metrics_path]:
            if not os.path.exists(file_path):
                missing_files.append(file_path)

        if missing_files:
            logger.error(f"Les fichiers suivants sont manquants : {', '.join(missing_files)}")
            sys.exit(1)

        # Charger le modèle sauvegardé
        model = tf.keras.models.load_model(model_path)
        
        # Configurer MLflow
        mlflow.set_tracking_uri(params["mlflow"]["trackingUri"])
        mlflow.set_experiment(params["mlflow"]["experiment"]["name"])
        
        with mlflow.start_run(run_name=params["mlflow"]["experiment"]["run"]["name"]) as run:
            run_id = run.info.run_id  # Obtenir le run_id
            logger.info(f"Run ID actif : {run_id}")

            # Sauvegarder le run_id dans un fichier json
            mlflow_id_path = os.path.join(model_dir, "mlflow_id.json")
            with open(mlflow_id_path, "w") as f:
                json.dump({"run_id": run_id}, f)
            logger.info(f"Run ID enregistré dans : {mlflow_id_path}")

            # Loguer les artefacts dans MLflow
            mlflow.keras.log_model(model, artifact_path="model")
            mlflow.log_artifact(accuracy_plot_path, artifact_path="plots")
            mlflow.log_artifact(loss_plot_path, artifact_path="plots")
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
            logger.info("Modèle et artefacts enregistrés dans MLflow.")

    else:
        # Processus complet incluant l'entraînement
        # Création des répertoires si nécessaire
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Charger les données GloVe
        logger.info(f"Chargement des vecteurs GloVe depuis : {params['glove_vectors_file']}")
        with open(params["glove_vectors_file"], "rb") as f:
            glove_data = pickle.load(f)
        embedding_matrix = glove_data["embedding_matrix"]

        # Charger les données d'entraînement et de validation
        logger.info(f"Chargement des données d'entraînement depuis : {params['input_train_file']}")
        with open(params["input_train_file"], "rb") as f:
            train_data = pickle.load(f)
        logger.info(f"Chargement des données de validation depuis : {params['input_val_file']}")
        with open(params["input_val_file"], "rb") as f:
            val_data = pickle.load(f)

        # Construire le modèle
        logger.info("Construction du modèle LSTM bidirectionnel...")
        input_dim = embedding_matrix.shape[0]
        model = build_model(input_dim, embedding_matrix, params["model_params"])

        # Créer des datasets TensorFlow
        train_dataset = tf.data.Dataset.from_tensor_slices((train_data["X"], train_data["y"])).batch(params["model_params"]["batch_size"])
        val_dataset = tf.data.Dataset.from_tensor_slices((val_data["X"], val_data["y"])).batch(params["model_params"]["batch_size"])

        # Entraîner le modèle
        logger.info("Démarrage de l'entraînement...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=params["model_params"]["epochs"],
            verbose=1
        )

        # Sauvegarder le modèle
        model_path = os.path.join(model_dir, "model.h5")
        model.save(model_path)
        logger.info(f"Modèle sauvegardé dans : {model_path}")

        # Générer des plots
        logger.info("Génération des plots d'entraînement...")
        plot_training_history(history, plots_dir)

        # Sauvegarder les métriques dans un fichier JSON
        metrics = {
            "val_accuracy": max(history.history["val_accuracy"]),
            "val_loss": min(history.history["val_loss"])
        }
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        logger.info(f"Métriques sauvegardées dans : {metrics_path}")

        # Enregistrer l'expérience MLflow
        mlflow.set_tracking_uri(params["mlflow"]["trackingUri"])
        mlflow.set_experiment(params["mlflow"]["experiment"]["name"])
        with mlflow.start_run(run_name=params["mlflow"]["experiment"]["run"]["name"]) as run:
            run_id = run.info.run_id  # Obtenir le run_id
            logger.info(f"Run ID actif : {run_id}")

            # Sauvegarder le run_id dans un fichier json
            mlflow_id_path = os.path.join(model_dir, "mlflow_id.json")
            with open(mlflow_id_path, "w") as f:
                json.dump({"run_id": run_id}, f)
            logger.info(f"Run ID enregistré dans : {mlflow_id_path}")

            mlflow.log_param("embedding_dim", embedding_matrix.shape[1])
            mlflow.log_metrics(metrics)
            mlflow.keras.log_model(model, artifact_path="model")
            mlflow.log_artifact(metrics_path, artifact_path="metrics")
            log_plots_to_mlflow(plots_dir)
            logger.info("Modèle et artefacts enregistrés dans MLflow.")

if __name__ == "__main__":
    main()
