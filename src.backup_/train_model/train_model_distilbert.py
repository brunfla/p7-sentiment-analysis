import os
import sys
import json
import pickle
import numpy as np
import tensorflow as tf
from transformers import TFAutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import mlflow.models.signature
from mlflow.models.signature import ModelSignature
from mlflow.types import Schema, TensorSpec
# Démarrage de l'entraînement avec MLflow
import mlflow
import mlflow.keras
import logging

# Sauvegarder l'état initial de sys.path
original_sys_path = sys.path.copy()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params_utils.params_utils import load_params
from logging_utils.logging_utils import get_logger
sys.path = original_sys_path

# Spécifiez un répertoire temporaire pour les artefacts MLflow
#os.environ["MLFLOW_ARTIFACTS_TMP"] = "/mnt/c/Users/bruno/Documents/GitHub/p7-sentiment-analysis/tmp"

# Configurer le logger
logger = get_logger(__name__)

def save_mlflow_run_id(run_id, output_dir):
    """
    Sauvegarde le run_id de MLflow dans un fichier JSON.

    Args:
        run_id (str): Identifiant du run MLflow.
        output_dir (str): Répertoire où enregistrer le fichier JSON.
    """
    os.makedirs(output_dir, exist_ok=True)
    mlflow_id_path = os.path.join(output_dir, "mlflow_id.json")
    with open(mlflow_id_path, "w") as f:
        json.dump({"run_id": run_id}, f, indent=4)
    logger.info(f"MLflow run_id sauvegardé dans : {mlflow_id_path}")

def save_history_as_json(history, output_dir, prefix=""):
    """
    Sauvegarde l'historique d'entraînement au format JSON.

    Args:
        history (dict): Historique d'entraînement (history.history).
        output_dir (str): Répertoire de sortie.
        prefix (str): Préfixe optionnel pour le fichier.
    """
    os.makedirs(output_dir, exist_ok=True)
    history_path = os.path.join(output_dir, f"{prefix}training_history.json")

    # Convertir les valeurs en types natifs Python
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}

    with open(history_path, "w") as f:
        json.dump(history_serializable, f, indent=4)
    logger.info(f"Historique d'entraînement sauvegardé dans : {history_path}")

def validate_data(data, name):
    """
    Valide que les données sont homogènes et prêtes pour la conversion en tensor.
    """
    logger.info(f"Validation des données pour : {name}")
    for key in ["input_ids", "attention_mask", "labels"]:
        if key not in data:
            raise ValueError(f"Clé '{key}' manquante dans les données {name}.")
        for idx, item in enumerate(data[key]):
            if key != "labels" and not all(isinstance(x, int) for x in item):
                logger.error(f"Données non entières dans '{key}' (index {idx} dans {name}): {item}")
                raise ValueError(f"Données invalides dans '{key}' (index {idx} dans {name}).")
            if key == "labels" and not isinstance(item, (int, float)):
                logger.error(f"Label non valide dans '{key}' (index {idx} dans {name}): {item}")
                raise ValueError(f"Label non valide dans '{key}' (index {idx} dans {name}).")
            if key == "labels" and item not in [0.0, 1.0]:
                logger.warning(f"Label inattendu détecté dans '{key}' (index {idx} dans {name}): {item}")

def clean_labels(data, name):
    """
    Nettoyer et convertir les labels en type valide (0 ou 1 uniquement).
    """
    logger.info(f"Nettoyage des labels pour : {name}")
    data["labels"] = [int(label) for label in data["labels"] if label in [0.0, 1.0]]

def ensure_max_length(data, max_length):
    """
    S'assurer que les données tokenisées respectent max_length.
    """
    logger.info(f"Application de max_length ({max_length}) aux données.")
    data["input_ids"] = [ids[:max_length] + [0] * (max_length - len(ids)) for ids in data["input_ids"]]
    data["attention_mask"] = [mask[:max_length] + [0] * (max_length - len(mask)) for mask in data["attention_mask"]]

def plot_metrics(history, output_dir, prefix="", log_to_mlflow=False):
    """
    Générer les graphiques des métriques d'entraînement et de validation et les sauvegarder dans MLflow.
    
    Args:
        history (dict): Historique d'entraînement (history.history).
        output_dir (str): Répertoire de sortie.
        prefix (str): Préfixe optionnel pour les fichiers.
        log_to_mlflow (bool): Si True, les graphiques seront aussi enregistrés dans MLflow.
    """
    # Plot des pertes
    loss_plot_path = os.path.join(output_dir, f'{prefix}loss_plot.png')
    plt.figure(figsize=(10, 6))
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.legend()
    plt.savefig(loss_plot_path)
    plt.close()
    if log_to_mlflow:
        mlflow.log_artifact(loss_plot_path)

    # Plot des précisions
    accuracy_plot_path = os.path.join(output_dir, f'{prefix}accuracy_plot.png')
    plt.figure(figsize=(10, 6))
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Epoch')
    plt.legend()
    plt.savefig(accuracy_plot_path)
    plt.close()
    if log_to_mlflow:
        mlflow.log_artifact(accuracy_plot_path)

def load_pkl_data(file_path):
    """
    Charger des données tokenisées depuis un fichier pickle.
    """
    logger.info(f"Chargement des données depuis : {file_path}")
    with open(file_path, "rb") as f:
        return pickle.load(f)

def train_distilbert_model(params, quick_test=False):
    """
    Entraîner un modèle basé sur DistilBERT avec des données tokenisées.
    """
    prefix = "dryrun_" if quick_test else ""

    # Charger les données tokenisées depuis les fichiers .pkl
    train_data = load_pkl_data(params["input_train_token"])
    val_data = load_pkl_data(params["input_val_token"])

    # Nettoyage des labels pour s'assurer qu'ils sont valides (0.0 ou 1.0 uniquement)
    #train_data["labels"] = [float(label) for label in train_data["labels"] if label in [0.0, 1.0]]
    #val_data["labels"] = [float(label) for label in val_data["labels"] if label in [0.0, 1.0]]

    # Réduire la taille des données pour un test rapide
    if quick_test:
        logger.info("Mode test rapide activé : utilisation de 15 échantillons pour l'entraînement et la validation.")
        train_data["input_ids"] = train_data["input_ids"][:15]
        train_data["attention_mask"] = train_data["attention_mask"][:15]
        train_data["labels"] = train_data["labels"][:15]
        val_data["input_ids"] = val_data["input_ids"][:15]
        val_data["attention_mask"] = val_data["attention_mask"][:15]
        val_data["labels"] = val_data["labels"][:15]

    # S'assurer que les données respectent max_length
    max_length = params["model_params"]["max_length"]
    ensure_max_length(train_data, max_length)
    ensure_max_length(val_data, max_length)

    # Initialisation de DistilBERT
    logger.info(f"Chargement du modèle DistilBERT : {params['pretrained_model']}")
    distilbert = TFAutoModel.from_pretrained(params["pretrained_model"], force_download=False)

    input_ids = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

    bert_output = distilbert(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
    cls_output = tf.reduce_mean(bert_output, axis=1)

    dense1 = tf.keras.layers.Dense(128, activation="relu")(cls_output)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout1)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["training_params"]["learning_rate"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Configurer les répertoires
    output_dir = params["output_dir"]
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    metric_dir = os.path.join(output_dir, "metric")
    os.makedirs(metric_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoints_dir, "model_epoch_{epoch:02d}.h5"),
        save_weights_only=True,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min", verbose=1, restore_best_weights=True
    )

    mlflow.set_tracking_uri(params["mlflow"]["trackingUri"])
    logging.getLogger("mlflow.utils.environment").setLevel(logging.DEBUG)
    mlflow.set_experiment(params["mlflow"]["experiment"]["name"])

    with mlflow.start_run(run_name=params["mlflow"]["experiment"]["run"]["name"]) as run:
        run_id = run.info.run_id  # Obtenir le run_id actuel
        # Enregistrer le run_id dans un fichier JSON
        save_mlflow_run_id(run_id, model_dir)
        logger.info(f"Run enregistré avec run_id : {run_id}")

        mlflow.log_params({
            "learning_rate": params["training_params"]["learning_rate"],
            "batch_size": params["model_params"]["batch_size"],
            "max_length": max_length,
            "epochs": params["model_params"]["epochs"],
        })

        logger.info("Démarrage de l'entraînement...")

        history = model.fit(
            {
                "input_ids": tf.convert_to_tensor(train_data["input_ids"]),
                "attention_mask": tf.convert_to_tensor(train_data["attention_mask"])
            },
            tf.convert_to_tensor(train_data["labels"]),
            validation_data=(
                {
                    "input_ids": tf.convert_to_tensor(val_data["input_ids"]),
                    "attention_mask": tf.convert_to_tensor(val_data["attention_mask"])
                },
                tf.convert_to_tensor(val_data["labels"])
            ),
            epochs=params["model_params"]["epochs"],
            batch_size=params["model_params"]["batch_size"],
            callbacks=[checkpoint_callback, early_stopping]
        )

        # Sauvegarder l'historique sous forme de JSON
        save_history_as_json(history.history, metric_dir, prefix)
        # Enregistrer l'historique comme artefact dans MLflow
        mlflow.log_artifact(os.path.join(metric_dir, f"{prefix}training_history.json"))

        plot_metrics(history.history, metric_dir, prefix)

        # Log des métriques
        for epoch, metrics in enumerate(zip(history.history['loss'], history.history['val_loss']), 1):
            mlflow.log_metric("train_loss", metrics[0], step=epoch)
            mlflow.log_metric("val_loss", metrics[1], step=epoch)

        # Sauvegarde du modèle final
        final_model_path = os.path.join(model_dir, "final_model.keras")
        model.save(final_model_path)
        logger.info(f"Modèle final sauvegardé dans : {final_model_path}")

        # Log du modèle dans MLflow
        # Définir la signature des entrées et sorties
        input_schema = Schema([
            TensorSpec(np.dtype("int32"), (-1, max_length), "input_ids"),
            TensorSpec(np.dtype("int32"), (-1, max_length), "attention_mask"),
        ])
        output_schema = Schema([TensorSpec(np.dtype("float32"), (-1, 1))])
        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        #input_example = {
        #    "input_ids": np.array(train_data["input_ids"][:1]).tolist(),
        #    "attention_mask": np.array(train_data["attention_mask"][:1]).tolist()
        #}

        # Log du modèle avec la signature et l'exemple
        mlflow.keras.log_model(
            model,
            artifact_path="model",
            signature=signature,
        #    input_example={
        #        "input_ids": np.array(input_example["input_ids"]).tolist(),
        #        "attention_mask": np.array(input_example["attention_mask"]).tolist(),
        #    }
        )

    return history


def main():
    params_file = "params.yaml"
    train_params = load_params(params_file, "train_distilbert")

    #logger.info("Exécution d'un test rapide pour vérifier l'entraînement.")
    #try:
    #    train_distilbert_model(train_params, quick_test=True)
    #except Exception as e:
    #    logger.error(f"Erreur durant le test rapide : {e}")
    #    return
    
    train_distilbert_model(train_params, quick_test=False)

if __name__ == "__main__":
    main()
