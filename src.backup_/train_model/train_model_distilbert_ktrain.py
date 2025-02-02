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
from sklearn.metrics import confusion_matrix

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

def plot_confusion_matrix(conf_matrix, class_names, output_dir):
    """
    Génère un graphique de matrice de confusion et l'enregistre dans output_dir.
    
    Args:
        conf_matrix (np.ndarray): Matrice de confusion.
        class_names (list): Liste des noms des classes.
        output_dir (str): Répertoire de sortie pour sauvegarder le graphique.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédictions")
    plt.ylabel("Réel")

    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    logger.info(f"Matrice de confusion sauvegardée dans : {confusion_matrix_path}")
    mlflow.log_artifact(confusion_matrix_path)

def plot_lr_find(lrs, losses, optimal_lr, output_dir):
    """
    Génère un graphique de lr_find et l'enregistre dans output_dir.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel("Taux d'apprentissage (log scale)")
    plt.ylabel("Perte")
    plt.title("Recherche du meilleur taux d'apprentissage")
    plt.axvline(x=optimal_lr, color='r', linestyle='--', label=f"Taux optimal : {optimal_lr:.2e}")
    plt.legend()

    lr_plot_path = os.path.join(output_dir, "lr_find_plot.png")
    plt.savefig(lr_plot_path)
    plt.close()
    logger.info(f"Graphique de lr_find sauvegardé dans : {lr_plot_path}")
    mlflow.log_artifact(lr_plot_path)

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

def train_model_with_ktrain(params):
    input_dir = params["input_dir"]
    train_data_path = os.path.join(input_dir, params["input_files"][0])
    train_labels_path = os.path.join(input_dir, params["input_files"][1])
    val_data_path = os.path.join(input_dir, params["input_files"][2])
    val_labels_path = os.path.join(input_dir, params["input_files"][3])

    for path in [train_data_path, train_labels_path, val_data_path, val_labels_path]:
        if not os.path.exists(path):
            logger.error(f"Fichier introuvable : {path}")
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    logger.info(f"Chargement des données depuis : {train_data_path} et {val_data_path}")
    train_data = pd.read_csv(train_data_path)
    train_labels = pd.read_csv(train_labels_path)
    val_data = pd.read_csv(val_data_path)
    val_labels = pd.read_csv(val_labels_path)

    train_data["label"] = train_labels["label"]
    val_data["label"] = val_labels["label"]

    logger.info("Analyse des longueurs des textes...")
    train_data['text_length'] = train_data['feature'].apply(lambda x: len(x.split()))
    val_data['text_length'] = val_data['feature'].apply(lambda x: len(x.split()))
    logger.info(train_data['text_length'].describe())
    logger.info(val_data['text_length'].describe())

    logger.info("Prétraitement des données avec ktrain...")
    trn, val, preproc = text.texts_from_df(
        train_df=train_data,
        text_column="feature",
        label_columns="label",
        val_df=val_data,
        preprocess_mode='distilbert',
        maxlen=params["model_params"]["max_length"]
    )

    logger.info("Calcul des longueurs de séquences après tokenization...")
    sequence_stats = text.seqlen_stats(train_data['feature'].tolist())
    logger.info(f"Statistiques des longueurs de séquences après tokenization : {sequence_stats}")

    model = text.text_classifier('distilbert', train_data=trn, preproc=preproc)

    learner = ktrain.get_learner(
        model, 
        train_data=trn, 
        val_data=val, 
        batch_size=params["model_params"]["batch_size"]
    )

    output_dir = params["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    mlflow.set_tracking_uri(params["mlflow"]["trackingUri"])
    mlflow.set_experiment(params["mlflow"]["experiment"]["name"])

    with mlflow.start_run(run_name=params["mlflow"]["experiment"]["run"]["name"]) as run:
        try:
            run_id = run.info.run_id
            logger.info("Recherche du meilleur taux d'apprentissage avec lr_find...")
            learner.lr_find(show_plot=False)

            optimal_lr = learner.lr_estimate()
            logger.info(f"lr_find : {optimal_lr}")
            optimal_lr = optimal_lr[1] if isinstance(optimal_lr, tuple) else 7.98363529611379e-05
            optimal_lr = min(optimal_lr, 1e-3)

            plot_lr_find(learner.lr_finder.lrs, learner.lr_finder.losses, optimal_lr, output_dir)

            logger.info(f"Taux d'apprentissage sélectionné : {optimal_lr}")
            mlflow.log_params({
                "learning_rate": optimal_lr,
                "batch_size": params["model_params"]["batch_size"],
                "max_length": params["model_params"]["max_length"],
                "epochs": params["model_params"]["epochs"],
            })

            es_callback = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            logger.info("Démarrage de l'entraînement...")
            history = learner.fit_onecycle(lr=optimal_lr, epochs=params["model_params"]["epochs"], callbacks=[es_callback])

            for epoch, (loss, val_loss, acc, val_acc) in enumerate(
                zip(history.history['loss'], history.history['val_loss'], history.history['accuracy'], history.history['val_accuracy'])
            ):
                mlflow.log_metrics({
                    "train_loss": to_scalar(loss),
                    "val_loss": to_scalar(val_loss),
                    "train_accuracy": to_scalar(acc),
                    "val_accuracy": to_scalar(val_acc),
                }, step=epoch)

            model_path = os.path.join(output_dir, "final_model")
            predictor = ktrain.get_predictor(learner.model, preproc)
            predictor.save(model_path)
            mlflow.log_artifact(model_path)
            logger.info(f"Modèle sauvegardé dans : {model_path}")
            save_mlflow_run_id(run_id, output_dir)

            logger.info(f"Colonnes disponibles dans val_data : {val_data.columns}")

            y_pred = predictor.predict(val_data["feature"].tolist())
            logger.info(f"Prédictions effectuées : {y_pred[:10]}")

            #y_true = val_data["label"].astype(str)
            #y_pred = [str(label) for label in y_pred]
            #conf_matrix = confusion_matrix(y_true, y_pred)
            #plot_confusion_matrix(conf_matrix, ["Classe 0", "Classe 1"], output_dir)
        except Exception as e:
            logger.error(f"Une erreur s'est produite : {str(e)}")
            mlflow.log_param("training_error", str(e))
            raise

def main():
    print("GPU Disponible : ", tf.config.list_physical_devices('GPU'))

    params_file = "params.yaml"
    train_params = load_params(params_file, "train_distilbert_ktrain")

    train_model_with_ktrain(train_params)

if __name__ == "__main__":
    main()
