import os
import json
import mlflow
import logging
import pandas as pd
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, create_optimizer
from datasets import Dataset
from params_utils import load_params
from logging_utils import get_logger

logger = get_logger(__name__)

def prepare_data(tokenizer, file_path, label_column, text_column, max_length):
    """
    Prépare les données pour l'entraînement en utilisant un tokenizer.
    """
    logger.info(f"Chargement des données depuis : {file_path}")
    data = pd.read_csv(file_path)
    data = data.dropna(subset=[text_column, label_column])
    data = data[data[text_column].str.strip() != ""]
    
    dataset = Dataset.from_pandas(data)
    
    logger.info("Tokenisation des tweets...")
    dataset = dataset.map(
        lambda x: tokenizer(x[text_column], truncation=True, padding=True, max_length=max_length),
        batched=True
    )
    dataset = dataset.rename_column(label_column, "labels")
    dataset.set_format(type="tensorflow", columns=["input_ids", "attention_mask", "labels"])
    
    return dataset

def convert_to_tf_dataset(dataset, batch_size):
    """
    Convertit un objet Dataset en tf.data.Dataset compatible TensorFlow.
    """
    logger.info("Conversion du dataset Hugging Face en tf.data.Dataset...")
    tf_dataset = tf.data.Dataset.from_generator(
        lambda: dataset,
        output_signature=(
            {
                "input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32),
                "attention_mask": tf.TensorSpec(shape=(None,), dtype=tf.int32),
            },
            tf.TensorSpec(shape=(), dtype=tf.int64),
        )
    )
    return tf_dataset.batch(batch_size)

def train_distilbert(params, train_file, val_file):
    """
    Entraîne un modèle DistilBERT avec des données tokenisées.
    """
    logger.info("Chargement du tokenizer et du modèle DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = TFDistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )
    
    # Préparation des données
    train_dataset = prepare_data(
        tokenizer, 
        train_file, 
        "id", 
        "tweet", 
        params["model_params"]["max_length"]
    )
    val_dataset = prepare_data(
        tokenizer, 
        val_file, 
        "id", 
        "tweet", 
        params["model_params"]["max_length"]
    )

    train_tensors = convert_to_tf_dataset(train_dataset, params["model_params"]["batch_size"])
    val_tensors = convert_to_tf_dataset(val_dataset, params["model_params"]["batch_size"])

    # Calculer les steps par epoch et le nombre total de steps
    steps_per_epoch = len(train_dataset) // params["model_params"]["batch_size"]
    total_steps = steps_per_epoch * params["model_params"]["epochs"]
    warmup_steps = int(0.1 * total_steps)  # Exemple : 10% des pas totaux pour le warmup

    optimizer, lr_schedule = create_optimizer(
        init_lr=params["training_params"]["learningRateScheduler"]["factor"],
        num_train_steps=total_steps,
        num_warmup_steps=warmup_steps,
        weight_decay_rate=0.01
    )

    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=["accuracy"])

    # Callbacks
    callbacks = []
    if params["training_params"]["earlyStopping"]["enabled"]:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor=params["training_params"]["earlyStopping"]["monitor"],
            patience=params["training_params"]["earlyStopping"]["patience"],
            mode=params["training_params"]["earlyStopping"]["mode"],
            restore_best_weights=True
        ))

    # Entraînement
    logger.info("Début de l'entraînement...")
    history = model.fit(
        train_tensors,
        validation_data=val_tensors,
        epochs=params["model_params"]["epochs"],
        batch_size=params["model_params"]["batch_size"],
        callbacks=callbacks
    )

    # Configuration MLflow
    mlflow.set_tracking_uri(params["mlflow"]["trackingUri"])
    mlflow.set_experiment(params["mlflow"]["experiment"]["name"])

    # Enregistrer le modèle avec MLflow
    with mlflow.start_run(run_name=params["mlflow"]["experiment"]["run"]["name"]):
        mlflow.log_params(params["model_params"])
        mlflow.tensorflow.log_model(model, artifact_path="bert_with_glove_model")
        logger.info("Modèle entraîné et sauvegardé avec MLflow.")

def main():
    params_file = "params.yaml"
    section = "train_bert_with_glove"

    params = load_params(params_file, section)
    train_distilbert(params, params["input_train_file"], params["input_val_file"])

if __name__ == "__main__":
    main()
