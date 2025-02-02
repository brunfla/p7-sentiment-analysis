import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
import yaml
import json

# Keras / TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def load_params(params_file, section):
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

# (Optionnel) MLflow
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

logger = get_logger(__name__)

def build_bilstm_model(
    max_length,
    embedding_dim,
    lstm_units,
    dropout_rate,
    dense_units,
    learning_rate=1e-3
):
    """
    Construit un modèle BiLSTM en fonction des paramètres.
    lstm_units: liste (ex: [128, 64]) => nb de couches LSTM
    dropout_rate: entre 0 et 1
    dense_units: taille de la couche Dense finale avant la sortie
    max_length, embedding_dim: dimension de l'input (seq_len, emb_dim)
    """
    inputs = Input(shape=(max_length, embedding_dim), name="bilstm_input")
    x = inputs

    for i, units in enumerate(lstm_units):
        # Sur toutes les couches sauf la dernière, on met return_sequences=True
        return_sequences = (i < len(lstm_units) - 1)
        x = Bidirectional(LSTM(units, return_sequences=return_sequences), name=f"bilstm_{i}")(x)
        x = Dropout(dropout_rate, name=f"dropout_{i}")(x)

    # Couche Dense cachée
    x = Dense(dense_units, activation='relu', name='dense_hidden')(x)
    x = Dropout(dropout_rate, name='dense_dropout')(x)

    # Sortie binaire (0 ou 1) => sigmoid
    outputs = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs, outputs, name="BiLSTM_model")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    # 1) Charger les paramètres depuis le YAML
    params_file = "params.yaml"
    section = "train_lstm_bidirectional"

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur de paramétrage dans '{params_file}' (section {section}): {e}")
        raise

    # Récupérer les chemins
    input_train_vec   = params["input_train_vec"]
    input_train_labels= params["input_train_labels"]
    input_val_vec     = params["input_val_vec"]
    input_val_labels  = params["input_val_labels"]
    glove_file        = params.get("glove_file", None)  # pas forcément utilisé si on a déjà les vecteurs
    output_dir        = params["output_dir"]

    # Récupérer les hyperparamètres
    model_params   = params["model_params"]
    max_length     = model_params["max_length"]
    lstm_units     = model_params["lstm_units"]   # ex: [128, 64]
    dropout_rate   = model_params["dropout_rate"] # ex: 0.3
    dense_units    = model_params["dense_units"]  # ex: 32
    batch_size     = model_params.get("batch_size", 128)
    epochs         = model_params.get("epochs", 5)
    learning_rate  = model_params.get("learning_rate", 1e-3)

    # Récup param pour callbacks (earlyStopping, learningRateScheduler, ...)
    training_params = params.get("training_params", {})

    # Param GloVe (si besoin)
    glove_params   = params.get("glove_params", {})
    embedding_dim  = glove_params.get("embedding_dim", 50)  # par ex. 200
    # trainable = glove_params.get("trainable", False)       # si on utilisait une Embedding layer

    # Param data (padding/truncating) - potentiellement déjà géré dans la vectorisation
    data_params = params.get("data_params", {})
    padding    = data_params.get("padding", "post")
    truncating = data_params.get("truncating", "post")
    # (Ici, on suppose que la vectorisation a déjà géré le padding.)

    # 2) Charger les données
    logger.info("Chargement du train set vectorisé...")
    with open(input_train_vec, "rb") as f:
        train_data = pickle.load(f)
        # train_data = {"vectors": np.array([...]), "ids": [...]}
    X_train = train_data["vectors"]
    df_y_train = pd.read_csv(input_train_labels)

    # Charger val set
    logger.info("Chargement du val set vectorisé...")
    with open(input_val_vec, "rb") as f:
        val_data = pickle.load(f)
    X_val = val_data["vectors"]
    df_y_val = pd.read_csv(input_val_labels)

    logger.info(f"X_train shape = {X_train.shape}, X_val shape = {X_val.shape}")
    # On suppose y_train, y_val contiennent une colonne "label" (0 ou 1)
    y_train = df_y_train["label"].values
    y_val   = df_y_val["label"].values

    # 3) Construire le modèle
    logger.info("Construction du modèle BiLSTM...")
    model = build_bilstm_model(
        max_length=max_length,
        embedding_dim=embedding_dim,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        dense_units=dense_units,
        learning_rate=learning_rate
    )
    model.summary(print_fn=logger.info)

    # 4) Préparer les callbacks
    callbacks = []

    # Early Stopping
    if training_params.get("earlyStopping", {}).get("enabled", False):
        es_config = training_params["earlyStopping"]
        monitor = es_config.get("monitor", "val_loss")
        patience = es_config.get("patience", 3)
        mode = es_config.get("mode", "min")
        callbacks.append(EarlyStopping(monitor=monitor, patience=patience, mode=mode, verbose=1))

    # ReduceLROnPlateau
    if training_params.get("learningRateScheduler", {}).get("enabled", False):
        lr_config = training_params["learningRateScheduler"]
        monitor = lr_config.get("monitor", "val_loss")
        factor  = lr_config.get("factor", 0.5)
        patience= lr_config.get("patience", 2)
        min_lr  = lr_config.get("min_lr", 1e-5)
        callbacks.append(ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience,
                                           min_lr=min_lr, verbose=1))

    # (Optionnel) thresholdStop => implémentation custom si besoin
    # ...

    # 5) (Optionnel) MLflow
    mlflow_params = params.get("mlflow", {})
    mlflow_on = False
    if MLFLOW_AVAILABLE and mlflow_params:
        mlflow_on = True
        tracking_uri = mlflow_params.get("trackingUri", None)
        if tracking_uri:  # Configurer l'URI de tracking
            mlflow.set_tracking_uri(tracking_uri)

        experiment_config = mlflow_params.get("experiment", {})
        exp_name = experiment_config.get("name", "Default")
        mlflow.set_experiment(exp_name)

        run_config = experiment_config.get("run", {})
        run_name = run_config.get("name", "LSTM_Bi_model")
        run_description = run_config.get("description", "")
        run_tags = run_config.get("tags", {})

        mlflow_run = mlflow.start_run(run_name=run_name, tags=run_tags)
        run = mlflow.active_run()
        if run_description:
            mlflow.set_tag("mlflow.note.content", run_description)

        # Logguer les hyperparams
        mlflow.log_params({
            "max_length": max_length,
            "embedding_dim": embedding_dim,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "dense_units": dense_units,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate
        })

    # 6) Entraînement
    logger.info("Démarrage de l'entraînement...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 7) Log metrics MLflow (optionnel)
    if mlflow_on:
        # Récupérer l'historique pour la dernière epoch
        final_epoch = len(history.history["loss"]) - 1
        final_train_loss = history.history["loss"][final_epoch]
        final_train_acc = history.history["accuracy"][final_epoch]
        final_val_loss = history.history["val_loss"][final_epoch]
        final_val_acc = history.history["val_accuracy"][final_epoch]

        mlflow.log_metrics({
            "train_loss": final_train_loss,
            "train_accuracy": final_train_acc,
            "val_loss": final_val_loss,
            "val_accuracy": final_val_acc
        })

        if run:
            run_id = run.info.run_id
            mlflow_id_path = os.path.join(output_dir, "mlflow_id.json")
            with open(mlflow_id_path, "w") as f:
                json.dump({"run_id": run_id}, f)
            logger.info(f"MLflow run_id = {run_id} sauvegardé dans : {mlflow_id_path}")
        else:
            logger.error("Erreur : Impossible de récupérer l'ID du run MLflow.")

        # On peut aussi logger l'historique complet
        # for epoch_i in range(len(history.history["loss"])):
        #     mlflow.log_metric("train_loss", history.history["loss"][epoch_i], step=epoch_i)
        #     ...
        mlflow.end_run()

    # 8) Sauvegarder le modèle
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "bilstm_model.h5")
    logger.info(f"Sauvegarde du modèle dans {model_path} ...")
    model.save(model_path)
    logger.info("=== Fin du script d'entraînement BiLSTM ===")


if __name__ == "__main__":
    main()
