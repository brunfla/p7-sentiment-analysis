import mlflow
import mlflow.keras
import logging
import numpy as np
import json

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)

def log_config_to_logs_and_mlflow(cfg):
    """
    Log la configuration complète dans les logs et la sauvegarde comme tag dans MLflow.
    """
    logger.info("Configuration utilisée :")
    logger.info(json.dumps(cfg, indent=4, default=str))
    flattened_cfg = flatten_dict(cfg)
    mlflow.set_tags(flattened_cfg)

def flatten_dict(d, parent_key='', sep='.'):
    """
    Aplatit un dictionnaire pour qu'il soit compatible avec les tags MLflow.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))  # Convertir en chaîne pour MLflow
    return dict(items)

class StopIfBelowThreshold(Callback):
    def __init__(self, threshold, metric="accuracy", patience_batches=10):
        """
        Callback pour arrêter l'entraînement si une métrique est en dessous d'un seuil.

        :param threshold: Seuil en dessous duquel l'entraînement s'arrête.
        :param metric: Nom de la métrique à surveiller (e.g., 'accuracy', 'val_accuracy').
        :param patience_batches: Nombre de batches consécutifs autorisés en dessous du seuil avant arrêt.
        """
        super().__init__()
        self.threshold = threshold
        self.metric = metric
        self.patience_batches = patience_batches
        self.below_threshold_count = 0

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None:
            value = logs.get(self.metric)
            if value is not None and value < self.threshold:
                self.below_threshold_count += 1
                print(f"Batch {batch}: {self.metric}={value:.4f} < seuil={self.threshold:.4f}")
                if self.below_threshold_count >= self.patience_batches:
                    print(
                        f"Arrêt anticipé après {batch + 1} batches : "
                        f"{self.metric} sous le seuil {self.threshold:.4f} pendant {self.patience_batches} batches consécutifs."
                    )
                    self.model.stop_training = True
            else:
                self.below_threshold_count = 0

def validate_config(cfg, required_keys):
    """
    Valide que tous les paramètres nécessaires sont présents dans la configuration.
    """
    for key in required_keys:
        if key not in cfg or cfg[key] is None:
            raise ValueError(f"Le paramètre requis '{key}' est manquant dans la configuration.")

def get_callbacks(training_cfg, X_val, y_val):
    callbacks = []
    validate_config(training_cfg, ["earlyStopping", "learningRateScheduler", "thresholdStop"])
    es_cfg = training_cfg["earlyStopping"]
    lr_cfg = training_cfg["learningRateScheduler"]
    ts_cfg = training_cfg["thresholdStop"]

    if es_cfg["enabled"]:
        if X_val is None or y_val is None:
            raise ValueError("EarlyStopping activé mais aucun ensemble de validation fourni.")
        logger.info("EarlyStopping activé.")
        early_stopping = EarlyStopping(
            monitor=es_cfg["monitor"],
            patience=es_cfg["patience"],
            mode=es_cfg["mode"],
            restore_best_weights=True
        )
        callbacks.append(early_stopping)

    if lr_cfg["enabled"]:
        logger.info("Scheduler de Learning Rate activé.")
        lr_scheduler = ReduceLROnPlateau(
            monitor=lr_cfg["monitor"],
            factor=lr_cfg["factor"],
            patience=lr_cfg["patience"],
            min_lr=lr_cfg["min_lr"]
        )
        callbacks.append(lr_scheduler)

    if ts_cfg["enabled"]:
        logger.info("StopIfBelowThreshold activé.")
        stop_if_below_threshold = StopIfBelowThreshold(
            threshold=ts_cfg["threshold"],
            metric=ts_cfg["metric"],
            patience_batches=ts_cfg["patience_batches"]
        )
        callbacks.append(stop_if_below_threshold)

    return callbacks

def create_lstm_model(
    vocab_size,
    embedding_dim,
    max_length,
    lstm_units_0,
    lstm_units_1,
    dropout_rate,
    dense_units
):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length, trainable=True),
        Bidirectional(LSTM(lstm_units_0, return_sequences=True)),
        Dropout(dropout_rate),
        Bidirectional(LSTM(lstm_units_1)),
        Dense(dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_lstm(cfg, training_cfg, tuning_cfg, mlflow_cfg, partitioned_data):
    required_keys = [
        "vocab_size", "embedding_dim", "max_length", "lstm_units", "dropout_rate",
        "dense_units", "batch_size", "epochs"
    ]
    validate_config(cfg, required_keys)

    X_train, y_train = partitioned_data.data["X_train"], partitioned_data.data["y_train"]
    X_val, y_val = partitioned_data.data.get("X_val"), partitioned_data.data.get("y_val")
    logger.info(f"Shape de X_train : {X_train.shape}")
    logger.info(f"Shape de y_train : {y_train.shape}")

    callbacks = get_callbacks(training_cfg, X_val, y_val)
    validation_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None

    base_params = {
        "vocab_size": cfg["vocab_size"],
        "embedding_dim": cfg["embedding_dim"],
        "max_length": cfg["max_length"],
        "lstm_units_0": cfg["lstm_units"][0],
        "lstm_units_1": cfg["lstm_units"][1],
        "dropout_rate": cfg["dropout_rate"],
        "dense_units": cfg["dense_units"]
    }

    mlflow.set_tracking_uri(mlflow_cfg["trackingUri"])
    mlflow.set_experiment(mlflow_cfg["experiment"]["name"])

    run_name = mlflow_cfg["experiment"]["run"]["name"]
    run_description = mlflow_cfg["experiment"]["run"]["description"]
    run_tags = mlflow_cfg["experiment"]["run"]["tags"]

    with mlflow.start_run(run_name=run_name, description=run_description):
        log_config_to_logs_and_mlflow(cfg)
        if run_tags:
            mlflow.set_tags(run_tags)

        if tuning_cfg.get("type") == "grid_search":
            logger.info("Tuning LSTM via GridSearchCV (KerasClassifier).")
            model_keras_clf = KerasClassifier(
                model=create_lstm_model,
                vocab_size=base_params["vocab_size"],
                embedding_dim=base_params["embedding_dim"],
                max_length=base_params["max_length"],
                lstm_units_0=base_params["lstm_units_0"],
                lstm_units_1=base_params["lstm_units_1"],
                dropout_rate=base_params["dropout_rate"],
                dense_units=base_params["dense_units"],
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                verbose=1
            )
            param_grid = {
                "lstm_units_0": tuning_cfg.get('paramGrid', {}).get('lstm_units_0', [64, 128]),
                "lstm_units_1": tuning_cfg.get('paramGrid', {}).get('lstm_units_1', [64, 128]),
                "dropout_rate": tuning_cfg.get('paramGrid', {}).get('dropout_rate', [0.2, 0.4]),
                "dense_units": tuning_cfg.get('paramGrid', {}).get('dense_units', [32, 64])
            }
            logger.info(f"GridSearch param grid : {param_grid}")
            grid_search = GridSearchCV(
                estimator=model_keras_clf,
                param_grid=param_grid,
                cv=tuning_cfg.get("cv", 3),
                verbose=3,
                error_score="raise",
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train, validation_data=validation_data, callbacks=callbacks)
            best_estimator = grid_search.best_estimator_
            final_model = best_estimator.model_
            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)
        else:
            logger.info("Pas de tuning => entraînement direct.")
            model = create_lstm_model(**base_params)
            history = model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=cfg["epochs"],
                batch_size=cfg["batch_size"],
                callbacks=callbacks
            )
            for epoch in range(cfg["epochs"]):
                mlflow.log_metric("loss", history.history['loss'][epoch], step=epoch)
                mlflow.log_metric("accuracy", history.history['accuracy'][epoch], step=epoch)
                if validation_data is not None:
                    mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)
                    mlflow.log_metric("val_accuracy", history.history['val_accuracy'][epoch], step=epoch)
            final_model = model
        mlflow.keras.log_model(final_model, "lstm_model")
    return final_model
