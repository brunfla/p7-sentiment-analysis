from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, Dense
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------
# Gestion des callbacks
# ------------------------------------------------
def get_callbacks(training_cfg, X_val, y_val):
    callbacks = []

    # EarlyStopping
    if training_cfg["earlyStopping"].earlyStopping.enabled:
        if X_val is None or y_val is None:
            logger.warning("EarlyStopping activé, mais aucun ensemble de validation n'est disponible. Ignoré.")
        else:
            logger.info("EarlyStopping activé.")
            early_stopping = EarlyStopping(
                monitor=training_cfg["earlyStopping"].earlyStopping.monitor,
                patience=training_cfg["earlyStopping"].earlyStopping.patience,
                mode=training_cfg["earlyStopping"].earlyStopping.mode
            )
            callbacks.append(early_stopping)

    # LearningRateScheduler
    if training_cfg["learningRateScheduler"].learningRateScheduler.enabled:
        logger.info("Scheduler de Learning Rate activé.")
        lr_scheduler = ReduceLROnPlateau(
            monitor=training_cfg["learningRateScheduler"].learningRateScheduler.monitor,
            factor=training_cfg["learningRateScheduler"].learningRateScheduler.factor,
            patience=training_cfg["learningRateScheduler"].learningRateScheduler.patience,
            min_lr=training_cfg["learningRateScheduler"].learningRateScheduler.min_lr
        )
        callbacks.append(lr_scheduler)

    return callbacks

# ------------------------------------------------
# Entraîner un modèle LSTM
# ------------------------------------------------
def train_lstm(cfg, training_cfg, tuning_cfg, tuning_parameters_cfg, data):
    X_train, y_train, X_val, y_val, X_test, y_test = data

    callbacks = get_callbacks(training_cfg, X_val, y_val)

    model = Sequential([
        Embedding(cfg.model.vocab_size, cfg.model.embedding_dim, input_length=cfg.model.max_length, trainable=True),
        Bidirectional(LSTM(cfg.model.lstm_units[0], return_sequences=True)),
        Dropout(cfg.model.dropout_rate),
        Bidirectional(LSTM(cfg.model.lstm_units[1])),
        Dense(cfg.model.dense_units, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.model.epochs,
        batch_size=cfg.model.batch_size,
        callbacks=callbacks
    )
    return model

# ------------------------------------------------
# Entraîner un modèle Logistic Regression
# ------------------------------------------------
def train_logistic_regression(cfg, tuning_cfg, tuning_parameters_cfg, data):
    X_train, y_train, X_val, y_val, X_test, y_test = data
    model = LogisticRegression(**cfg.model.parameters)
    return apply_tuning(tuning_cfg, tuning_parameters_cfg, model, lambda params, X, y, _: model.fit(X, y), X_train, y_train, X_val, y_val)

