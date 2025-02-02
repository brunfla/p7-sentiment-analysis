import logging
from tensorflow.keras.callbacks import Callback

logger = logging.getLogger(__name__)

class StopIfBelowThreshold(Callback):
    """
    Callback pour arrêter l'entraînement si une métrique est en dessous d'un seuil.
    """
    def __init__(self, threshold, metric="accuracy", patience_batches=10):
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
                logger.info(
                    f"Batch {batch}: {self.metric}={value:.4f} < seuil={self.threshold:.4f}."
                )
                if self.below_threshold_count >= self.patience_batches:
                    logger.info(
                        f"Arrêt anticipé : {self.metric} est sous le seuil {self.threshold:.4f} pendant {self.patience_batches} batches."
                    )
                    self.model.stop_training = True
            else:
                self.below_threshold_count = 0


def flatten_dict(d, parent_key='', sep='.'):
    """
    Aplati un dictionnaire pour MLflow.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, str(v)))
    return dict(items)

