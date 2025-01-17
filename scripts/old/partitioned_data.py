import os
import pickle
import logging
import psutil
import pandas as pd
from sklearn.model_selection import train_test_split, KFold

# -------------------------
# Configurer le logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Classe PartitionedData
# ------------------------------------------------
class PartitionedData:
    def __init__(self, partition_type, data, metadata=None):
        self.partition_type = partition_type
        self.data = data
        self.metadata = metadata or {}

    @staticmethod
    def create(X, y, partition_cfg, metadata):
        if partition_cfg._target_ == "trainValTest":
            X_train_full, X_test, y_train_full, y_test = train_test_split(
                X, y,
                test_size=partition_cfg.testSize,
                random_state=partition_cfg.randomSeed
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=partition_cfg.validationSize,
                random_state=partition_cfg.randomSeed
            )
            logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            return PartitionedData(
                "trainValTest",
                {
                    "X_train": X_train, "y_train": y_train,
                    "X_val": X_val, "y_val": y_val,
                    "X_test": X_test, "y_test": y_test
                },
                metadata
            )

        elif partition_cfg._target_ == "trainTest":
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=partition_cfg.testSize,
                random_state=partition_cfg.randomSeed
            )
            logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            return PartitionedData(
                "trainTest",
                {
                    "X_train": X_train, "y_train": y_train,
                    "X_test": X_test, "y_test": y_test
                },
                metadata
            )

        elif partition_cfg._target_ == "crossValidation":
            kfold = KFold(
                n_splits=partition_cfg.folds,
                shuffle=True,
                random_state=partition_cfg.randomSeed
            )
            folds = list(kfold.split(X, y))
            logger.info(f"Nombre de folds: {len(folds)}")
            return PartitionedData(
                "crossValidation",
                {
                    "folds": folds, "X": X, "y": y
                },
                metadata
            )

        else:
            raise ValueError(f"Partition de découpage non reconnue: {partition_cfg._target_}")

    def save(self, output_path):
        with open(output_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Données partitionnées sauvegardées avec succès dans {output_path} (type: {type(self)})")

    @staticmethod
    def load(input_path):
        with open(input_path, "rb") as f:
            partitioned_data = pickle.load(f)
            if not isinstance(partitioned_data, PartitionedData):
                raise ValueError(f"Le fichier {input_path} ne contient pas un objet PartitionedData valide.")
            logger.info(f"Données partitionnées chargées depuis {input_path}")
            return partitioned_data

# ------------------------------------------------
# Fonction pour afficher les métriques système
# ------------------------------------------------
def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

# ------------------------------------------------
# Fonction pour charger les données depuis un fichier CSV
# ------------------------------------------------
def load_data(input_path, text_column, label_column):
    logger.info(f"Chargement des données depuis {input_path}...")
    data = pd.read_csv(input_path)

    if text_column not in data.columns or label_column not in data.columns:
        raise ValueError(
            f"Les colonnes spécifiées ne sont pas présentes dans le fichier. "
            f"Colonnes trouvées : {list(data.columns)}"
        )

    X = data[text_column]
    y = data[label_column]

    logger.info(f"Données chargées avec succès. Taille X: {len(X)}, Taille y: {len(y)}")
    return X, y