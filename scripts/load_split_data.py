import pickle
import logging

logger = logging.getLogger(__name__)

def load_partitioned_data(dataset_path, partition_cfg):
    """
    Charge les données partitionnées en fonction de la configuration.

    Args:
        dataset_path (str): Chemin vers le fichier pickle contenant les données partitionnées.
        partition_cfg (DictConfig): Configuration du partitionnement.

    Returns:
        Tuple contenant les données partitionnées, ou None si la configuration est invalide.
    """
    try:
        with open(dataset_path, "rb") as f:
            if partition_cfg._target_ == "trainValTest":
                logger.info("Chargement des données partitionnées pour 'trainValTest'.")
                X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)
                return X_train, y_train, X_val, y_val, X_test, y_test

            elif partition_cfg._target_ == "trainTest":
                logger.info("Chargement des données partitionnées pour 'trainTest'.")
                X_train, y_train, X_test, y_test = pickle.load(f)
                X_val, y_val = None, None
                return X_train, y_train, X_val, y_val, X_test, y_test

            elif partition_cfg._target_ == "crossValidation":
                logger.info("Chargement des données partitionnées pour 'crossValidation'.")
                folds, X, y = pickle.load(f)
                return folds, X, y

            else:
                raise ValueError(f"Partition de découpage non reconnue: {partition_cfg._target_}")
    except FileNotFoundError:
        logger.error(f"Le fichier {dataset_path} est introuvable.")
        raise
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données partitionnées : {e}")
        raise

