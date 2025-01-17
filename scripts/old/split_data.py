import os
import logging
import psutil
import pandas as pd
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sklearn.model_selection import train_test_split, StratifiedKFold

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
# Fonction pour afficher les métriques système
# ------------------------------------------------
def log_system_metrics():
    """
    Affiche les métriques CPU et mémoire pour le monitoring.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

# ------------------------------------------------
# Fonction pour charger et nettoyer les données depuis un fichier CSV
# ------------------------------------------------
def load_data(input_path, text_column, label_column):
    """
    Charge les données d'un fichier CSV, supprime les tweets vides, et transforme les classes 4 -> 1.

    Args:
        input_path (str): Chemin du fichier d'entrée.
        text_column (str): Nom de la colonne contenant les tweets.
        label_column (str): Nom de la colonne contenant les labels.

    Returns:
        pd.DataFrame: Données nettoyées.
    """
    logger.info(f"Chargement des données depuis {input_path}...")
    data = pd.read_csv(input_path, header=None, names=["id", "timestamp", "date", "query", "user", "tweet"])
    logger.info(f"Dataset chargé avec {len(data)} lignes.")

    # Transformer les classes 4 -> 1
    logger.info("Transformation des classes : 4 -> 1...")
    data["id"] = data["id"].apply(lambda x: 1 if x == 4 else x)

    # Supprimer les tweets vides
    logger.info("Suppression des tweets vides...")
    initial_size = len(data)
    data = data[~data[text_column].isna()]
    data = data[data[text_column].str.strip() != ""]
    logger.info(f"{initial_size - len(data)} tweets vides supprimés. Taille finale : {len(data)}")

    return data

# ------------------------------------------------
# Fonction pour initialiser les fichiers de sortie
# ------------------------------------------------
def init_output(file_paths):
    """
    Initialise les fichiers de sortie.
    Crée un fichier vide ou écrase un fichier existant.

    Args:
        file_paths (list): Liste des chemins des fichiers à initialiser.
    """
    for file_path in file_paths:
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        # Écrire un fichier CSV vide
        pd.DataFrame().to_csv(file_path, index=False)
        logger.info(f"Fichier initialisé : {file_path}")

# ------------------------------------------------
# Fonction de partitionnement des données
# ------------------------------------------------
def partition_data(data, strategy, text_column, label_column, output_path, random_state=42):
    """
    Partitionne les données en fonction de la stratégie spécifiée :
    - trainTest : Partitionnement simple en train/test.
    - trainValTest : Partitionnement en train/validation/test.
    - crossValidation : Génère des splits pour une validation croisée.

    Args:
        data (pd.DataFrame): Données brutes.
        strategy (str): Stratégie de partitionnement ("trainTest", "trainValTest", "crossValidation").
        text_column (str): Nom de la colonne contenant les tweets.
        label_column (str): Nom de la colonne contenant les labels.
        output_path (str): Répertoire où sauvegarder les partitions.
        random_state (int): État aléatoire pour la reproductibilité.

    Returns:
        None
    """
    if strategy == "trainTest":
        logger.info("Partitionnement train/test...")
        train_path = os.path.join(output_path, "train.csv")
        test_path = os.path.join(output_path, "test.csv")
        init_output([train_path, test_path])

        train, test = train_test_split(data, test_size=0.3, random_state=random_state, stratify=data[label_column])
        train.to_csv(train_path, mode='a', index=False, header=True)
        test.to_csv(test_path, mode='a', index=False, header=True)

    elif strategy == "trainValTest":
        logger.info("Partitionnement train/validation/test...")
        train_path = os.path.join(output_path, "train.csv")
        val_path = os.path.join(output_path, "val.csv")
        test_path = os.path.join(output_path, "test.csv")
        init_output([train_path, val_path, test_path])

        train, temp = train_test_split(data, test_size=0.3, random_state=random_state, stratify=data[label_column])
        val, test = train_test_split(temp, test_size=0.5, random_state=random_state, stratify=temp[label_column])
        train.to_csv(train_path, mode='a', index=False, header=True)
        val.to_csv(val_path, mode='a', index=False, header=True)
        test.to_csv(test_path, mode='a', index=False, header=True)

    elif strategy == "crossValidation":
        logger.info("Partitionnement pour validation croisée...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        file_paths = [os.path.join(output_path, f"train_fold_{i}.csv") for i in range(5)] + \
                     [os.path.join(output_path, f"val_fold_{i}.csv") for i in range(5)]
        init_output(file_paths)

        for fold, (train_idx, val_idx) in enumerate(skf.split(data[text_column], data[label_column])):
            train_path = os.path.join(output_path, f"train_fold_{fold}.csv")
            val_path = os.path.join(output_path, f"val_fold_{fold}.csv")
            train_fold = data.iloc[train_idx]
            val_fold = data.iloc[val_idx]
            train_fold.to_csv(train_path, mode='a', index=False, header=True)
            val_fold.to_csv(val_path, mode='a', index=False, header=True)

    else:
        raise ValueError("Stratégie non reconnue : choisissez parmi 'trainTest', 'trainValTest', ou 'crossValidation'.")

    logger.info(f"Partitions sauvegardées dans {output_path}")

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    # Afficher les métriques système au démarrage
    log_system_metrics()

    # Charger la configuration avec Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    # Récupérer les paramètres de configuration
    input_path = cfg.dataset.path
    text_column = cfg.dataset.text_column
    label_column = cfg.dataset.label_column
    partition_strategy = cfg.preprocess.partitioner.strategy
    output_path = cfg.preprocess.partitioner.output
    random_state = cfg.resources.random_state

    # Charger et nettoyer les données
    data = load_data(input_path, text_column, label_column)

    # Partitionner les données
    partition_data(data, partition_strategy, text_column, label_column, output_path, random_state)

    # Afficher les métriques système à la fin
    log_system_metrics()

if __name__ == "__main__":
    main()

