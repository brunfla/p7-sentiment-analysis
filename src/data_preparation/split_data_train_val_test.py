import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

# Sauvegarder l'état initial de sys.path
original_sys_path = sys.path.copy()
# Ajouter le répertoire parent au chemin pour l'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params_utils.params_utils import load_params  
from logging_utils.logging_utils import get_logger
# Restaurer l'état initial de sys.path
sys.path = original_sys_path

# Obtenir le logger
logger = get_logger(__name__)

def split_trainvaltest(input_path, output_path, text_column, label_column, id_column, test_size=0.2, val_size=0.1, random_state=42):
    """
    Partitionne les données en ensembles d'entraînement, de validation et de test avec des fichiers séparés pour les features et les labels.

    Args:
        input_path (str): Chemin vers le fichier d'entrée.
        output_path (str): Répertoire pour sauvegarder les partitions.
        text_column (str): Colonne contenant les tweets (features).
        label_column (str): Colonne contenant les labels.
        id_column (str): Colonne contenant les identifiants uniques.
        test_size (float): Proportion des données pour l'ensemble de test.
        val_size (float): Proportion des données pour l'ensemble de validation, par rapport à l'ensemble total.
        random_state (int): Graine aléatoire pour la reproductibilité.

    Returns:
        None
    """
    logger.info(f"Chargement des données depuis : {input_path}")
    data = pd.read_csv(input_path, header=None, names=["id", "timestamp", "date", "query", "user", "tweet"])
    logger.info(f"Dataset chargé avec {len(data)} lignes.")

    # Vérifier la présence des colonnes nécessaires
    required_columns = [text_column, label_column, id_column]
    for column in required_columns:
        if column not in data.columns:
            logger.error(f"La colonne nécessaire '{column}' est absente.")
            raise ValueError(f"La colonne '{column}' est absente du fichier d'entrée.")
        
    data["id"] = data["id"].apply(lambda x: 1 if x == 4 else x)

    # Supprimer les doublons d'IDs
    if not data[id_column].is_unique:
        logger.warning(f"La colonne '{id_column}' contient des IDs non uniques. Suppression des doublons...")
        initial_size = len(data)
        data = data.drop_duplicates(subset=[id_column])
        logger.info(f"{initial_size - len(data)} doublons d'IDs supprimés.")

    # Supprimer les tweets en double
    logger.info("Suppression des tweets en double...")
    initial_size = len(data)
    data = data.drop_duplicates(subset=[text_column])
    logger.info(f"{initial_size - len(data)} doublons de tweets supprimés.")

    # Calcul des proportions pour validation et test
    test_ratio = test_size
    val_ratio = val_size / (1 - test_size)  # Proportion relative à l'ensemble train+val

    # Partitionnement train/test
    logger.info("Début du partitionnement train/val/test...")
    os.makedirs(output_path, exist_ok=True)
    train_val, test = train_test_split(data, test_size=test_ratio, random_state=random_state, stratify=data[label_column])
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state, stratify=train_val[label_column])

    # Sauvegarde des partitions
    logger.info("Sauvegarde des partitions dans des fichiers séparés pour features et labels...")

    # Train
    train[[id_column, text_column]].rename(columns={id_column: "id", text_column: "feature"}).to_csv(os.path.join(output_path, "x_train.csv"), index=False)
    train[[id_column, label_column]].rename(columns={id_column: "id", label_column: "label"}).to_csv(os.path.join(output_path, "y_train.csv"), index=False)

    # Validation
    val[[id_column, text_column]].rename(columns={id_column: "id", text_column: "feature"}).to_csv(os.path.join(output_path, "x_val.csv"), index=False)
    val[[id_column, label_column]].rename(columns={id_column: "id", label_column: "label"}).to_csv(os.path.join(output_path, "y_val.csv"), index=False)

    # Test
    test[[id_column, text_column]].rename(columns={id_column: "id", text_column: "feature"}).to_csv(os.path.join(output_path, "x_test.csv"), index=False)
    test[[id_column, label_column]].rename(columns={id_column: "id", label_column: "label"}).to_csv(os.path.join(output_path, "y_test.csv"), index=False)

    logger.info(f"Partitions train/val/test sauvegardées dans {output_path}")

if __name__ == "__main__":
    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]  # Nom du script comme section

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        # Extraire les paramètres nécessaires
        input_path = params["input_file"]
        output_path = params["output_dir"]
        text_column = params["text_column"]
        label_column = params["label_column"]
        id_column = params["id_column"]
        test_size = params.get("test_size", 0.2)
        val_size = params.get("val_size", 0.1)
        random_state = params.get("random_state", 42)
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Appeler la fonction de partitionnement
    split_trainvaltest(input_path, output_path, text_column, label_column, id_column, test_size, val_size, random_state)
