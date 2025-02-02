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

def split_traintest(input_path, output_path, text_column, label_column, test_size=0.3, random_state=42):
    """
    Partitionne les données en ensembles d'entraînement et de test.

    Args:
        input_path (str): Chemin vers le fichier d'entrée.
        output_path (str): Répertoire pour sauvegarder les partitions.
        text_column (str): Colonne contenant les tweets.
        label_column (str): Colonne contenant les labels.
        test_size (float): Proportion des données pour l'ensemble de test.
        random_state (int): Graine aléatoire pour la reproductibilité.

    Returns:
        None
    """
    logger.info(f"Chargement des données depuis : {input_path}")
    data = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Dataset chargé avec {len(data)} lignes.")

    # Vérifier la présence des colonnes nécessaires
    if text_column not in data.columns or label_column not in data.columns:
        logger.error(f"Les colonnes nécessaires '{text_column}' ou '{label_column}' sont absentes.")
        raise ValueError(f"Les colonnes '{text_column}' ou '{label_column}' sont absentes du fichier d'entrée.")

    # Suppression des tweets vides
    logger.info("Suppression des tweets vides...")
    initial_size = len(data)
    data = data[~data[text_column].isna()]
    data = data[data[text_column].str.strip() != ""]
    logger.info(f"{initial_size - len(data)} lignes supprimées à cause de tweets vides.")

    # Validation et conversion des types pour la colonne label_column
    logger.info("Validation et conversion des types de la colonne de labels...")
    initial_size = len(data)
    data[label_column] = pd.to_numeric(data[label_column], errors='coerce')
    data = data.dropna(subset=[label_column])
    data[label_column] = data[label_column].astype(int)
    logger.info(f"{initial_size - len(data)} lignes supprimées à cause de valeurs invalides dans '{label_column}'.")

    # Validation des données nettoyées
    if data.empty:
        logger.error("Les données sont vides après nettoyage. Vérifiez le fichier d'entrée.")
        raise ValueError("Les données sont vides après nettoyage. Impossible de continuer.")

    logger.info(f"Nombre de lignes restantes après nettoyage : {len(data)}")

    # Partitionnement
    logger.info("Début du partitionnement train/test...")
    os.makedirs(output_path, exist_ok=True)
    train, test = train_test_split(data, test_size=test_size, random_state=random_state, stratify=data[label_column])
    
    # Sauvegarde des partitions
    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)
    logger.info(f"Partitions train/test sauvegardées dans {output_path}")

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
        test_size = params.get("test_size", 0.3)
        random_state = params.get("random_state", 42)
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Appeler la fonction de partitionnement
    split_traintest(input_path, output_path, text_column, label_column, test_size, random_state)
