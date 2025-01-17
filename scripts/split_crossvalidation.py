import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from logging_utils import get_logger
from params_utils import load_params

# Obtenir le logger
logger = get_logger(__name__)

def stratified_kfold_split(input_path, output_path, text_column, label_column, random_state=42, n_splits=5):
    """
    Partitionne les données pour une validation croisée stratifiée.

    Args:
        input_path (str): Chemin vers le fichier d'entrée.
        output_path (str): Répertoire pour sauvegarder les partitions.
        text_column (str): Colonne contenant les textes.
        label_column (str): Colonne contenant les labels.
        random_state (int): Graine aléatoire pour la reproductibilité.
        n_splits (int): Nombre de folds pour la validation croisée.

    Returns:
        None
    """
    logger.info(f"Chargement des données depuis : {input_path}")
    data = pd.read_csv(input_path, low_memory=False)
    logger.info(f"Dataset chargé avec {len(data)} lignes.")

    # Vérifier la présence des colonnes nécessaires
    if text_column not in data.columns or label_column not in data.columns:
        raise ValueError(f"Les colonnes nécessaires '{text_column}' ou '{label_column}' sont absentes dans le dataset.")

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
        raise ValueError("Les données sont vides après nettoyage. Impossible de continuer.")

    logger.info(f"Nombre de lignes restantes après nettoyage : {len(data)}")

    # Partitionnement pour validation croisée
    logger.info("Début du partitionnement pour validation croisée...")
    os.makedirs(output_path, exist_ok=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (train_idx, val_idx) in enumerate(skf.split(data[text_column], data[label_column])):
        train = data.iloc[train_idx]
        val = data.iloc[val_idx]

        # Sauvegarder les partitions
        train.to_csv(os.path.join(output_path, f"train_fold_{fold}.csv"), index=False)
        val.to_csv(os.path.join(output_path, f"val_fold_{fold}.csv"), index=False)
        logger.info(f"Partitions pour le fold {fold} sauvegardées.")

    logger.info(f"Partitions pour validation croisée sauvegardées dans {output_path}")

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
        random_state = params.get("random_state", 42)
        n_splits = params.get("n_splits", 5)
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Appeler la fonction de partitionnement
    stratified_kfold_split(input_path, output_path, text_column, label_column, random_state, n_splits)
