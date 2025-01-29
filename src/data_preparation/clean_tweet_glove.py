import os
import pandas as pd
import sys
from tweet_cleaning import clean_tweet_with_glove, is_valid_tweet, validate_and_clean
from glove_loader import load_glove_model  # Import du loader GloVe

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

def preprocess_with_glove(input_file_x, input_file_y, output_file_x, output_file_y, glove_file, threshold=0.6):
    """
    Applique le nettoyage avec GloVe sur un fichier contenant les features (x_*.csv) et synchronise avec y.

    Args:
        input_file_x (str): Chemin du fichier d'entrée des features nettoyées de base.
        input_file_y (str): Chemin du fichier d'entrée des labels.
        output_file_x (str): Chemin pour sauvegarder le fichier features nettoyé avec GloVe.
        output_file_y (str): Chemin pour sauvegarder le fichier labels synchronisé.
        glove_file (str): Chemin vers le fichier GloVe.
        threshold (float): Seuil pour la similarité cosinus.

    Returns:
        None
    """
    logger.info(f"Chargement du modèle GloVe depuis : {glove_file}")
    glove_model = load_glove_model(glove_file)  # Chargement via le cache GloVe

    logger.info(f"Chargement des données depuis : {input_file_x} et {input_file_y}")
    data_x = pd.read_csv(input_file_x)
    data_y = pd.read_csv(input_file_y)

    if "feature" not in data_x.columns or "id" not in data_y.columns:
        raise ValueError(f"Les colonnes 'feature' ou 'id' sont absentes dans les fichiers d'entrée.")

    # Nettoyage des tweets avec GloVe
    logger.info("Nettoyage des tweets avec GloVe...")
    data_x["feature"] = data_x["feature"].apply(
        lambda tweet: clean_tweet_with_glove(tweet, glove_model, threshold)
    )

    # Validation et synchronisation des features et labels
    logger.info("Validation et synchronisation des features et labels...")
    data_x, data_y = validate_and_clean(data_x, data_y)

    # Sauvegarder les fichiers nettoyés
    os.makedirs(os.path.dirname(output_file_x), exist_ok=True)
    data_x.to_csv(output_file_x, index=False)
    data_y.to_csv(output_file_y, index=False)
    logger.info(f"Fichiers nettoyés sauvegardés dans : {output_file_x} et {output_file_y}")

if __name__ == "__main__":
    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]  # Nom du script comme section

    try:
        preprocess_params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        input_dir = preprocess_params["input_dir"]
        input_files = preprocess_params["input_files"]
        output_dir = preprocess_params["output_dir"]
        glove_file = preprocess_params["glove_file"]
        threshold = preprocess_params.get("glove_similarity_threshold", 0.6)
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Créer le répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Traiter chaque fichier d'entrée
    for input_file in input_files:
        input_path_x = os.path.join(input_dir, input_file)
        input_path_y = os.path.join(input_dir, input_file.replace("x_", "y_"))
        output_path_x = os.path.join(output_dir, input_file)
        output_path_y = os.path.join(output_dir, input_file.replace("x_", "y_"))

        try:
            preprocess_with_glove(input_path_x, input_path_y, output_path_x, output_path_y, glove_file, threshold)
        except Exception as e:
            logger.error(f"Erreur lors du traitement du fichier {input_file} : {e}")
