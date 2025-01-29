import os
import pandas as pd
import sys
from tweet_cleaning import clean_tweet, lemmatize_and_remove_stopwords

# Sauvegarder l'état initial de sys.path
original_sys_path = sys.path.copy()
# Ajouter le répertoire parent au chemin pour l'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params_utils.params_utils import load_params  
from logging_utils.logging_utils import get_logger
from clean_tweet_glove import is_valid_tweet, validate_and_clean  # Import des fonctions
# Restaurer l'état initial de sys.path
sys.path = original_sys_path

# Obtenir le logger
logger = get_logger(__name__)

def preprocess_batch_cleaning(input_file_x, output_file_x):
    """
    Applique les étapes de nettoyage sur un fichier contenant les features (x_*.csv).

    Args:
        input_file_x (str): Chemin du fichier d'entrée des features.
        output_file_x (str): Chemin du fichier de sortie des features.

    Returns:
        pd.DataFrame: Données nettoyées des features.
    """
    # Charger les données
    logger.info(f"Chargement des données depuis : {input_file_x}")
    data_x = pd.read_csv(input_file_x)

    if "feature" not in data_x.columns:
        raise ValueError(f"La colonne 'feature' est absente dans {input_file_x}.")

    # Appliquer les étapes de nettoyage sur les features
    logger.info("Début du nettoyage des tweets...")
    data_x["feature"] = data_x["feature"].apply(
        lambda tweet: lemmatize_and_remove_stopwords(clean_tweet(tweet))
    )

    return data_x

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
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Créer le répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)

    # Traiter uniquement les fichiers x_
    for input_file in input_files:
        if input_file.startswith("x_"):
            input_path_x = os.path.join(input_dir, input_file)
            input_path_y = os.path.join(input_dir, input_file.replace("x_", "y_"))
            output_path_x = os.path.join(output_dir, input_file)
            output_path_y = os.path.join(output_dir, input_file.replace("x_", "y_"))

            try:
                # Nettoyage des features
                logger.info(f"Nettoyage des features depuis : {input_path_x}")
                data_x = preprocess_batch_cleaning(input_path_x, output_path_x)

                # Charger les labels
                logger.info(f"Chargement des labels depuis : {input_path_y}")
                data_y = pd.read_csv(input_path_y)

                if "id" not in data_y.columns:
                    raise ValueError(f"La colonne 'id' est absente dans {input_path_y}.")

                # Synchroniser les features et labels
                logger.info("Validation et synchronisation des features et labels...")
                data_x, data_y = validate_and_clean(data_x, data_y)

                # Sauvegarder les données synchronisées
                data_x.to_csv(output_path_x, index=False)
                data_y.to_csv(output_path_y, index=False)
                logger.info(f"Données synchronisées sauvegardées dans : {output_path_x} et {output_path_y}")

            except Exception as e:
                logger.error(f"Erreur lors du traitement du fichier {input_file} : {e}")
