import os
import pandas as pd
import sys
from preprocess_utils import preprocess_tweet, lemmatize_and_remove_stopwords
from params_utils import load_params
from logging_utils import get_logger

# Obtenir le logger
logger = get_logger(__name__)

def preprocess_batch_cleaning(input_path, output_path):
    """
    Applique les étapes de nettoyage sur un fichier en mode batch.
    """

    # Charger les données
    logger.info(f"Chargement des données depuis : {input_path}")
    data = pd.read_csv(input_path, header=None, names=["id", "timestamp", "date", "query", "user", "tweet"])
    logger.info(f"Dataset chargé avec {len(data)} lignes.")
    data["id"] = data["id"].apply(lambda x: 1 if x == 4 else x)  # Transformer 4 -> 1

    text_column = "tweet"
    data = data[~data[text_column].isna()]
    data = data[data[text_column].str.strip() != ""]

    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' est absente dans {input_path}.")

    # Diagnostiquer les NaN avant le nettoyage
    nan_before = data[data[text_column].isna()]
    if not nan_before.empty:
        logger.warning(f"Lignes contenant NaN avant nettoyage :\n{nan_before}")

    # Appliquer les étapes de nettoyage
    logger.info("Début du nettoyage des tweets...")
    data[text_column] = data[text_column].apply(
        lambda tweet: lemmatize_and_remove_stopwords(preprocess_tweet(tweet))
    )

    # Diagnostiquer les NaN après le nettoyage
    nan_after_cleaning = data[data[text_column].isna()]
    if not nan_after_cleaning.empty:
        logger.warning(f"Lignes contenant NaN après nettoyage :\n{nan_after_cleaning}")

    # Supprimer les tweets `NaN` ou vides
    logger.info("Suppression des tweets NaN ou vides...")
    initial_size = len(data)
    data = data.dropna(subset=[text_column])
    data = data[data[text_column].str.strip() != ""]
    logger.info(f"{initial_size - len(data)} tweets vides ou NaN supprimés après nettoyage.")

    # Vérifier les NaN restants après suppression
    nan_remaining = data[data.isna().any(axis=1)]
    if not nan_remaining.empty:
        logger.warning(f"Lignes contenant des NaN après suppression :\n{nan_remaining}")

    # Filtrer les colonnes intéressantes (id et tweet)
    logger.info("Filtrage des colonnes pour ne conserver que 'id' et 'tweet'...")
    data = data[["id", text_column]]

    # Sauvegarder les données nettoyées
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"Données nettoyées sauvegardées dans : {output_path}")

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
        input_path = preprocess_params["input_file"]
        output_path = preprocess_params["output_file"]
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    preprocess_batch_cleaning(input_path, output_path)
