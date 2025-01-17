import os
import pandas as pd
from gensim.models import KeyedVectors
import sys
from params_utils import load_params
from logging_utils import get_logger

# Obtenir le logger
logger = get_logger(__name__)

# -------------------------
# Fonction de nettoyage avec GloVe
# -------------------------
def clean_with_glove(tweet, glove_model, threshold=0.6):
    """
    Nettoyage final avec GloVe :
    - Supprime les mots non valides.
    - Corrige les fautes d'orthographe si possible.
    """
    def is_valid_word(word):
        if word in glove_model:
            return word  # Mot valide
        return None  # Mot invalide

    def get_similar_word(word):
        # Tente de corriger les mots invalides
        try:
            similar_words = glove_model.most_similar(word, topn=1)
            if similar_words and similar_words[0][1] > threshold:
                return similar_words[0][0]  # Retourner le mot corrigé
        except KeyError:
            pass
        return None

    if not isinstance(tweet, str) or tweet.strip() == "":
        return ""

    words = tweet.split()
    cleaned_words = [
        is_valid_word(word) if is_valid_word(word) else get_similar_word(word)
        for word in words
    ]
    return " ".join([word for word in cleaned_words if word is not None])

# -------------------------
# Fonction principale
# -------------------------
def preprocess_with_glove(input_path, output_path, glove_path, threshold=0.6):
    """
    Applique le nettoyage avec GloVe sur le dataset.

    Args:
        input_path (str): Chemin vers le fichier d'entrée nettoyé de base.
        output_path (str): Chemin pour sauvegarder le fichier nettoyé avec GloVe.
        glove_path (str): Chemin vers le fichier GloVe.
        threshold (float): Seuil pour la similarité cosinus.

    Returns:
        None
    """
    logger.info(f"Chargement du modèle GloVe depuis : {glove_path}")
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    logger.info(f"Chargement des données depuis : {input_path}")
    data = pd.read_csv(input_path)
    text_column = "tweet"

    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' est absente dans {input_path}.")

    logger.info("Nettoyage des tweets avec GloVe...")
    data[text_column] = data[text_column].apply(
        lambda tweet: clean_with_glove(tweet, glove_model, threshold)
    )

    # Supprimer les tweets vides ou NaN après nettoyage
    initial_size = len(data)
    data = data.dropna(subset=[text_column])  # Supprime les NaN dans la colonne tweet
    data = data[data[text_column].str.strip() != ""]  # Supprime les tweets vides
    logger.info(f"{initial_size - len(data)} tweets supprimés après le nettoyage avec GloVe.")

    # Filtrer les colonnes intéressantes (id et tweet)
    logger.info("Filtrage des colonnes pour ne conserver que 'id' et 'tweet'...")
    data = data[["id", text_column]]

    # Sauvegarder le fichier nettoyé
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"Fichier nettoyé avec GloVe sauvegardé dans : {output_path}")

# -------------------------
# Point d'entrée principal
# -------------------------
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
        glove_path = preprocess_params["glove_file"]
        threshold = preprocess_params.get("glove_similarity_threshold", 0.6)
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    logger.info("Démarrage du nettoyage avec GloVe...")
    preprocess_with_glove(input_path, output_path, glove_path, threshold)
    logger.info("Processus terminé.")
