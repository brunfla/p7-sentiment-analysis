import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
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

def generate_tfidf_vectorizer(input_path, output_path, text_column="tweet", max_features=10000):
    """
    Génère un vectoriseur TF-IDF basé sur le corpus donné et le sauvegarde.

    Args:
        input_path (str): Chemin vers le fichier CSV contenant les données.
        output_path (str): Chemin pour sauvegarder le vectoriseur TF-IDF.
        text_column (str): Nom de la colonne contenant les tweets.
        max_features (int): Nombre maximum de caractéristiques dans le vectoriseur.
    """
    logger.info(f"Chargement des données depuis : {input_path}")
    data = pd.read_csv(input_path)

    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' est absente dans le fichier {input_path}.")

    # Charger les tweets pour le TF-IDF
    corpus = [str(tweet) for tweet in data[text_column].dropna().tolist()]
    if not corpus:
        logger.error("Le corpus est vide après nettoyage. Impossible de générer le vectoriseur TF-IDF.")
        raise ValueError("Corpus vide")

    logger.info(f"Initialisation du vectoriseur TF-IDF avec max_features={max_features}...")
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        vectorizer.fit(corpus)
        logger.info(f"Nombre réel de caractéristiques générées : {len(vectorizer.get_feature_names_out())}")
    except Exception as e:
        logger.error(f"Erreur lors de la génération du vectoriseur TF-IDF : {e}")
        raise

    # Sauvegarder le vectoriseur
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Vectoriseur TF-IDF sauvegardé dans : {output_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du vectoriseur : {e}")
        raise

if __name__ == "__main__":
    # Charger les paramètres depuis params.yaml
    params_file = "params.yaml"

    section = os.path.splitext(os.path.basename(__file__))[0]  # Nom du script comme section

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        input_path = params["input_file"]
        output_path = params["output_file"]
        max_features = params["max_features"]
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    # Appeler la fonction pour générer le vectoriseur TF-IDF
    try:
        generate_tfidf_vectorizer(input_path, output_path, max_features=max_features)
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution du script : {e}")
        sys.exit(1)
