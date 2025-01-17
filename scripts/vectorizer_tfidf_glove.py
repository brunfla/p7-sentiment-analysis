import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from logging_utils import get_logger
from params_utils import load_params
from vectorizer_tfidf import generate_tfidf_vectorizer 

# Obtenir le logger
logger = get_logger(__name__)


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
    generate_tfidf_vectorizer(input_path, output_path, max_features=max_features)
