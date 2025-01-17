import os
import pickle
import pandas as pd
from tqdm import tqdm
from logging_utils import get_logger
from params_utils import load_params
from preprocess_tfidf import preprocess_batch_tfidf
import sys

# Obtenir le logger
logger = get_logger(__name__)

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
        partition_dir = params["input_dir"]
        processed_dir = params["output_dir"]
        vectorizer_path = params["vectorizer_file"]
        min_tfidf = params.get("min_tfidf", 0.1)
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    preprocess_batch_tfidf(partition_dir, processed_dir, vectorizer_path, min_tfidf=min_tfidf)
