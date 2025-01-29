import os
import sys
import pandas as pd
import pickle
import numpy as np
from glove_loader import load_glove_model
from tweet_vectorization import tweet_vectors_glove

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

def apply_glove_vectorizer(input_file, glove_model, output_file, text_column, label_column=None, embedding_dim=300):
    """
    Applique la vectorisation GloVe à un fichier d'entrée et sauvegarde les résultats.

    Args:
        input_file (str): Chemin vers le fichier CSV contenant les tweets.
        glove_model (KeyedVectors): Modèle GloVe chargé.
        output_file (str): Chemin pour sauvegarder le fichier vectorisé.
        text_column (str): Nom de la colonne contenant les textes.
        label_column (str, optional): Nom de la colonne contenant les labels ou IDs. Defaults to None.
        embedding_dim (int): Dimension des vecteurs GloVe.

    Returns:
        None
    """
    logger.info(f"Chargement des données depuis : {input_file}")
    data = pd.read_csv(input_file)

    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' est absente dans {input_file}.")

    logger.info("Application de la vectorisation GloVe...")
    vectors = np.array([
        tweet_vectors_glove(tweet, glove_model, embedding_dim) for tweet in data[text_column].fillna("")
    ])

    output_data = {"vectors": vectors}
    if label_column and label_column in data.columns:
        output_data["ids"] = data[label_column].tolist()

    logger.info(f"Sauvegarde des données vectorisées dans : {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)

    logger.info("Vectorisation GloVe terminée.")

# -------------------------
# Point d'entrée principal
# -------------------------
if __name__ == "__main__":
    # Charger les paramètres
    params_file = "params.yaml"
    section = "batch_prepare_glove_vectors"

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        raise

    # Chemins des fichiers
    glove_path = params["glove_file"]
    input_files = params["input"]
    output_dir = params["output_dir"]
    text_column = params["text_column"]
    label_column = params.get("label_column", None)
    embedding_dim = params.get("embedding_dim", 50)

    # Charger le modèle GloVe
    logger.info(f"Chargement du modèle GloVe depuis : {glove_path}")
    glove_model = load_glove_model(glove_path)

    # Appliquer le vectorizer à chaque fichier
    for split_name, input_file in input_files.items():
        logger.info(f"Traitement du fichier : {input_file}")
        output_file = os.path.join(output_dir, f"{split_name}_glove.pkl")
        apply_glove_vectorizer(input_file, glove_model, output_file, text_column, label_column, embedding_dim)
