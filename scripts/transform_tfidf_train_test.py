import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.sparse import vstack, save_npz
from logging_utils import get_logger
from params_utils import load_params
import sys

logger = get_logger(__name__)

def preprocess_with_tfidf(tweet, vectorizer):
    """
    Applique le vectoriseur TF-IDF à un tweet spécifique.

    Args:
        tweet (str): Tweet à transformer.
        vectorizer (TfidfVectorizer): Vectoriseur TF-IDF.

    Returns:
        scipy.sparse.csr_matrix: Matrice TF-IDF du tweet.
    """
    return vectorizer.transform([tweet])

def preprocess_batch_tfidf(input_dir, processed_dir, vectorizer_path, labels_dir, text_column="tweet", label_column="id"):
    """
    Applique le vectoriseur TF-IDF sur chaque fichier et génère des matrices sparse et des fichiers de labels.

    Args:
        input_dir (str): Répertoire contenant les partitions à traiter.
        processed_dir (str): Répertoire de sortie pour les fichiers traités.
        vectorizer_path (str): Chemin vers le fichier du vectoriseur TF-IDF.
        labels_dir (str): Répertoire pour les fichiers de labels.
        text_column (str): Nom de la colonne contenant les tweets.
        label_column (str): Nom de la colonne contenant les labels.
    """
    logger.info(f"Chargement du vectoriseur TF-IDF depuis : {vectorizer_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Le vectoriseur TF-IDF n'existe pas : {vectorizer_path}")

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    logger.info(f"Chargement des fichiers CSV depuis : {input_dir}")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Le répertoire des partitions n'existe pas : {input_dir}")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"Aucun fichier CSV trouvé dans le répertoire des partitions : {input_dir}")

    for file_name in tqdm(files, desc="Traitement des fichiers", unit="fichier"):
        partition_path = os.path.join(input_dir, file_name)
        output_vectors_path = os.path.join(processed_dir, file_name.replace('.csv', '.npz'))
        output_labels_path = os.path.join(labels_dir, file_name.replace('.csv', '_labels.csv'))

        try:
            # Charger la partition
            data = pd.read_csv(partition_path)

            # Vérification stricte des colonnes requises
            if text_column not in data.columns:
                raise ValueError(f"La colonne '{text_column}' est absente dans {partition_path}.")
            if label_column not in data.columns:
                raise ValueError(f"La colonne '{label_column}' est absente dans {partition_path}.")

            data = data.dropna(subset=[text_column, label_column])
            data = data[data[text_column].str.strip() != ""]
            if data.empty:
                raise ValueError(f"Partition vide après validation : {partition_path}.")

            tweets = data[text_column].tolist()
            labels = data[label_column].tolist()

            # Appliquer TF-IDF
            logger.info(f"Application du vectoriseur TF-IDF sur {file_name}...")
            vectors = vectorizer.transform(tweets)

            # Sauvegarder les vecteurs TF-IDF dans un fichier sparse
            save_npz(output_vectors_path, vectors)
            logger.info(f"Matrice TF-IDF sauvegardée dans : {output_vectors_path}")

            # Sauvegarder les labels dans un fichier CSV
            pd.DataFrame({label_column: labels}).to_csv(output_labels_path, index=False)
            logger.info(f"Labels sauvegardés dans : {output_labels_path}")

        except Exception as e:
            logger.error(f"Erreur lors du traitement de {file_name} : {e}")
            sys.exit(1)

if __name__ == "__main__":
    params_file = "params.yaml"

    section = os.path.splitext(os.path.basename(__file__))[0]

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        input_dir = params["input_dir"]
        processed_dir = params["output_dir"]
        vectorizer_path = params["vectorizer_file"]
        labels_dir = params["output_labels_dir"]
        text_column = params.get("text_column", "tweet")
        label_column = params.get("label_column", "id")
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    preprocess_batch_tfidf(input_dir, processed_dir, vectorizer_path, labels_dir, text_column, label_column)
