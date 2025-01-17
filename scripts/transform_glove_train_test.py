import os
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from logging_utils import get_logger
from params_utils import load_params

logger = get_logger(__name__)

def load_glove_vectors(glove_file):
    """
    Charge le tokenizer et la matrice d'embedding GloVe depuis un fichier pickle.

    Args:
        glove_file (str): Chemin vers le fichier pickle contenant le tokenizer et la matrice.

    Returns:
        tuple: Tokenizer et matrice d'embedding.
    """
    logger.info(f"Chargement des vecteurs GloVe depuis : {glove_file}")
    with open(glove_file, "rb") as f:
        glove_data = pickle.load(f)
    return glove_data["tokenizer"], glove_data["embedding_matrix"]

def tokenize_and_pad_texts(tokenizer, texts, max_length):
    """
    Tokenise et applique du padding aux textes.

    Args:
        tokenizer (Tokenizer): Tokenizer à utiliser pour convertir les textes en séquences.
        texts (list): Liste de textes à tokeniser.
        max_length (int): Longueur maximale des séquences.

    Returns:
        numpy.ndarray: Séquences tokenisées et paddées.
    """
    logger.info("Tokenisation et padding des textes...")
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = tf.keras.utils.pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')
    return padded_sequences

def save_split_data(train_data, test_data, output_dir):
    """
    Sauvegarde les ensembles d'entraînement et de test.

    Args:
        train_data (tuple): Données d'entraînement (X_train, y_train).
        test_data (tuple): Données de test (X_test, y_test).
        output_dir (str): Répertoire de sortie pour les fichiers.
    """
    logger.info(f"Sauvegarde des ensembles d'entraînement et de test dans : {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.pkl"), "wb") as f:
        pickle.dump({"X": train_data[0], "y": train_data[1]}, f)
    with open(os.path.join(output_dir, "test.pkl"), "wb") as f:
        pickle.dump({"X": test_data[0], "y": test_data[1]}, f)

def main():
    # Charger les paramètres
    params_file = "params.yaml"
    section = "transform_glove_train_test"
    params = load_params(params_file, section)

    glove_file = params["input_file"]
    output_dir = params["output_dir"]
    text_column = params["text_column"]
    label_column = params["label_column"]
    test_size = params["test_size"]
    random_state = params["random_state"]

    # Charger le tokenizer et la matrice GloVe
    tokenizer, _ = load_glove_vectors(glove_file)

    # Charger les données
    input_data_file = params["data_file"]
    logger.info(f"Chargement des données depuis : {input_data_file}")
    data = pd.read_csv(input_data_file)
    texts = data[text_column].fillna("").tolist()
    labels = data[label_column].tolist()

    # Tokenisation et padding
    logger.info("Préparation des données avec GloVe...")
    max_length = params.get("max_length", 100)
    X = tokenize_and_pad_texts(tokenizer, texts, max_length)

    # Découpage en train/test
    logger.info("Découpage des données en ensembles d'entraînement et de test...")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=test_size, random_state=random_state)

    # Sauvegarde
    save_split_data((X_train, y_train), (X_test, y_test), output_dir)
    logger.info("Traitement terminé.")

if __name__ == "__main__":
    main()
