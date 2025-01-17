import os
import pickle
import numpy as np
import pandas as pd
from logging_utils import get_logger
from params_utils import load_params

logger = get_logger(__name__)

def load_glove_embeddings(glove_file, embedding_dim):
    """
    Charge les vecteurs GloVe depuis un fichier.

    Args:
        glove_file (str): Chemin vers le fichier GloVe.
        embedding_dim (int): Dimension des vecteurs.

    Returns:
        dict: Dictionnaire {mot: vecteur GloVe}.
    """
    embeddings_index = {}
    logger.info(f"Chargement des vecteurs GloVe depuis : {glove_file}")
    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector
    logger.info(f"Nombre de vecteurs chargés : {len(embeddings_index)}")
    return embeddings_index

def create_embedding_matrix(vocab, embeddings_index, embedding_dim):
    """
    Crée une matrice d'embedding pour un vocabulaire donné.

    Args:
        vocab (dict): Dictionnaire {mot: index}.
        embeddings_index (dict): Dictionnaire GloVe {mot: vecteur}.
        embedding_dim (int): Dimension des vecteurs.

    Returns:
        np.array: Matrice d'embedding.
    """
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    for word, i in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector  # Utiliser le vecteur GloVe
        else:
            embedding_matrix[i] = np.random.uniform(-0.01, 0.01, embedding_dim)  # Vecteur aléatoire
    return embedding_matrix

def main():
    params_file = "params.yaml"
    section = "generate_glove_vectors"

    # Charger les paramètres
    params = load_params(params_file, section)
    glove_file = params["glove_file"]
    input_file = params["input_file"]
    output_file = params["output_file"]
    embedding_dim = params["embedding_dim"]
    vocab_size = params["vocab_size"]

    # Charger les tweets
    logger.info(f"Chargement des tweets depuis : {input_file}")
    data = pd.read_csv(input_file)
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(data["tweet"])
    vocab = tokenizer.word_index

    # Charger les vecteurs GloVe
    embeddings_index = load_glove_embeddings(glove_file, embedding_dim)

    # Créer la matrice d'embedding
    embedding_matrix = create_embedding_matrix(vocab, embeddings_index, embedding_dim)

    # Sauvegarder la matrice et le tokenizer
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump({"embedding_matrix": embedding_matrix, "tokenizer": tokenizer}, f)
    logger.info(f"Vecteurs GloVe et tokenizer sauvegardés dans : {output_file}")

if __name__ == "__main__":
    main()

