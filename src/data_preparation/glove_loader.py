import os
from gensim.models import KeyedVectors

_glove_model = None  # Cache pour le modèle GloVe

def load_glove_model(glove_path):
    """
    Charge le modèle GloVe dans une variable globale.

    Args:
        glove_path (str): Chemin vers le fichier GloVe.

    Returns:
        KeyedVectors: Modèle GloVe chargé.
    """
    global _glove_model
    if _glove_model is None:
        print(f"Chargement du modèle GloVe depuis : {glove_path}")
        _glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
    return _glove_model

