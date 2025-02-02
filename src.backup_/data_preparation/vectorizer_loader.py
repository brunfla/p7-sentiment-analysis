import pickle

_vectorizer_cache = None  # Cache pour le vectorizer TF-IDF

def load_tfidf_vectorizer(vectorizer_path):
    """
    Charge le vectorizer TF-IDF depuis un fichier pickle.

    Args:
        vectorizer_path (str): Chemin vers le fichier pickle contenant le vectorizer.

    Returns:
        TfidfVectorizer: Instance du vectorizer TF-IDF chargé.
    """
    global _vectorizer_cache
    if _vectorizer_cache is None:
        print(f"Chargement du vectorizer TF-IDF depuis : {vectorizer_path}")
        with open(vectorizer_path, "rb") as f:
            _vectorizer_cache = pickle.load(f)
    return _vectorizer_cache

