import pandas as pd
import numpy as np
from vectorizer_loader import load_tfidf_vectorizer
from glove_loader import load_glove_model
from bert_loader import load_bert_tokenizer
import pickle

def tweet_vector_tfidf(tweet, vectorizer):
    """
    Applique la vectorisation TF-IDF sur un tweet unique.

    Args:
        tweet (str): Le texte du tweet à vectoriser.
        vectorizer (TfidfVectorizer): Instance du vectorizer TF-IDF chargé.

    Returns:
        sparse matrix: Le vecteur TF-IDF du tweet.
    """
    if not isinstance(tweet, str):
        tweet = ""
    return vectorizer.transform([tweet])

def tweet_vectors_glove(tweet, glove_model, embedding_dim=50):
    """
    Convertit un tweet en vecteur GloVe en moyennant les embeddings des mots.

    Args:
        tweet (str): Texte du tweet.
        glove_model (KeyedVectors): Modèle GloVe chargé.
        embedding_dim (int): Dimension des vecteurs d'embedding.

    Returns:
        numpy.ndarray: Vecteur GloVe représentant le tweet.
    """
    if not isinstance(tweet, str) or tweet.strip() == "":
        return np.zeros(embedding_dim)

    words = tweet.split()
    vectors = [glove_model[word] for word in words if word in glove_model]

    if not vectors:  # Si aucun mot n'est dans le modèle GloVe
        return np.zeros(embedding_dim)

    return np.mean(vectors, axis=0)

def tweet_token_bert(tweet, tokenizer, max_length=64):
    """
    Tokenise un tweet à l'aide du tokenizer BERT.

    Args:
        tweet (str): Texte du tweet à tokeniser.
        tokenizer (AutoTokenizer): Tokenizer BERT chargé.
        max_length (int): Longueur maximale des séquences.

    Returns:
        dict: Tokenisation avec les clés 'input_ids', 'attention_mask'.
    """
    tokens = tokenizer(
        tweet if isinstance(tweet, str) and tweet.strip() else "",
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors=None,  # Retourne sous forme de listes
    )
    return {
        "input_ids": tokens.get("input_ids", [0] * max_length),
        "attention_mask": tokens.get("attention_mask", [0] * max_length),
    }