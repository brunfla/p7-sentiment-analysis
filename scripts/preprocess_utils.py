import os
import pandas as pd
import re
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dvc_utils import get_stage_dependencies

from logging_utils import get_logger

def preprocess_tweet(tweet):
    """Nettoyage de base d'un tweet."""
    if not isinstance(tweet, str) or tweet.strip() == "":
        return ""

    tweet = tweet.lower()
    tweet = re.sub(r"http\S+|www\S+", "", tweet)  # Supprimer les URLs
    tweet = re.sub(r"@\w+", "", tweet)  # Supprimer les mentions
    tweet = re.sub(r"#\w+", "", tweet)  # Supprimer les hashtags
    tweet = re.sub(r"[^\w\s]", "", tweet)  # Supprimer la ponctuation
    tweet = re.sub(r"\s+", " ", tweet).strip()  # Supprimer les espaces multiples
    return tweet

def lemmatize_and_remove_stopwords(tweet):
    """Lemmatisation et suppression des stopwords."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    words = [
        lemmatizer.lemmatize(word)
        for word in tweet.split()
        if word not in stop_words
    ]
    return " ".join(words)

def validate_corpus(corpus, error_message="Le corpus est vide après prétraitement."):
    logger = get_logger()
    """Valide qu'un corpus n'est pas vide."""
    if not corpus or all(doc.strip() == "" for doc in corpus):
        logger.warning(error_message)
        return False
    return True


# ------------------------------------------------
# Fonction commune de prétraitement d'une partition
# ------------------------------------------------
def preprocess_partition(partition_path, output_path, preprocess_fn, **kwargs):
    """
    Applique une fonction de prétraitement sur une partition.

    Args:
        partition_path (str): Chemin vers la partition CSV d'entrée.
        output_path (str): Chemin de sauvegarde pour la partition prétraitée.
        preprocess_fn (function): Fonction de prétraitement à appliquer.
        kwargs: Arguments supplémentaires pour la fonction de prétraitement.

    Returns:
        None
    """
    logger = get_logger()
    logger.info(f"Prétraitement de la partition : {partition_path}")
    data = pd.read_csv(partition_path)

    # Vérifier si la colonne texte existe
    text_column = kwargs.get("text_column", "tweet")
    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' n'existe pas dans {partition_path}")

    # Appliquer la fonction de prétraitement
    data[text_column] = data[text_column].apply(preprocess_fn)

    # Supprimer les tweets vides
    initial_size = len(data)
    data = data[~data[text_column].isna()]
    data = data[data[text_column].str.strip() != ""]
    logger.info(f"{initial_size - len(data)} tweets vides supprimés après prétraitement.")

    # Sauvegarder
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"Partition prétraitée sauvegardée : {output_path}")