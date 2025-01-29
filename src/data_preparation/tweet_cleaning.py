import os
import re
import pandas as pd
from gensim.models import KeyedVectors  # Pour charger le modèle GloVe
from nltk.corpus import stopwords       # Pour les stopwords
from nltk.stem import WordNetLemmatizer  # Pour la lemmatisation
import nltk
from glove_loader import load_glove_model  # Import du module de chargement

# Assurez-vous que les ressources NLTK nécessaires sont disponibles
nltk.download('stopwords')
nltk.download('wordnet')

def is_valid_tweet(tweet):
    """
    Vérifie si un tweet est valide ou non.

    Args:
        tweet (str): Le texte du tweet à valider.

    Returns:
        bool: True si le tweet est valide, False sinon.
    """
    if pd.isna(tweet):
        return False
    if not isinstance(tweet, str):
        return False
    if tweet.strip() == "":
        return False
    return True

def validate_and_clean(data_x, data_y):
    """
    Valide et nettoie les tweets, tout en synchronisant les fichiers x (features) et y (labels).

    Args:
        data_x (pd.DataFrame): Données features contenant les colonnes 'id' et 'feature'.
        data_y (pd.DataFrame): Données labels contenant les colonnes 'id' et 'label'.

    Returns:
        pd.DataFrame, pd.DataFrame: Données nettoyées pour x et y.
    """
    print("Validation et nettoyage des tweets...")

    # Appliquer la fonction de validation
    initial_size = len(data_x)
    valid_data = data_x[data_x["feature"].apply(is_valid_tweet)]
    cleaned_size = len(valid_data)

    print(f"{initial_size - cleaned_size} tweets invalides supprimés.")

    # Synchroniser avec les labels
    valid_ids = valid_data["id"].unique()
    data_y = data_y[data_y["id"].isin(valid_ids)]

    print(f"Synchronisation effectuée : {len(data_y)} entrées restantes dans les labels.")

    return valid_data, data_y

def clean_tweet(tweet):
    """Nettoyage de base d'un tweet."""
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

def clean_tweet_with_glove(tweet, glove_path, threshold=0.6):
    """
    Nettoyage final avec GloVe :
    - Supprime les mots non valides.
    - Retourne une erreur si le tweet nettoyé est vide.

    Args:
        tweet (str): Texte du tweet.
        glove_path (str): Chemin vers le fichier du modèle GloVe.
        threshold (float): Seuil pour les corrections de mots avec GloVe.

    Returns:
        str: Tweet nettoyé.

    Raises:
        ValueError: Si le tweet est vide ou si le résultat nettoyé est vide.
    """
    if not isinstance(tweet, str) or tweet.strip() == "":
        raise ValueError("Le tweet fourni est vide ou invalide.")

    # Charger le modèle GloVe
    glove_model = load_glove_model(glove_path)

    def is_valid_word(word):
        """Vérifie si le mot est valide dans le vocabulaire GloVe."""
        return word in glove_model

    def get_similar_word(word):
        """Corrige les mots invalides en cherchant des mots similaires."""
        try:
            similar_words = glove_model.most_similar(word, topn=1)
            if similar_words and similar_words[0][1] > threshold:
                return similar_words[0][0]  # Retourne le mot corrigé
        except KeyError:
            return None
        return None

    words = tweet.split()
    cleaned_words = []

    for word in words:
        if is_valid_word(word):
            cleaned_words.append(word)  # Ajouter le mot valide
        else:
            corrected_word = get_similar_word(word)
            if corrected_word:
                cleaned_words.append(corrected_word)  # Ajouter le mot corrigé

    # Vérifier si le tweet nettoyé est vide
    if not cleaned_words:
        return ""
        #raise ValueError("Le tweet nettoyé est vide après traitement.")
    return " ".join(cleaned_words)

