#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import psutil
import re
import pandas as pd
from collections import Counter
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import matplotlib.pyplot as plt

# ------------------------------------------------
# Configuration du logging
# ------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Fonctions utilitaires
# ------------------------------------------------
def log_system_metrics():
    """
    Affiche l'usage CPU et mémoire pour un monitoring de base.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

def pre_clean(tweet):
    """
    Nettoyage initial :
    - Suppression des URLs, mentions, hashtags, ponctuation, espaces multiples.
    """
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+', '', tweet)  # Enlever URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Enlever mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Enlever hashtags
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Enlever ponctuation
    return re.sub(r'\s+', ' ', tweet).strip()  # Enlever espaces multiples

def lemmatize_and_filter(tweet, lemmatizer, stop_words):
    """
    Lemmatisation et suppression des stopwords.
    """
    words = tweet.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

def tfidf_filter(tweets, min_tfidf=0.1):
    """
    Filtrage des mots avec un faible score TF-IDF.
    """
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(tweets)
    feature_names = np.array(tfidf.get_feature_names_out())

    filtered_tweets = []
    for row in tfidf_matrix:
        words_with_scores = zip(feature_names, row.toarray()[0])
        important_words = [word for word, score in words_with_scores if score > min_tfidf]
        filtered_tweets.append(" ".join(important_words))
    return filtered_tweets

def clean_with_glove(tweet, glove_model, threshold=0.6):
    """
    Nettoyage final avec GloVe :
    - Supprime les mots non valides.
    - Corrige les fautes d'orthographe.
    """
    def is_valid_word(word):
        if word in glove_model:
            return word  # Mot valide
        similar_words = glove_model.most_similar(word, topn=1) if word else []
        if similar_words and similar_words[0][1] > threshold:
            return similar_words[0][0]  # Mot corrigé
        return None

    words = tweet.split()
    cleaned_words = [is_valid_word(word) for word in words if is_valid_word(word)]
    return " ".join(cleaned_words)

def analyze_vocabulary(tweets, output_dir):
    """
    Analyse et visualisation du vocabulaire :
    - Affiche la taille du vocabulaire.
    - Enregistre la distribution des fréquences dans un PNG.
    """
    logger.info("Analyse du vocabulaire...")
    all_words = " ".join(tweets).split()
    word_counts = Counter(all_words)

    # Taille du vocabulaire
    vocab_size = len(word_counts)
    logger.info(f"Taille du vocabulaire unique : {vocab_size}")

    # Distribution des fréquences
    frequencies = [count for word, count in word_counts.items()]
    frequencies.sort(reverse=True)

    # Enregistrement de la courbe de distribution
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies)
    plt.title("Distribution des fréquences des mots")
    plt.xlabel("Rang des mots (trié par fréquence)")
    plt.ylabel("Fréquence")
    plt.grid()
    output_file = os.path.join(output_dir, "word_frequency_distribution.png")
    plt.savefig(output_file)
    logger.info(f"Courbe de distribution sauvegardée dans {output_file}")

    return vocab_size

def analyze_tweet_lengths(tweets):
    """
    Analyse les longueurs des tweets :
    - Longueur minimale, maximale, moyenne.
    """
    lengths = [len(tweet.split()) for tweet in tweets]
    min_length = min(lengths)
    max_length = max(lengths)
    mean_length = np.mean(lengths)

    logger.info(f"Longueur minimale des tweets : {min_length} mots")
    logger.info(f"Longueur maximale des tweets : {max_length} mots")
    logger.info(f"Longueur moyenne des tweets : {mean_length:.2f} mots")

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    # 1) Afficher les métriques système
    log_system_metrics()

    # 2) Récupérer la config Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    # 3) Charger la configuration
    data_path = cfg.cleaner.input
    output_path = cfg.cleaner.output

    # 4) Charger le dataset
    logger.info(f"Chargement du dataset depuis {data_path}...")
    df = pd.read_csv(data_path, header=None, names=["id", "timestamp", "date", "query", "user", "tweet"])
    logger.info(f"Dataset chargé avec {len(df)} lignes.")

    # Transformer les classes 4 -> 1
    df['id'] = df['id'].apply(lambda x: 1 if x == 4 else x)

    # 5) Nettoyage des tweets
    logger.info("Nettoyage des tweets...")
    df['tweet'] = df['tweet'].apply(pre_clean)

    # Suppression des tweets vides
    df = df[~(df['tweet'].isna() | (df['tweet'].str.strip() == ""))]

    # Initialisation des outils de lemmatisation et stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    df['tweet'] = df['tweet'].apply(lambda x: lemmatize_and_filter(x, lemmatizer, stop_words))

    # Première passe de TF-IDF
    logger.info("Première passe de TF-IDF...")
    df['tweet'] = tfidf_filter(df['tweet'])

    # Chargement de GloVe
    logger.info("Chargement du modèle GloVe...")
    glove_file = cfg.cleaner.glove_path
    glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)

    # Nettoyage final avec GloVe
    logger.info("Nettoyage final avec GloVe...")
    df['tweet'] = df['tweet'].apply(lambda x: clean_with_glove(x, glove_model))

    # Deuxième passe de TF-IDF
    logger.info("Deuxième passe de TF-IDF...")
    df['tweet'] = tfidf_filter(df['tweet'])

    # Analyse du vocabulaire
    output_dir = os.path.dirname(output_path)
    vocab_size = analyze_vocabulary(df['tweet'], output_dir)

    # Analyse des longueurs des tweets
    analyze_tweet_lengths(df['tweet'])

    # 6) Sauvegarde
    logger.info(f"Sauvegarde du dataset nettoyé dans {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info("Dataset sauvegardé avec succès.")

    # 7) Afficher les métriques système
    log_system_metrics()

if __name__ == "__main__":
    main()
