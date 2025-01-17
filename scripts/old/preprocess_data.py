import os
import logging
import psutil
import pandas as pd
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# -------------------------
# Configurer le logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Fonction pour afficher les métriques système
# ------------------------------------------------
def log_system_metrics():
    """
    Affiche les métriques CPU et mémoire pour le monitoring.
    """
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

# ------------------------------------------------
# Fonction de prétraitement unifiée
# ------------------------------------------------
def preprocess_tweet(tweet, glove_model, threshold=0.6):
    """
    Nettoyage complet d'un tweet :
    - Nettoyage initial : suppression des URLs, mentions, hashtags, ponctuation, espaces multiples.
    - Lemmatisation et suppression des stopwords.
    - Nettoyage final avec GloVe : suppression ou correction des mots invalides.

    Args:
        tweet (str): Tweet brut.
        glove_model (KeyedVectors): Modèle GloVe chargé.
        threshold (float): Seuil pour la similarité cosinus lors de la correction des mots.

    Returns:
        str: Tweet nettoyé.
    """
    if not isinstance(tweet, str) or tweet.strip() == "":
        return ""

    # Nettoyage initial
    tweet = tweet.lower()
    tweet = re.sub(r'http\S+|www\S+', '', tweet)  # Supprimer les URLs
    tweet = re.sub(r'@\w+', '', tweet)  # Supprimer les mentions
    tweet = re.sub(r'#\w+', '', tweet)  # Supprimer les hashtags
    tweet = re.sub(r'[^\w\s]', '', tweet)  # Supprimer la ponctuation
    tweet = re.sub(r'\s+', ' ', tweet).strip()  # Supprimer les espaces multiples

    # Lemmatisation et suppression des stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in tweet.split() if word not in stop_words]

    # Nettoyage avec GloVe
    def is_valid_word(word):
        if word in glove_model:
            return word
        similar_words = glove_model.most_similar(word, topn=1) if word else []
        if similar_words and similar_words[0][1] > threshold:
            return similar_words[0][0]
        return None

    cleaned_words = [is_valid_word(word) for word in words if is_valid_word(word)]

    return " ".join(cleaned_words)

# ------------------------------------------------
# Prétraitement des partitions
# ------------------------------------------------
def preprocess_partition(partition_path, output_path, glove_model, text_column):
    """
    Prétraite une partition donnée.

    Args:
        partition_path (str): Chemin vers le fichier CSV de la partition.
        output_path (str): Chemin de sauvegarde pour la partition prétraitée.
        glove_model (KeyedVectors): Modèle GloVe chargé.
        text_column (str): Nom de la colonne contenant les tweets.

    Returns:
        None
    """
    logger.info(f"Prétraitement de la partition : {partition_path}")

    # Charger la partition
    data = pd.read_csv(partition_path)

    # Vérifier la colonne de texte
    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' n'est pas présente dans la partition.")

    # Prétraiter les tweets
    logger.info(f"Prétraitement des tweets dans la partition...")
    data[text_column] = data[text_column].apply(lambda x: preprocess_tweet(x, glove_model))

    # Supprimer les tweets vides après prétraitement
    initial_size = len(data)
    data = data[~data[text_column].isna()]
    data = data[data[text_column].str.strip() != ""]
    logger.info(f"{initial_size - len(data)} tweets vides supprimés après prétraitement.")

    # Sauvegarder la partition prétraitée
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data.to_csv(output_path, index=False)
    logger.info(f"Partition prétraitée sauvegardée dans : {output_path}")

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    # Afficher les métriques système au démarrage
    log_system_metrics()

    # Charger la configuration avec Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    # Récupérer les paramètres de configuration
    partitions_dir = cfg.preprocess.partitioner.output
    processed_dir = cfg.preprocess.cleaner.output
    text_column = cfg.dataset.text_column
    glove_path = cfg.preprocess.cleaner.glove_path
    glove_threshold = cfg.preprocess.cleaner.glove_threshold

    # Charger le modèle GloVe
    logger.info("Chargement du modèle GloVe...")
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    # Identifier les partitions à prétraiter
    logger.info(f"Recherche des partitions dans : {partitions_dir}")
    for partition_file in os.listdir(partitions_dir):
        partition_path = os.path.join(partitions_dir, partition_file)
        processed_path = os.path.join(processed_dir, partition_file)

        # Prétraiter la partition
        preprocess_partition(partition_path, processed_path, glove_model, text_column)

    logger.info("Prétraitement terminé pour toutes les partitions.")
    log_system_metrics()

if __name__ == "__main__":
    main()
