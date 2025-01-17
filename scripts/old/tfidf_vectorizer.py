import logging
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def handle_tfidf(partitioned_data, cfg):
    """Gestion du vectoriseur TF-IDF."""
    logger.info("Création du vectoriseur TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words=cfg.vectorizer.stopWords,
        max_features=cfg.vectorizer.maxFeatures,
        ngram_range=tuple(cfg.vectorizer.ngramRange),
    )

    logger.info("Application TF-IDF...")
    X_train = partitioned_data.data["X_train"]
    partitioned_data.data["X_train"] = vectorizer.fit_transform(X_train)

    for key in ["X_val", "X_test"]:
        if key in partitioned_data.data and partitioned_data.data[key] is not None:
            partitioned_data.data[key] = vectorizer.transform(partitioned_data.data[key])

    return partitioned_data, vectorizer
