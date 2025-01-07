#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import logging
import psutil
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sklearn.feature_extraction.text import TfidfVectorizer
from load_split_data import load_partitioned_data
import pickle

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def log_system_metrics():
    """Affiche la consommation CPU/RAM"""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(f"Memory Usage: {memory.percent}% "
                f"(Total: {memory.total / (1024**3):.2f} GB, "
                f"Available: {memory.available / (1024**3):.2f} GB)")

# Afficher les métriques système au démarrage
log_system_metrics()

# ------------------------------------------------
# Fonction pour afficher les métriques système
# ------------------------------------------------
# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Récupérer le chemin de configuration et la stratégie depuis les variables d'environnement
config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
strategy = os.getenv('HYDRA_STRATEGY', 'baseline')
# Initialiser Hydra avec la stratégie choisie
initialize(config_path=config_path, version_base=None)
cfg = compose(config_name=strategy)

# Afficher la stratégie utilisée
print(f"Stratégie sélectionnée : {strategy}")

# Afficher la configuration globale
logger.info("Configuration vectorizer:")
logger.info(cfg.vectorizer)


def handle_tfidf_vectorizer():
    """Gère la vectorisation TF-IDF du dataset en fonction de la config Hydra."""
    logger.info("Target 'tfidfVectorizer' détecté. Traitement en cours...")

    dataset_path = cfg.vectorizer.input
    logger.info(f"Chargement des données depuis {dataset_path}...")

    data = load_partitioned_data(dataset_path, cfg.partitioner)

    # Gestion des différents types de partitionnement
    if cfg.partitioner._target_ == "trainValTest":
        X_train, y_train, _, _, _, _ = data
    elif cfg.partitioner._target_ == "trainTest":
        X_train, y_train, _, _, _, _ = data
    elif cfg.partitioner._target_ == "crossValidation":
        _, X_train, y_train = data
    else:
        raise ValueError(f"Partition de découpage non reconnue: {cfg.partitioner._target_}")

    # Créer ou charger le vectorizer
    logger.info("Création d'un nouveau vectoriseur TF-IDF...")
    vectorizer = TfidfVectorizer(
        stop_words=cfg.vectorizer.stopWords,
        max_features=cfg.vectorizer.maxFeatures,
        ngram_range=tuple(cfg.vectorizer.ngramRange),
    )

    logger.info("Appliquer TF-IDF sur les données d'entraînement...")
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Sauvegarder (X, y) vectorisés
    with open(cfg.vectorizer.outputData, "wb") as f:
        pickle.dump((X_train_vectorized, y_train), f)
    logger.info(f"Données vectorisées sauvegardées dans {cfg.vectorizer.outputData}.")

    # Sauvegarder le modèle du vectoriseur
    with open(cfg.vectorizer.outputPath, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectoriseur sauvegardé dans {cfg.vectorizer.outputPath}.")

def handle_default():
    logger.warning(f"Target '{cfg.vectorizer._target_}' non pris en charge. Fin du script.")
    sys.exit(0)

targets = {
    "tfidfVectorizer": handle_tfidf_vectorizer,
}

# Appeler la fonction correspondant au target, sinon handle_default()
targets.get(cfg.vectorizer._target_, handle_default)()

# Afficher les métriques système à la fin du script
log_system_metrics()
