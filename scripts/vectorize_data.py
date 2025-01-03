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

# Récupérer le chemin de configuration et la stratégie depuis les variables d'environnement
config_path = os.getenv('HYDRA_CONFIG_PATH', '../notebooks/config')
strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

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

    # 1) Déterminer le chemin du dataset (CSV nettoyé)
    dataset_path = cfg.cleaner.output if cfg.vectorizer.enabled else cfg.dataset.input.path
    logger.info(f"Chargement du dataset depuis {dataset_path}...")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")

    # 2) Créer ou charger le vectorizer
    if cfg.vectorizer.existingVectorizer and os.path.exists(cfg.vectorizer.outputPath):
        with open(cfg.vectorizer.outputPath, "rb") as f:
            vectorizer = pickle.load(f)
        logger.info(f"Vectoriseur chargé depuis {cfg.vectorizer.outputPath}.")
    else:
        logger.info("Création d'un nouveau vectoriseur TF-IDF...")
        vectorizer = TfidfVectorizer(
            stop_words=cfg.vectorizer.stopWords,
            max_features=cfg.vectorizer.maxFeatures,
            ngram_range=tuple(cfg.vectorizer.ngramRange),
        )

    # 3) Soit on charge des données vectorisées existantes, soit on (re)vectorise
    if cfg.vectorizer.existingData and os.path.exists(cfg.vectorizer.outputData):
        logger.info(f"Chargement des données vectorisées existantes depuis {cfg.vectorizer.outputData}...")
        with open(cfg.vectorizer.outputData, "rb") as f:
            X, y = pickle.load(f)  # On suppose qu'on y a déjà stocké (X, y)
        logger.info("Données vectorisées déjà présentes. Aucun fit_transform n'est refait.")
    else:
        logger.info("Appliquer TF-IDF sur les tweets...")
        X = vectorizer.fit_transform(df['tweet'])
        # *** Calcul de y si vous voulez le transmettre au pipeline d'entraînement
        y = df['id'].apply(lambda x: 1 if x == 4 else 0)

        # Sauvegarder (X, y) vectorisés
        with open(cfg.vectorizer.outputData, "wb") as f:
            pickle.dump((X, y), f)
        logger.info(f"Données vectorisées sauvegardées dans {cfg.vectorizer.outputData}.")

    # 4) Sauvegarder le vectoriseur (si nouveau)
    if not cfg.vectorizer.existingVectorizer or not os.path.exists(cfg.vectorizer.outputPath):
        with open(cfg.vectorizer.outputPath, "wb") as f:
            pickle.dump(vectorizer, f)
        logger.info(f"Vectoriseur sauvegardé dans {cfg.vectorizer.outputPath}.")

def handle_default():
    logger.warning(f"Target '{cfg.vectorizer._target_}' non pris en charge. Fin du script.")
    sys.exit(0)

# Si on veut éviter de re-vectoriser quand existingData est True et le fichier existe déjà
if cfg.vectorizer.existingData and os.path.exists(cfg.vectorizer.outputData):
    logger.info(
        f"Les données vectorisées existent déjà ({cfg.vectorizer.outputData}). "
        "Traitement bypassé (pas de re-vectorisation)."
    )
    sys.exit(0)

targets = {
    "tfidfVectorizer": handle_tfidf_vectorizer,
}

# Appeler la fonction correspondant au target, sinon handle_default()
targets.get(cfg.vectorizer._target_, handle_default)()

# Afficher les métriques système à la fin du script
log_system_metrics()