#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import re
import spacy
import logging
import psutil
from datetime import datetime

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Initialiser Spacy pour la lemmatisation
nlp = spacy.load("en_core_web_sm")

# Fonction de lemmatisation
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

# Fonction pour afficher les métriques système
def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(f"Memory Usage: {memory.percent}% (Total: {memory.total / (1024**3):.2f} GB, Available: {memory.available / (1024**3):.2f} GB)")

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Initialiser Hydra avec une nouvelle configuration
initialize(config_path="../notebooks/config", version_base=None)
cfg = compose(config_name="simple-model-ci-github")

# Afficher la configuration globale
logger.info("Configuration preprocess:")
logger.info(cfg.preprocess)

# Vérifier si le prétraitement est activé
if not cfg.preprocess.enabled:
    logger.info("Le prétraitement est désactivé. Fin du script.")
    exit()

# Afficher les métriques système
log_system_metrics()

# Charger le dataset
dataset_path = cfg.dataset.input.path  # Utiliser la clé correcte
logger.info(f"Chargement du dataset depuis {dataset_path}...")
df = pd.read_csv(dataset_path,
                 header=None,
                 names=["id", "timestamp", "date", "query", "user", "tweet"],
                 )
logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")
logger.info(f"Dataset columns: {df.columns.tolist()}")

# Supprimer les tweets vides (NaN ou chaînes vides)
df = df[~(df['tweet'].isna() | (df['tweet'].str.strip() == ""))]
logger.info(f"Suppression des tweets vides. {len(df)} tweets restants.")

# Nettoyer les tweets
logger.info("Nettoyage des tweets...")
df['tweet'] = df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))  # Supprimer les URLs
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))  # Supprimer les mentions
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#\w+', '', x))  # Supprimer les hashtags
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Supprimer la ponctuation
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x))  # Supprimer les espaces multiples
df['tweet'] = df['tweet'].apply(lambda x: x.strip())  # Supprimer les espaces en début et fin de chaîne
logger.info("Nettoyage terminé.")

# Vérifier si la lemmatisation est activée
if cfg.preprocess.lemmatization.enabled:
    logger.info("Lemmatisation activée. Appliquer la lemmatisation...")
    df['tweet'] = df['tweet'].apply(lemmatize_text)
    logger.info("Lemmatisation terminée.")
else:
    logger.info("Lemmatisation désactivée.")

# Sauvegarder le dataset nettoyé
output_path = cfg.preprocess.output.path
logger.info(f"Sauvegarde du dataset dans {output_path}...")
df.to_csv(output_path, index=False)
logger.info(f"Dataset sauvegardé dans {output_path}.")

# Afficher les métriques système à la fin du script
log_system_metrics()
