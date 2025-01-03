#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import re
import spacy
import logging
import psutil
from datetime import datetime
import os

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Fonction pour afficher les métriques système
def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(f"Memory Usage: {memory.percent}% (Total: {memory.total / (1024**3):.2f} GB, Available: {memory.available / (1024**3):.2f} GB)")

# Afficher les métriques système
log_system_metrics()

#### --- HYDRA --- ####
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


#### --- MAIN --- ####
# Vérifier si le prétraitement est activé
if not cfg.cleaner.enabled:
    logger.info("Le traitement est désactivé. Fin du script.")
    exit()
    
# Afficher la configuration globale
logger.info("Configuration preprocess:")
logger.info(cfg.cleaner)


# Gérer le cas où existingData est true
output_path = cfg.cleaner.output
if cfg.cleaner.existingData:
    if os.path.exists(output_path):
        logger.info(f"Le fichier nettoyé existe déjà ({output_path}). Traitement bypassé.")
        exit()
    else:
        logger.warning(f"Le paramètre existingData est activé, mais le fichier {output_path} est introuvable. Le traitement sera effectué.")

# Charger le dataset
dataset_path = cfg.dataset.path  # Utiliser la clé correcte
logger.info(f"Chargement du dataset depuis {dataset_path}...")
df = pd.read_csv(dataset_path,
                 header=None,
                 names=["id", "timestamp", "date", "query", "user", "tweet"],
                 )
logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")
logger.info(f"Dataset columns: {df.columns.tolist()}")

# Nettoyer les tweets
logger.info("Nettoyage des tweets...")
if cfg.cleaner.lowercase:
    df['tweet'] = df['tweet'].str.lower()
if cfg.cleaner.remove_urls:
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
if cfg.cleaner.remove_mentions:
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))
if cfg.cleaner.remove_hashtags:
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#\w+', '', x))
if cfg.cleaner.remove_punctuation:
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
if cfg.cleaner.strip_whitespace:
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x))
    df['tweet'] = df['tweet'].apply(lambda x: x.strip())

# Supprimer les tweets vides (NaN ou chaînes vides)
df = df[~(df['tweet'].isna() | (df['tweet'].str.strip() == ""))]
logger.info(f"Suppression des tweets vides. {len(df)} tweets restants.")

logger.info("Nettoyage terminé.")

# Sauvegarder le dataset nettoyé
logger.info(f"Sauvegarde du dataset dans {output_path}...")
df.to_csv(output_path, index=False)
logger.info(f"Dataset sauvegardé dans {output_path}.")

# Afficher les métriques système à la fin du script
log_system_metrics()
