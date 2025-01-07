#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging
import psutil
import re
import pandas as pd

from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

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

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    # 1) Afficher les métriques système
    log_system_metrics()

    # 2) Récupérer la config Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    # Réinitialiser Hydra si déjà initialisé
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialiser Hydra avec la stratégie choisie
    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    # 4) Afficher la stratégie et la config
    logger.info(f"Stratégie sélectionnée : {strategy}")
    logger.info("Configuration preprocess:")
    logger.info(cfg.cleaner)

    # 5) Définir les chemins
    data_path = cfg.cleaner.input
    output_path = cfg.cleaner.output

    # 6) Charger le dataset
    logger.info(f"Chargement du dataset depuis {data_path}...")
    df = pd.read_csv(
        data_path,
        header=None,
        names=["id", "timestamp", "date", "query", "user", "tweet"]
    )
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")

    # 7) Nettoyer les tweets
    logger.info("Nettoyage des tweets...")
    df['tweet'] = df['tweet'].str.lower()  # minuscule
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+', '', x))  # enlever URLs
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))           # enlever mentions
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#\w+', '', x))           # enlever hashtags
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))        # enlever ponctuation
    df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())   # enlever espaces multiples

    # Retirer les tweets vides
    df = df[~(df['tweet'].isna() | (df['tweet'].str.strip() == ""))]
    logger.info(f"Suppression des tweets vides. {len(df)} tweets restants.")
    logger.info("Nettoyage terminé.")

    # 8) Sauvegarder le dataset nettoyé
    logger.info(f"Sauvegarde du dataset dans {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info("Dataset sauvegardé avec succès.")

    # 9) Afficher les métriques système à la fin
    log_system_metrics()

if __name__ == "__main__":
    main()
