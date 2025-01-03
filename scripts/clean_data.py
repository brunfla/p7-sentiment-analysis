#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import hashlib
import subprocess
import os
import sys
import logging
import psutil
from datetime import datetime
import re

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

def get_git_commit_hash():
    """
    Récupère le hash du commit Git courant.
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception as e:
        logger.warning(f"Impossible de récupérer le hash du commit Git : {e}")
        return None

def is_processing_needed(output_path):
    """
    Vérifie si le traitement est nécessaire en se basant sur le hash du script (commit Git).
    """
    logger.info("Début de la vérification du besoin de traitement pour le script.")

    # Récupérer le hash du commit Git courant
    script_hash = get_git_commit_hash()
    if script_hash:
        logger.info(f"Hash du commit Git courant : {script_hash}")
    else:
        logger.warning("Impossible de récupérer le hash du commit Git. Le traitement sera effectué.")
        return True

    # Chemin du fichier d'état
    state_file = output_path + ".state"

    # Vérifier l'existence du fichier d'état
    if os.path.exists(state_file):
        logger.info(f"Lecture du fichier d'état précédent : {state_file}")
        with open(state_file, "r") as f:
            saved_script_hash = f.read().strip()
        logger.info(f"Hash précédent du script : {saved_script_hash}")

        # Comparaison des hash
        if script_hash == saved_script_hash:
            logger.info("Le hash du script est identique. Le traitement n'est pas nécessaire.")
            return False
        else:
            logger.info("Le hash du script a changé. Le traitement est nécessaire.")
    else:
        logger.info(f"Le fichier d'état {state_file} est introuvable. Le traitement est nécessaire.")

    # Sauvegarder le nouveau hash du script
    logger.info(f"Sauvegarde du hash du script dans {state_file}")
    with open(state_file, "w") as f:
        f.write(script_hash)
    logger.info("Le nouveau hash du script a été sauvegardé.")

    return True

def pull_data_from_dvc(data_path):
    """
    Exécute 'dvc pull' pour récupérer un fichier géré par DVC.
    """
    try:
        subprocess.run(["dvc", "pull", data_path + ".dvc"], check=True)
        logger.info(f"DVC pull effectué pour récupérer {data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de dvc pull : {e}")
        raise e

def push_data_to_dvc(data_path):
    """
    Exécute 'dvc push' pour envoyer un fichier géré par DVC.
    """
    try:
        subprocess.run(["dvc", "push", data_path + ".dvc"], check=True)
        logger.info(f"DVC push effectué pour envoyer {data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de dvc push : {e}")
        raise e

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------

# 1) Afficher les métriques système au démarrage
log_system_metrics()

# 2) Récupérer la config Hydra
config_path = os.getenv('HYDRA_CONFIG_PATH', '../notebooks/config')
strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Initialiser Hydra avec la stratégie choisie
initialize(config_path=config_path, version_base=None)
cfg = compose(config_name=strategy)

# 3) Vérifier si la partie "cleaner" est activée
if not cfg.cleaner.enabled:
    logger.info("Le traitement est désactivé. Fin du script.")
    sys.exit(0)

# 4) Afficher la stratégie et la config
logger.info(f"Stratégie sélectionnée : {strategy}")
logger.info("Configuration preprocess:")
logger.info(cfg.cleaner)

# 5) Définir les chemins
data_path = cfg.dataset.path
output_path = cfg.cleaner.output

# 6) Vérifier la présence du fichier d'entrée
if not os.path.exists(data_path):
    logger.info(f"Le fichier {data_path} n'existe pas localement, récupération via DVC...")
    pull_data_from_dvc(data_path)

if not os.path.exists(data_path):
    logger.error(f"ÉCHEC : Le fichier {data_path} est introuvable après DVC pull.")
    sys.exit(1)

# 7) Vérifier si le traitement est nécessaire (basé sur le hash du script)
if not is_processing_needed(output_path):
    logger.info("Le traitement a déjà été effectué (script inchangé). Sortie.")
    sys.exit(0)

# 8) Charger les données
logger.info(f"Chargement du dataset depuis {data_path}...")
df = pd.read_csv(
    data_path,
    header=None,
    names=["id", "timestamp", "date", "query", "user", "tweet"]
)
logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")

# 9) Nettoyer les tweets
logger.info("Nettoyage des tweets...")
df['tweet'] = df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#\w+', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Retirer les tweets vides
df = df[~(df['tweet'].isna() | (df['tweet'].str.strip() == ""))]
logger.info(f"Suppression des tweets vides. {len(df)} tweets restants.")
logger.info("Nettoyage terminé.")

# 10) Sauvegarder le dataset nettoyé
logger.info(f"Sauvegarde du dataset dans {output_path}...")
df.to_csv(output_path, index=False)
logger.info("Dataset sauvegardé avec succès.")

# 11) Pousser les données nettoyées vers DVC
logger.info("Envoi des données nettoyées vers DVC...")
push_data_to_dvc(output_path)

# 12) Afficher les métriques système à la fin
log_system_metrics()

