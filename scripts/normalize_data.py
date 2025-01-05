#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from tqdm import tqdm
import spacy
import logging
import psutil
import os
import time

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

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Récupérer le chemin de configuration et la stratégie depuis les variables d'environnement
config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')
# Initialiser Hydra avec la stratégie choisie
initialize(config_path=config_path, version_base=None)
cfg = compose(config_name=strategy)

# Afficher la stratégie utilisée
print(f"Stratégie sélectionnée : {strategy}")

# Afficher la configuration globale
logger.info("Configuration normalizer:")
logger.info(cfg.normalizer)

#### --- MAIN --- ####
# Fonction pour lemmatiser un seul texte
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

# Ajouter nlp comme argument
def lemmatize_with_pipe(df, batch_size=1000, nlp=None):
    start_time = time.time()  # Temps de départ

    # Barre de progression
    lemmatized_texts = []
    for batch in tqdm(range(0, len(df), batch_size), desc="Lemmatisation", unit="batch"):
        texts = df['tweet'].iloc[batch:batch+batch_size].tolist()
        docs = nlp.pipe(texts, batch_size=batch_size)  # Traitement en batch
        lemmatized_texts.extend([" ".join([token.lemma_ for token in doc if not token.is_punct]) for doc in docs])

    elapsed_time = time.time() - start_time
    logger.info(f"Temps total écoulé: {elapsed_time:.2f} secondes.")

    return lemmatized_texts

# Gestion des targets
def handle_lemmatization():
    logger.info("Target 'lemmatization' détecté. Traitement en cours...")

    # Initialiser Spacy pour la lemmatisation
    nlp = spacy.load(cfg.normalizer.model)

    # Charger le dataset
    dataset_path = cfg.cleaner.output  # Utiliser la clé correcte
    logger.info(f"Chargement du dataset depuis {dataset_path}...")
    df = pd.read_csv(dataset_path)
    logger.info(f"Dataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")
    logger.info(f"Dataset columns: {df.columns.tolist()}")

    # Appliquer la lemmatisation
    logger.info("Lemmatisation activée. Appliquer la lemmatisation...")
    df['tweet'] = lemmatize_with_pipe(df, batch_size=1000, nlp=nlp)
    logger.info("Lemmatisation terminée.")

    # Sauvegarder le dataset nettoyé
    output_path = cfg.normalizer.output
    logger.info(f"Sauvegarde du dataset dans {output_path}...")
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset sauvegardé dans {output_path}.")

def handle_default():
    logger.warning(f"Target '{cfg.normalizer._target_}' non pris en charge. Fin du script.")
    exit()

# Dictionnaire des targets
targets = {
    "lemmatization": handle_lemmatization,
}

# Appeler la fonction correspondant au target ou gérer le cas par défaut
targets.get(cfg.normalizer._target_, handle_default)()

# Afficher les métriques système à la fin du script
log_system_metrics()
