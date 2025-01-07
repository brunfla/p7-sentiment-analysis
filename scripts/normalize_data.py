#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
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
strategy = os.getenv('HYDRA_STRATEGY', 'baseline')
# Initialiser Hydra avec la stratégie choisie
initialize(config_path=config_path, version_base=None)
cfg = compose(config_name=strategy)

# Afficher la stratégie utilisée
logger.info(f"Stratégie sélectionnée : {strategy}")

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

    # Charger le dataset depuis le fichier pkl
    partition_cfg = cfg.partitioner
    dataset_path = partition_cfg.output  # Chemin du fichier pickle
    logger.info(f"Chargement des données depuis {dataset_path}...")
    
    with open(dataset_path, "rb") as f:
        if partition_cfg._target_ == "trainValTest":
            X_train, y_train, X_val, y_val, X_test, y_test = pickle.load(f)
        elif partition_cfg._target_ == "trainTest":
            X_train, y_train, X_test, y_test = pickle.load(f)
            X_val, y_val = None, None
        elif partition_cfg._target_ == "crossValidation":
            folds, X, y = pickle.load(f)
        else:
            raise ValueError(f"Partition de découpage non reconnue: {partition_cfg._target_}")

    logger.info("Données chargées avec succès.")

    # Appliquer la lemmatisation sur chaque partition
    def apply_lemmatization(X, nlp):
        if X is not None:
            df = pd.DataFrame({"tweet": X})
            df['tweet'] = lemmatize_with_pipe(df, batch_size=1000, nlp=nlp)
            return df['tweet'].tolist()
        return None

    X_train = apply_lemmatization(X_train, nlp)
    X_val = apply_lemmatization(X_val, nlp)
    X_test = apply_lemmatization(X_test, nlp)

    # Sauvegarder le dataset lemmatisé dans un fichier pkl
    output_path = cfg.normalizer.output
    logger.info(f"Sauvegarde des données lemmatisées dans {output_path}...")

    if partition_cfg._target_ == "trainValTest":
        with open(output_path, "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
    elif partition_cfg._target_ == "trainTest":
        with open(output_path, "wb") as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
    elif partition_cfg._target_ == "crossValidation":
        with open(output_path, "wb") as f:
            pickle.dump((folds, X, y), f)

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