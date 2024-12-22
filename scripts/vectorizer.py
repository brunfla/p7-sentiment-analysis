#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Initialiser Hydra avec une nouvelle configuration
initialize(config_path="../notebooks/config", version_base=None)
cfg = compose(config_name="simple-model-ci-github")

#Afficher la configuration globale
print("Configuration vectorizer :")
print(cfg.vectorizer)

# Vérifier si la clé preprocess.enabled est activée
if cfg.preprocess.enabled:
    dataset_path = cfg.preprocess.output.path  # Utiliser la sortie prétraitée
else:
    dataset_path = cfg.dataset.input.path  # Utiliser l'entrée brute

# Charger le dataset
df = pd.read_csv(dataset_path)
print(f"\nDataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")
print(f"\nDataset columns : {df.columns.tolist()}")

# Vérifier le type de vectorizer configuré
print("\nConfiguration du vectoriseur TF-IDF...")
if cfg.vectorizer._target_ == "tfidfVectorizer":
    vectorizer = TfidfVectorizer(
        stop_words=cfg.vectorizer.stopWords,
        max_features=cfg.vectorizer.maxFeatures,
        ngram_range=tuple(cfg.vectorizer.ngramRange)
    )
else:
    raise KeyError("La configuration 'tfidVectorizer' est absente ou mal définie dans 'vectorizer'.")

# Appliquer fit_transform sur les tweets
print("\nAppliquer TF-IDF sur les tweets...")
X = vectorizer.fit_transform(df['tweet'])

# Enregistrer
with open("../data/output/vectorizer.pkl", "wb") as f:
    pickle.dump(X, f)

