#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import re
import spacy

# Initialiser Spacy pour la lemmatisation
nlp = spacy.load("en_core_web_sm")

# Fonction de lemmatisation
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_punct])

# Réinitialiser Hydra si déjà initialisé
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

# Initialiser Hydra avec une nouvelle configuration
initialize(config_path="../notebooks/config", version_base=None)
cfg = compose(config_name="simple-model-ci-github")

#Afficher la configuration globale
print("Configuration preprocess :")
print(cfg.preprocess)

# Vérifier si le prétraitement est activé
if not cfg.preprocess.enabled:
    print("\nLe prétraitement est désactivé. Fin du script.")
    exit()

# Charger le dataset
dataset_path = cfg.dataset.input.path  # Utiliser la clé correcte
df = pd.read_csv(dataset_path)
print(f"\nDataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")
print(f"\nDataset columns : {df.columns.tolist()}")

# Supprimer les tweets vides (NaN ou chaînes vides)
df = df[~(df['tweet'].isna() | (df['tweet'].str.strip() == ""))]
print(f"\nSuppression des tweets vides. {len(df)} tweets restants.")

# Nettoyer les tweets
print("\nNettoyage des tweets...")
df['tweet'] = df['tweet'].str.lower()
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))  # Supprimer les URLs
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))  # Supprimer les mentions
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#\w+', '', x))  # Supprimer les hashtags
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Supprimer la ponctuation
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x))  # Supprimer les espaces multiples
df['tweet'] = df['tweet'].apply(lambda x: x.strip())  # Supprimer les espaces en début et fin de chaîne

# Vérifier si la lemmatisation est activée
if cfg.preprocess.lemmatization.enabled:
    print("\nLemmatisation activée. Appliquer la lemmatisation...")
    df['tweet'] = df['tweet'].apply(lemmatize_text)
    print("\nLemmatisation terminée.")
else:
    print("\nLemmatisation désactivée.")

# Sauvegarder le dataset nettoyé
print("\nSauvegarde du dataset...")
df.to_csv(cfg.preprocess.output.path, index=False)
print(f"\nDataset sauvegardé dans {cfg.preprocess.output.path}.")
