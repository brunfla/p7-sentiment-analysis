#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

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
initialize(config_path="../notebook/config", version_base=None)
cfg = compose(config_name="simple-model-ci-github")

# Afficher la configuration globale
#print("Configuration globale :")
#print(cfg)

# Vérifier si le prétraitement est activé
if not cfg.preprocess.enabled:
    print("\nLe prétraitement est désactivé. Fin du script.")
    exit()

# Charger les paramètres du modèle
#model_config = cfg.model
#print(f"\nModèle sélectionné : {model_config.name}")
#print(f"Paramètres du modèle : {model_config.parameters}")

# Charger le dataset
dataset_path = cfg.dataset.input.path  # Utiliser la clé correcte
df = pd.read_csv(dataset_path)
print(f"\nDataset chargé avec {len(df)} lignes et {len(df.columns)} colonnes.")

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
print(f"TF-IDF vectorisation terminée avec {X.shape[1]} caractéristiques.")

# Sauvegarder le dataset nettoyé
print("\nSauvegarde du dataset nettoyé...")
df.to_csv(cfg.dataset.output.path, index=False)
print(f"\nDataset nettoyé sauvegardé dans {cfg.dataset.output.path}.")
