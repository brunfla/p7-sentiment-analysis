#!/usr/bin/env python
# coding: utf-8

import os
import logging
import psutil
import pickle

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

from sklearn.feature_extraction.text import TfidfVectorizer

from partitioned_data import PartitionedData
from glove_vectorizer import handle_glove  # <-- import ici
from tfidf_vectorizer import handle_tfidf

from log_system import log_system_metrics, logger

def apply_vectorizer(partitioned_data, cfg):
    """Applique le vectoriseur en fonction du type spécifié dans la config."""
    # Associer chaque type de vectoriseur à la fonction appropriée
    vectorizers = {
        "tfidfVectorizer": handle_tfidf,
        "gloveVectorizer": handle_glove,  # <-- on a changé la clé pour "gloveVectorizer"
    }

    vectorizer_type = cfg.vectorizer.type
    if vectorizer_type not in vectorizers:
        raise ValueError(f"Type de vectoriseur non supporté : {vectorizer_type}")

    logger.info(f"Vectoriseur sélectionné : {vectorizer_type}")
    return vectorizers[vectorizer_type](partitioned_data, cfg)

def main():
    log_system_metrics()

    # Initialiser Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")
    logger.info("Configuration vectorizer:")
    logger.info(cfg.vectorizer)

    # Charger les données partitionnées
    dataset_path = cfg.vectorizer.input
    logger.info(f"Chargement des données depuis {dataset_path}...")
    partitioned_data = PartitionedData.load(dataset_path)
    logger.info(f"Données chargées avec succès. Type de partition : {partitioned_data.partition_type}")

    # Appliquer le vectoriseur
    partitioned_data, vectorizer = apply_vectorizer(partitioned_data, cfg)

    # Sauvegarder les données transformées
    output_data_path = cfg.vectorizer.outputData
    logger.info(f"Sauvegarde des données vectorisées dans {output_data_path}...")
    partitioned_data.save(output_data_path)

    # Sauvegarder le modèle du vectoriseur
    output_model_path = cfg.vectorizer.outputPath
    logger.info(f"Sauvegarde du modèle de vectorisation dans {output_model_path}...")
    with open(output_model_path, "wb") as f:
        pickle.dump(vectorizer, f)

    logger.info("Traitement de vectorisation terminé avec succès.")
    log_system_metrics()

if __name__ == "__main__":
    main()
