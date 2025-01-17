#!/usr/bin/env python
# coding: utf-8

import os
import logging
import psutil
import spacy
from partitioned_data import PartitionedData

# Configurer le logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# ------------------------------------------------
# Fonction pour afficher les métriques système
# ------------------------------------------------
def log_system_metrics():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )

# ------------------------------------------------
# Gestion des différentes transformations
# ------------------------------------------------
def handle_lemmatization(partitioned_data, cfg):
    """Applique la lemmatisation sur les données de features."""
    logger.info("Initialisation du modèle spaCy pour la lemmatisation...")
    nlp = spacy.load(cfg.normalizer.model)

    def lemmatize_with_pipe(texts):
        """Lemmatise un lot de textes."""
        if texts is None:
            return None
        lemmatized_texts = []
        for doc in nlp.pipe(texts, batch_size=1000):
            lemmatized_texts.append(" ".join(token.lemma_ for token in doc if not token.is_punct))
        return lemmatized_texts

    logger.info("Application de la lemmatisation...")
    for key in partitioned_data.data:
        if key.startswith("X_"):  # Appliquer uniquement sur les features
            logger.info(f"Lemmatizing {key}...")
            partitioned_data.data[key] = lemmatize_with_pipe(partitioned_data.data[key])

    return partitioned_data

# ------------------------------------------------
# Fonction générique pour appliquer des transformations
# ------------------------------------------------
def apply_transformation(partitioned_data, cfg):
    """Applique une transformation basée sur la configuration."""
    transformations = {
        "lemmatization": handle_lemmatization,
        # Ajouter d'autres transformations si nécessaire
    }

    transformation_type = cfg.normalizer.type
    if transformation_type not in transformations:
        raise ValueError(f"Type de transformation non supporté : {transformation_type}")

    logger.info(f"Transformation sélectionnée : {transformation_type}")
    return transformations[transformation_type](partitioned_data, cfg)

# ------------------------------------------------
# Orchestration principale
# ------------------------------------------------
def main():
    # Afficher les métriques système
    log_system_metrics()

    # Charger la configuration
    from hydra import initialize, compose
    from hydra.core.global_hydra import GlobalHydra

    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")
    logger.info("Configuration normalizer:")
    logger.info(cfg.normalizer)

    # Charger les données partitionnées
    dataset_path = cfg.normalizer.input
    logger.info(f"Chargement des données depuis {dataset_path}...")
    partitioned_data = PartitionedData.load(dataset_path)
    logger.info(f"Données chargées avec succès. Type de partition : {partitioned_data.partition_type}")

    # Appliquer la transformation
    partitioned_data = apply_transformation(partitioned_data, cfg)

    # Sauvegarder les données après transformation
    output_path = cfg.normalizer.output
    logger.info(f"Sauvegarde des données transformées dans {output_path}...")
    partitioned_data.save(output_path)
    logger.info(f"Données sauvegardées avec succès dans {output_path}.")

    # Afficher les métriques système à la fin
    log_system_metrics()

if __name__ == "__main__":
    main()
