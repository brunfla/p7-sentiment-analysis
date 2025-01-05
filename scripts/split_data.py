#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import argparse
import logging
import psutil
import requests
import subprocess
import pickle
import optuna
import mlflow
import dvc.api 

from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score

from omegaconf import DictConfig
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

# -------------------------
# Configurer le logging
# -------------------------
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
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
    # --- Parsing d'arguments ---

    # Configurer le logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger(__name__)

    # 1) Afficher les métriques système au démarrage
    log_system_metrics()

    # 2) Récupérer la config Hydra
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

    # Réinitialiser Hydra si déjà initialisé
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialiser Hydra avec la stratégie choisie
    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    vectorized_data_path = cfg.vectorizer.outputData
    
    # Si le fichier existe, on peut procéder à son chargement
    logger.info(f"Chargement des données vectorisées depuis {vectorized_data_path}...")
    with open(vectorized_data_path, 'rb') as f:
        X, y = pickle.load(f)
    logger.info(
        f"Données vectorisées chargées avec succès. "
        f"Taille X: {X.shape if hasattr(X, 'shape') else len(X)}, "
        f"Taille y: {len(y)}"
    )

    # ------------------------------------------------
    # 2) PARTITIONNER 
    # ------------------------------------------------
    partition_cfg = cfg.partitioner
    split_path = getattr(partition_cfg, "outputSplit", None)

    X_train, y_train = None, None
    X_val, y_val = None, None
    X_test, y_test = None, None
    folds = None

    logger.info(f"[Partitioner] Génération du partionnement {partition_cfg._target_}")
    if partition_cfg._target_ == "trainValTest":
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y,
            test_size=partition_cfg.testSize,
            random_state=partition_cfg.randomSeed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full,
            test_size=partition_cfg.validationSize,
            random_state=partition_cfg.randomSeed
        )
        logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

        # Sauvegarder le split
        if split_path:
            with open(split_path, "wb") as f:
                pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
            logger.info(f"[Partitioner] Nouveau split trainValTest sauvegardé dans {split_path}")

    elif partition_cfg._target_ == "trainTest":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=partition_cfg.testSize,
            random_state=partition_cfg.randomSeed
        )
        logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        if split_path:
            with open(split_path, "wb") as f:
                pickle.dump((X_train, y_train, X_test, y_test), f)
            logger.info(f"[Partitioner] Nouveau split trainTest sauvegardé dans {split_path}")

    elif partition_cfg._target_ == "crossValidation":
        kfold = KFold(
            n_splits=partition_cfg.folds,
            shuffle=True,
            random_state=partition_cfg.randomSeed
        )
        folds = list(kfold.split(X, y))
        logger.info(f"Nombre de folds: {len(folds)}")

        if split_path:
            with open(split_path, "wb") as f:
                pickle.dump((folds, X, y), f)
            logger.info(f"[Partitioner] Nouveau split crossValidation sauvegardé dans {split_path}")

    else:
        raise ValueError(f"Partition de découpage non reconnue: {partition_cfg._target_}")


if __name__ == "__main__":
    main()
