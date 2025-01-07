#!/usr/bin/env python
# coding: utf-8

import os
import pickle
import logging
import argparse
import psutil
from sklearn.model_selection import train_test_split, KFold
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
# Fonctions utilitaires pour le partitionnement
# ------------------------------------------------
def save_partition(output_path, data):
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    logger.info(f"[Partitioner] Données sauvegardées dans {output_path}")

def train_val_test_split(X, y, cfg):
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=cfg.testSize,
        random_state=cfg.randomSeed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=cfg.validationSize,
        random_state=cfg.randomSeed
    )
    logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_test_split_only(X, y, cfg):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.testSize,
        random_state=cfg.randomSeed
    )
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, y_train, X_test, y_test

def cross_validation_split(X, y, cfg):
    kfold = KFold(
        n_splits=cfg.folds,
        shuffle=True,
        random_state=cfg.randomSeed
    )
    folds = list(kfold.split(X, y))
    logger.info(f"Nombre de folds: {len(folds)}")
    return folds

# ------------------------------------------------
# SCRIPT PRINCIPAL
# ------------------------------------------------
def main():
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

    input_path = cfg.partitioner.input
    output_path = cfg.partitioner.output

    # Chargement des données vectorisées
    logger.info(f"Chargement des données vectorisées depuis {input_path}...")
    with open(input_path, 'rb') as f:
        X, y = pickle.load(f)
    logger.info(
        f"Données vectorisées chargées avec succès. "
        f"Taille X: {X.shape if hasattr(X, 'shape') else len(X)}, "
        f"Taille y: {len(y)}"
    )

    # ------------------------------------------------
    # Partitionnement basé sur la configuration
    # ------------------------------------------------
    partition_cfg = cfg.partitioner
    if partition_cfg._target_ == "trainValTest":
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, partition_cfg)
        save_partition(output_path, (X_train, y_train, X_val, y_val, X_test, y_test))

    elif partition_cfg._target_ == "trainTest":
        X_train, y_train, X_test, y_test = train_test_split_only(X, y, partition_cfg)
        save_partition(output_path, (X_train, y_train, X_test, y_test))

    elif partition_cfg._target_ == "crossValidation":
        folds = cross_validation_split(X, y, partition_cfg)
        save_partition(output_path, (folds, X, y))

    else:
        raise ValueError(f"Partition de découpage non reconnue: {partition_cfg._target_}")

if __name__ == "__main__":
    main()
