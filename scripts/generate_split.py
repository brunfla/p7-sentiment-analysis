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


# Fonction pour effectuer un pull avec subprocess
def pull_data_from_dvc(vectorized_data_path):
    try:
        # Effectuer un pull via la commande DVC en ligne de commande
        subprocess.run(["dvc", "pull", vectorized_data_path + ".dvc"], check=True)
        logger.info(f"DVC pull effectué pour récupérer {vectorized_data_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de dvc pull via la commande en ligne de commande: {e}")
        raise e

# Fonction pour effectuer un push avec subprocess
def push_data_to_dvc(vectorized_data_path):
    try:
        # Effectuer un push via la commande DVC en ligne de commande
        subprocess.run(["dvc", "push", vectorized_data_path + ".dvc"], check=True)
        logger.info(f"DVC push effectué pour envoyer {vectorized_data_path} vers le stockage DVC.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors de l'exécution de dvc push via la commande en ligne de commande: {e}")
        raise e

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
    config_path = os.getenv('HYDRA_CONFIG_PATH', '../notebooks/config')
    strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

    # Réinitialiser Hydra si déjà initialisé
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    # Initialiser Hydra avec la stratégie choisie
    initialize(config_path=config_path, version_base=None)
    cfg = compose(config_name=strategy)

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # ------------------------------------------------
    # 0) TIRER (PULL) LES DONNÉES DEPUIS DVC (si nécessaire)
    # ------------------------------------------------
    vectorized_data_path = "../data/output/vectorized_data.pkl"
    
    # Si le fichier n'existe pas localement, on effectue un dvc pull
    if not os.path.exists(vectorized_data_path):
        logger.info(f"Le fichier {vectorized_data_path} n'existe pas localement, récupération via DVC...")
        pull_data_from_dvc(vectorized_data_path)
    else:
        logger.info(f"Le fichier {vectorized_data_path} existe déjà localement.")

    # Vérifier à nouveau si le fichier existe après le pull (au cas où DVC n'ait pas réussi)
    if not os.path.exists(vectorized_data_path):
        logger.error(
            f"ÉCHEC : le fichier vectorisé {vectorized_data_path} est introuvable après DVC pull. "
            "Veuillez d'abord exécuter l'étape de vectorisation ou vérifier DVC."
        )
        sys.exit(1)

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
    existing_split = getattr(partition_cfg, "existingSplitData", False)

    X_train, y_train = None, None
    X_val, y_val = None, None
    X_test, y_test = None, None
    folds = None

    if split_path and existing_split and os.path.exists(split_path):
        logger.info(f"[Partitioner] Chargement d'un split existant depuis {split_path}...")
        with open(split_path, "rb") as f:
            if partition_cfg._target_ == "trainValTest":
                (X_train, y_train, X_val, y_val, X_test, y_test) = pickle.load(f)
                logger.info(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            elif partition_cfg._target_ == "trainTest":
                (X_train, y_train, X_test, y_test) = pickle.load(f)
                logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
            elif partition_cfg._target_ == "crossValidation":
                (folds, X, y) = pickle.load(f)
                logger.info(f"Nombre de folds: {len(folds)} (cross-validation)")
            else:
                raise ValueError(f"Partitioner non reconnu: {partition_cfg._target_}")

    else:
        logger.info("[Partitioner] Pas de split existant. Génération d'un nouveau split...")
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
