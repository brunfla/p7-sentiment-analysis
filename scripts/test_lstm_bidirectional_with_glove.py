import os
import json
import logging
import mlflow
import pandas as pd
import numpy as np
import sys
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score
)
from params_utils import load_params
from logging_utils import get_logger

# Configurer le logger
logger = get_logger(__name__)

from test_logistic_tfidf import load_model_run_id, load_model, evaluate_model

def main():
    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]  # Nom du script comme section

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur lors du chargement des paramètres : {e}")
        sys.exit(1)

    # Charger le run_id
    try:
        model_run_id = load_model_run_id(params["model_run_id_file"])
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Erreur lors du chargement de l'identifiant du modèle : {e}")
        sys.exit(1)

    # Charger le modèle
    try:
        model = load_model(model_run_id)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        sys.exit(1)

    # Évaluer le modèle
    evaluate_model(
        model=model,
        test_data_path=params["input_file"],
        output_file=params["output_file"],
        threshold=params.get("threshold", 0.5),
        metrics_cfg=params.get("metrics", ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]),
    )

if __name__ == "__main__":
    main()