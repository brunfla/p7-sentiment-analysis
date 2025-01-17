import os
import mlflow
import mlflow.sklearn
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from params_utils import load_params
from logging_utils import get_logger
from train_logistic_tfidf import train_logistic_regression
import pandas as pd
import sys

# Obtenir le logger
logger = get_logger(__name__)


if __name__ == "__main__":
    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    try:
        input_file = params["input_file"]
        output_dir = params["output_dir"]
        model_type = params["model_type"]
        tuning_cfg = params.get("tuning", {})
        mlflow_cfg = params.get("mlflow", {})
    except KeyError as e:
        logger.error(f"Clé manquante dans la section '{section}': {e}")
        sys.exit(1)

    train_logistic_regression(input_file, output_dir, model_type, tuning_cfg, mlflow_cfg)
