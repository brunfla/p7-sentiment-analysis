from logging_utils import get_logger
import pandas as pd

# -------------------------
# Fonction pour charger et nettoyer les données
# -------------------------
def load_data(input_path, text_column, label_column):
    logger =  get_logger()
    logger.info(f"Chargement des données depuis {input_path}...")
    data = pd.read_csv(input_path, header=None, names=["id", "timestamp", "date", "query", "user", "tweet"])
    logger.info(f"Dataset chargé avec {len(data)} lignes.")
    data["id"] = data["id"].apply(lambda x: 1 if x == 4 else x)  # Transformer 4 -> 1
    data = data[~data[text_column].isna()]
    data = data[data[text_column].str.strip() != ""]
    logger.info(f"Données nettoyées : {len(data)} lignes restantes.")
    return data
