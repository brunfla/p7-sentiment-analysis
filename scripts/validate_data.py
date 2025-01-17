import os
import pandas as pd
import logging

# Configurer le logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def validate_data(file_path):
    """
    Valide les données dans un fichier CSV.
    Vérifie la présence et la validité des colonnes 'tweet' et 'id'.

    Args:
        file_path (str): Chemin vers le fichier CSV à valider.

    Returns:
        bool: True si les données sont valides, False sinon.
    """
    logger.info(f"Validation du fichier : {file_path}")

    if not os.path.exists(file_path):
        logger.error(f"Fichier introuvable : {file_path}")
        return False

    try:
        # Charger les données
        data = pd.read_csv(file_path)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier {file_path} : {e}")
        return False

    # Vérifier la présence des colonnes 'tweet' et 'id'
    required_columns = ["tweet", "id"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        logger.error(f"Colonnes manquantes dans {file_path} : {missing_columns}")
        return False

    # Vérifier les valeurs NaN ou vides dans 'tweet' et 'id'
    missing_tweet = data["tweet"].isna() | data["tweet"].str.strip().eq("")
    missing_id = data["id"].isna()

    invalid_rows = data[missing_tweet | missing_id]

    if not invalid_rows.empty:
        logger.warning(f"{len(invalid_rows)} lignes invalides détectées dans {file_path}.")
        logger.debug(f"Lignes invalides :\n{invalid_rows}")
        return False

    # Vérifier le type de la colonne 'id'
    try:
        data["id"] = pd.to_numeric(data["id"], errors="raise")
    except ValueError:
        logger.error(f"Certaines valeurs dans 'id' ne sont pas numériques dans {file_path}.")
        return False

    logger.info(f"Fichier {file_path} validé avec succès.")
    return True

def main():
    input_files = [
        "data/output/partitions/traintest/glove_cleaned/train.csv",
        "data/output/partitions/traintest/glove_cleaned/test.csv",
        "data/output/partitions/trainvaltest/glove_cleaned/train.csv",
        "data/output/partitions/trainvaltest/glove_cleaned/val.csv",
        "data/output/partitions/trainvaltest/glove_cleaned/test.csv"
    ]

    errors_detected = False

    for file_path in input_files:
        if not validate_data(file_path):
            errors_detected = True

    if errors_detected:
        logger.error("Certaines validations ont échoué. Veuillez corriger les erreurs.")
        exit(1)
    else:
        logger.info("Toutes les validations ont réussi.")

if __name__ == "__main__":
    main()

