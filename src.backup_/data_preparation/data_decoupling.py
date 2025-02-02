import pandas as pd
import os
import sys
import logging

# Charger les paramètres depuis params.yaml
def load_params(params_file, section):
    import yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

# Configurer le logger
def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

# Obtenir le logger
logger = get_logger(__name__)

def split_csv(parms):
    """
    Découpe un fichier CSV en deux fichiers :
    - Features.csv contenant les IDs et les features (text_column)
    - Labels.csv contenant les IDs et les labels (label_column)

    Args:
        input_csv (str): Chemin vers le fichier CSV d'entrée.
        text_column (str): Nom de la colonne contenant les features (texte).
        label_column (str): Nom de la colonne contenant les labels.
        id_column (str): Nom de la colonne contenant les IDs.
        output_dir (str): Répertoire de sortie pour les fichiers générés.

    Returns:
        None
    """
    input_csv=params["input_csv"],
    text_column=params["text_column"],
    label_column=params["label_column"],
    id_column=params["id_column"],
    output_dir=params["output_dir"]
    try:
        # Charger les données
        print(f"Chargement du fichier CSV : {input_csv}")
        data = pd.read_csv(input_csv, header=None, names=["id", "timestamp", "date", "query", "user", "tweet"])

        # Vérification des colonnes
        required_columns = [text_column, label_column, id_column]
        for column in required_columns:
            if column not in data.columns:
                raise ValueError(f"La colonne '{column}' est absente du fichier CSV.")

        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)

        # Extraire les features et les labels
        features = data[[id_column, text_column]]
        labels = data[[id_column, label_column]]

        # Sauvegarder les fichiers résultants
        features_file = os.path.join(output_dir, "features.csv")
        labels_file = os.path.join(output_dir, "labels.csv")

        features.to_csv(features_file, index=False)
        labels.to_csv(labels_file, index=False)

        print(f"Fichiers générés :\n- {features_file}\n- {labels_file}")

    except Exception as e:
        print(f"Erreur lors du découpage des données : {e}")

# Exemple d'utilisation
if __name__ == "__main__":

    # Charger les paramètres
    params_file = "params.yaml"
    section = os.path.splitext(os.path.basename(__file__))[0]  # Nom du script comme section

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        sys.exit(1)

    split_csv(params)

