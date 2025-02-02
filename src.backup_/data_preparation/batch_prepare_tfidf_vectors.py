import os
import sys
import pickle
import pandas as pd
from scipy.sparse import csr_matrix
from typing import Tuple

# Pour la vectorisation
from sklearn.feature_extraction.text import TfidfVectorizer

def load_params(params_file, section):
    import yaml
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

def get_logger(name):
    import logging
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# --------------------------------------------------------------------
# FONCTIONS DE NETTOYAGE
# --------------------------------------------------------------------
def is_valid_tweet(tweet: str, max_length: int = 10000) -> bool:
    """
    Détermine si un tweet est valide.
    - Non vide
    - Non nul
    - Nb de mots <= max_length
    """
    if not tweet or not isinstance(tweet, str):
        return False
    # Filtrer les tweets trop longs si nécessaire
    num_words = len(tweet.split())
    if num_words == 0 or num_words > max_length:
        return False
    return True

def validate_and_clean(data_x: pd.DataFrame, data_y: pd.DataFrame, max_length: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Valide et nettoie les tweets, en filtrant aussi
    les labels pour qu'ils soient synchronisés par ID.

    data_x: contient colonnes "id" et "feature"
    data_y: contient colonnes "id" et "label"

    Retourne: (x_nettoyé, y_nettoyé)
    """
    initial_size = len(data_x)
    valid_data_x = data_x[data_x["feature"].apply(lambda tw: is_valid_tweet(tw, max_length))]
    cleaned_size = len(valid_data_x)
    print(f"- {initial_size - cleaned_size} tweets invalides supprimés (vide ou > max_length).")

    valid_ids = valid_data_x["id"].unique()
    valid_data_y = data_y[data_y["id"].isin(valid_ids)]
    print(f"- Synchronisation: {len(valid_data_y)} labels restants (IDs correspondants).")

    return valid_data_x, valid_data_y


# --------------------------------------------------------------------
# FONCTION PRINCIPALE
# --------------------------------------------------------------------
def main():
    logger = get_logger(__name__)

    # 1) Lecture des paramètres
    params_file = "params.yaml"
    section = "batch_apply_tfidf_vectorizer"
    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier params.yaml : {e}")
        raise

    input_dir  = params["input_dir"]
    input_files = params["input_files"]  # ex: [x_train.csv, x_val.csv, x_test.csv, y_train.csv, y_val.csv, y_test.csv]
    output_dir = params["output_dir"]
    max_length = params.get("max_length", 10000)

    # Vérif: il faut 6 fichiers (x_train, x_val, x_test, y_train, y_val, y_test)
    if len(input_files) != 6:
        msg = f"On attend 6 fichiers (x_train, x_val, x_test, y_train, y_val, y_test). Trouvé: {len(input_files)}"
        logger.error(msg)
        raise ValueError(msg)

    # Mapper sur des variables plus claires
    x_train_file = os.path.join(input_dir, input_files[0])
    x_val_file   = os.path.join(input_dir, input_files[1])
    x_test_file  = os.path.join(input_dir, input_files[2])
    y_train_file = os.path.join(input_dir, input_files[3])
    y_val_file   = os.path.join(input_dir, input_files[4])
    y_test_file  = os.path.join(input_dir, input_files[5])

    logger.info("Lecture des CSV...")
    df_x_train = pd.read_csv(x_train_file)
    df_x_val   = pd.read_csv(x_val_file)
    df_x_test  = pd.read_csv(x_test_file)

    df_y_train = pd.read_csv(y_train_file)
    df_y_val   = pd.read_csv(y_val_file)
    df_y_test  = pd.read_csv(y_test_file)

    # 2) Nettoyage et synchronisation
    logger.info("Nettoyage et synchronisation TRAIN...")
    df_x_train, df_y_train = validate_and_clean(df_x_train, df_y_train, max_length)

    logger.info("Nettoyage et synchronisation VAL...")
    df_x_val, df_y_val     = validate_and_clean(df_x_val, df_y_val, max_length)

    logger.info("Nettoyage et synchronisation TEST...")
    df_x_test, df_y_test   = validate_and_clean(df_x_test, df_y_test, max_length)

    # 3) Création du vectoriseur TF-IDF entraîné SUR LE TRAIN UNIQUEMENT
    logger.info("Création du vectoriseur TF-IDF (fit sur TRAIN).")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(df_x_train["feature"])  # Entraîner sur les tweets "feature"

    # 4) Vectorisation des 3 splits
    logger.info("Transformation TF-IDF du train...")
    X_train_vec = vectorizer.transform(df_x_train["feature"])
    logger.info(f" X_train_vec shape: {X_train_vec.shape}")

    logger.info("Transformation TF-IDF du val...")
    X_val_vec   = vectorizer.transform(df_x_val["feature"])
    logger.info(f" X_val_vec shape: {X_val_vec.shape}")

    logger.info("Transformation TF-IDF du test...")
    X_test_vec  = vectorizer.transform(df_x_test["feature"])
    logger.info(f" X_test_vec shape: {X_test_vec.shape}")

    # 5) Sauvegarde des matrices + labels
    logger.info("Sauvegarde des résultats...")
    os.makedirs(output_dir, exist_ok=True)

    # (a) Sauvegarde X_train_vec.pkl
    train_vec_path = os.path.join(output_dir, "x_train_vec.pkl")
    with open(train_vec_path, "wb") as f:
        pickle.dump({"vectors": X_train_vec, "ids": df_x_train["id"].tolist()}, f)
    logger.info(f"Sauvegardé: {train_vec_path}")

    # (b) Sauvegarde X_val_vec.pkl
    val_vec_path = os.path.join(output_dir, "x_val_vec.pkl")
    with open(val_vec_path, "wb") as f:
        pickle.dump({"vectors": X_val_vec, "ids": df_x_val["id"].tolist()}, f)
    logger.info(f"Sauvegardé: {val_vec_path}")

    # (c) Sauvegarde X_test_vec.pkl
    test_vec_path = os.path.join(output_dir, "x_test_vec.pkl")
    with open(test_vec_path, "wb") as f:
        pickle.dump({"vectors": X_test_vec, "ids": df_x_test["id"].tolist()}, f)
    logger.info(f"Sauvegardé: {test_vec_path}")

    # (d) Sauvegarde y_*.csv nettoyés
    y_train_path = os.path.join(output_dir, "y_train.csv")
    df_y_train.to_csv(y_train_path, index=False)
    logger.info(f"Sauvegardé: {y_train_path}")

    y_val_path = os.path.join(output_dir, "y_val.csv")
    df_y_val.to_csv(y_val_path, index=False)
    logger.info(f"Sauvegardé: {y_val_path}")

    y_test_path = os.path.join(output_dir, "y_test.csv")
    df_y_test.to_csv(y_test_path, index=False)
    logger.info(f"Sauvegardé: {y_test_path}")

    # Bonus: si on veut aussi sauvegarder le vectorizer pour usage ultérieur
    vectorizer_path = os.path.join(output_dir, "tfidf_vectorizer.pkl")
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Vectorizer TF-IDF sauvegardé: {vectorizer_path}")

    logger.info("=== Fin ===")


if __name__ == "__main__":
    main()
