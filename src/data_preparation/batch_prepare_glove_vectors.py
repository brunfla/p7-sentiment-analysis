import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging
import yaml

def load_params(params_file, section):
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

def get_logger(name):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

# ------------------------------------------------
# 1) Charger GloVe
# ------------------------------------------------
def load_glove_model(glove_path: str, embedding_dim: int = 50):
    logger.info(f"Lecture du fichier GloVe : {glove_path}")
    glove_dict = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            if len(coefs) == embedding_dim:
                glove_dict[word] = coefs

    logger.info(f"Modèle GloVe chargé. Taille du vocab: {len(glove_dict)} mots.")
    return glove_dict

# ------------------------------------------------
# 2) Vectorisation SEQ d'un tweet
# ------------------------------------------------
def tweet_vectors_glove_seq(tweet: str, glove_dict: dict,
                            embedding_dim: int = 50,
                            max_seq_length: int = 15) -> np.ndarray:
    """
    Convertit un tweet en une matrice shape (max_seq_length, embedding_dim).
    Tronque et pad si besoin.
    """
    words = tweet.split()
    token_vectors = []
    for w in words:
        w_lower = w.lower()
        if w_lower in glove_dict:
            token_vectors.append(glove_dict[w_lower])
        else:
            token_vectors.append(np.zeros(embedding_dim, dtype="float32"))

    if len(token_vectors) > max_seq_length:
        token_vectors = token_vectors[:max_seq_length]

    token_vectors = np.array(token_vectors, dtype="float32")
    current_len = len(token_vectors)
    if current_len < max_seq_length:
        pad_len = max_seq_length - current_len
        pad_array = np.zeros((pad_len, embedding_dim), dtype="float32")
        token_vectors = np.concatenate([token_vectors, pad_array], axis=0)

    return token_vectors

# ------------------------------------------------
# 3) Nettoyage
# ------------------------------------------------
def is_valid_tweet(tweet: str, max_length: int = 10000) -> bool:
    if not tweet or not isinstance(tweet, str):
        return False
    num_words = len(tweet.split())
    if num_words == 0 or num_words > max_length:
        return False
    return True

def validate_and_clean(data_x: pd.DataFrame, data_y: pd.DataFrame, max_length: int):
    initial_size = len(data_x)
    valid_data_x = data_x[data_x["feature"].apply(lambda tw: is_valid_tweet(tw, max_length))]
    cleaned_size = len(valid_data_x)
    nb_removed = initial_size - cleaned_size
    logger.info(f"- {nb_removed} tweets invalides supprimés (vide ou > max_length).")

    valid_ids = valid_data_x["id"].unique()
    valid_data_y = data_y[data_y["id"].isin(valid_ids)]
    logger.info(f"- Synchronisation: {len(valid_data_y)} labels restants.")
    return valid_data_x, valid_data_y

# ------------------------------------------------
# 4) Vectorisation par chunks => concaténer en un seul tableau
# ------------------------------------------------
def vectorize_and_concat(df_x: pd.DataFrame,
                         glove_dict: dict,
                         embedding_dim: int,
                         max_seq_length: int,
                         chunk_size: int,
                         out_path: str):
    """
    Lit df_x par chunks de 'chunk_size', vectorise chaque chunk,
    puis concatène le tout en mémoire pour sauvegarder dans 'out_path'.
    => Retourne la forme finale de X.
    """
    n = len(df_x)
    all_ids = df_x["id"].tolist()
    all_tweets = df_x["feature"].fillna("").tolist()

    # On stocke temporairement les X de chaque chunk
    chunk_arrays = []

    start = 0
    part_idx = 0
    while start < n:
        end = min(start + chunk_size, n)
        sub_tweets = all_tweets[start:end]
        sub_ids = all_ids[start:end]

        logger.info(f"Vectorisation chunk {part_idx} (rows {start}:{end}) ...")
        chunk_vectors = []
        for tw in sub_tweets:
            seq_mat = tweet_vectors_glove_seq(tw, glove_dict, embedding_dim, max_seq_length)
            chunk_vectors.append(seq_mat)

        X_chunk = np.array(chunk_vectors, dtype="float32")
        logger.info(f" -> chunk shape = {X_chunk.shape}")

        chunk_arrays.append(X_chunk)

        start = end
        part_idx += 1

    # Concaténer tous les chunks en un seul array
    logger.info("Concaténation finale en mémoire ...")
    X_full = np.concatenate(chunk_arrays, axis=0)
    logger.info(f" X_full shape = {X_full.shape}")

    # Sauvegarde en pkl
    logger.info(f"Sauvegarde unique dans {out_path} ...")
    with open(out_path, "wb") as f:
        pickle.dump({"vectors": X_full, "ids": all_ids}, f)

    logger.info(f"Fichier sauvegardé: {out_path}")
    return X_full.shape

# ------------------------------------------------
# 5) Main
# ------------------------------------------------
def main():
    params_file = "params.yaml"
    section = "batch_prepare_glove_vectors"

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur de paramétrage dans '{params_file}' (section {section}): {e}")
        raise

    glove_path = params["glove_file"]
    input_dir  = params["input_dir"]
    input_files = params["input_files"]  # [x_train, x_val, x_test, y_train, y_val, y_test]
    output_dir = params["output_dir"]

    max_length = params.get("max_length", 10000)   # tweets > max_length tokens => invalid
    max_seq_length = 15  # On veut 15 tokens max par tweet (tronqué/pad)
    embedding_dim = params.get("embedding_dim", 50)
    chunk_size = params.get("chunk_size", 50000)

    # Vérifier 6 fichiers
    if len(input_files) != 6:
        msg = f"On attend 6 fichiers: x_train, x_val, x_test, y_train, y_val, y_test. Trouvé: {len(input_files)}"
        logger.error(msg)
        raise ValueError(msg)

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

    # Nettoyage
    logger.info("Nettoyage & synchro TRAIN...")
    df_x_train, df_y_train = validate_and_clean(df_x_train, df_y_train, max_length)
    logger.info("Nettoyage & synchro VAL...")
    df_x_val, df_y_val = validate_and_clean(df_x_val, df_y_val, max_length)
    logger.info("Nettoyage & synchro TEST...")
    df_x_test, df_y_test = validate_and_clean(df_x_test, df_y_test, max_length)

    logger.info(f"Chargement GloVe depuis {glove_path} ...")
    glove_dict = load_glove_model(glove_path, embedding_dim=embedding_dim)

    # Création du dossier output
    os.makedirs(output_dir, exist_ok=True)

    # TRAIN -> x_train_glove.pkl
    train_glove_path = os.path.join(output_dir, "x_train_glove.pkl")
    shape_train = vectorize_and_concat(df_x_train, glove_dict, embedding_dim, max_seq_length,
                                       chunk_size, train_glove_path)
    # Sauvegarde y_train
    y_train_path = os.path.join(output_dir, "y_train.csv")
    df_y_train.to_csv(y_train_path, index=False)
    logger.info(f"Sauvé {y_train_path}")

    # VAL -> x_val_glove.pkl
    val_glove_path = os.path.join(output_dir, "x_val_glove.pkl")
    shape_val = vectorize_and_concat(df_x_val, glove_dict, embedding_dim, max_seq_length,
                                     chunk_size, val_glove_path)
    y_val_path = os.path.join(output_dir, "y_val.csv")
    df_y_val.to_csv(y_val_path, index=False)
    logger.info(f"Sauvé {y_val_path}")

    # TEST -> x_test_glove.pkl
    test_glove_path = os.path.join(output_dir, "x_test_glove.pkl")
    shape_test = vectorize_and_concat(df_x_test, glove_dict, embedding_dim, max_seq_length,
                                      chunk_size, test_glove_path)
    y_test_path = os.path.join(output_dir, "y_test.csv")
    df_y_test.to_csv(y_test_path, index=False)
    logger.info(f"Sauvé {y_test_path}")

    logger.info("=== Fin de la préparation GloVe séquentielle unique ===")
    logger.info(f"Shapes: TRAIN={shape_train}, VAL={shape_val}, TEST={shape_test}")


if __name__ == "__main__":
    main()