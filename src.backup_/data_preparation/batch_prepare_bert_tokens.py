import os
import sys
import pandas as pd
import pickle
from bert_loader import load_bert_tokenizer
from tweet_vectorization import tweet_token_bert

# Sauvegarder l'état initial de sys.path
original_sys_path = sys.path.copy()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params_utils.params_utils import load_params  
from logging_utils.logging_utils import get_logger
sys.path = original_sys_path

logger = get_logger(__name__)

def clean_sequences(data, key, max_length, default_value=0):
    """
    Nettoie les séquences en remplaçant les valeurs invalides par une valeur par défaut.

    Args:
        data (dict): Dictionnaire contenant les données tokenisées.
        key (str): Clé des données à nettoyer (par exemple, 'input_ids').
        max_length (int): Longueur maximale des séquences.
        default_value (int): Valeur par défaut pour remplacer les entrées invalides.

    Returns:
        list: Séquences nettoyées.
    """
    cleaned_data = []
    for idx, seq in enumerate(data[key]):
        if not isinstance(seq, list) or len(seq) != max_length:
            # Si la séquence est invalide, la remplacer par une séquence par défaut
            logger.warning(f"Séquence invalide dans '{key}' à l'index {idx}. Remplacement par la séquence par défaut.")
            seq = [default_value] * max_length
        else:
            # Remplacer les valeurs `None` ou non entières
            seq = [x if isinstance(x, int) else default_value for x in seq]
        cleaned_data.append(seq)
    return cleaned_data

def validate_sequences(data, key, max_length):
    """
    Valide que toutes les séquences dans une clé donnée sont valides.

    Args:
        data (dict): Dictionnaire contenant les données tokenisées.
        key (str): Clé des données à valider (par exemple, 'input_ids').
        max_length (int): Longueur maximale des séquences.

    Raises:
        ValueError: Si des séquences invalides sont détectées.
    """
    for idx, seq in enumerate(data[key]):
        if not isinstance(seq, list) or len(seq) != max_length or any(not isinstance(x, int) for x in seq):
            raise ValueError(f"Séquence invalide détectée dans '{key}' à l'index {idx}: {seq}")

import os
import pickle
import pandas as pd
from logging import getLogger

logger = getLogger(__name__)

def apply_bert_tokenization(input_file, tokenizer, output_file, text_column, label_column=None, max_length=128):
    """
    Applique la tokenisation BERT à un fichier CSV et sauvegarde les résultats tokenisés.

    Args:
        input_file (str): Chemin vers le fichier CSV contenant les tweets.
        tokenizer: Tokenizer BERT chargé.
        output_file (str): Chemin pour sauvegarder les données tokenisées.
        text_column (str): Nom de la colonne contenant les tweets.
        label_column (str, optional): Nom de la colonne contenant les labels. Defaults to None.
        max_length (int): Longueur maximale des séquences. Defaults to 128.
    """
    logger.info(f"Chargement des données depuis : {input_file}")
    data = pd.read_csv(input_file)

    if text_column not in data.columns:
        raise ValueError(f"La colonne '{text_column}' est absente dans {input_file}.")

    if label_column and label_column not in data.columns:
        logger.warning(f"La colonne '{label_column}' est absente dans {input_file}. Les labels seront ignorés.")

    # Nettoyage des tweets
    data[text_column] = data[text_column].fillna("").apply(lambda x: x.strip() if isinstance(x, str) else "")

    # Initialiser les structures pour les données tokenisées
    tokenized_data = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [] if label_column else None,
    }

    logger.info("Application de la tokenisation BERT avec traitement ligne par ligne...")
    for idx, row in data.iterrows():
        tweet = row[text_column]
        label = row[label_column] if label_column and pd.notna(row[label_column]) else None

        # Tokenisation du tweet
        tokenized = tweet_token_bert(tweet, tokenizer, max_length=max_length)

        # Ajout des données tokenisées
        tokenized_data["input_ids"].append(tokenized["input_ids"])
        tokenized_data["attention_mask"].append(tokenized["attention_mask"])

        # Ajout du label si présent
        if label_column and label is not None:
            try:
                tokenized_data["labels"].append(float(label))
            except ValueError as e:
                logger.warning(f"Label invalide à l'index {idx} : {label}. Erreur : {e}")

    # Validation finale des données
    if label_column and tokenized_data["labels"]:
        for idx, label in enumerate(tokenized_data["labels"]):
            if not isinstance(label, float) or label not in [0.0, 1.0]:
                logger.warning(f"Label inattendu détecté à l'index {idx} : {label}")

    # Sauvegarde des données tokenisées
    logger.info(f"Sauvegarde des données tokenisées dans : {output_file}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(tokenized_data, f)

    logger.info("Tokenisation BERT terminée.")

# -------------------------
# Point d'entrée principal
# -------------------------
if __name__ == "__main__":
    params_file = "params.yaml"
    section = "batch_prepare_bert_token"

    try:
        params = load_params(params_file, section)
    except KeyError as e:
        logger.error(f"Erreur dans le fichier des paramètres : {e}")
        raise

    # Paramètres du modèle
    pretrained_model = params.get("pretrained_model", "bert-base-uncased")
    input_files = params["input"]
    output_dir = params["output_dir"]
    text_column = params["text_column"]
    label_column = params.get("label_column", None)
    max_length = params["max_length"]

    # Charger le tokenizer BERT
    logger.info(f"Chargement du tokenizer BERT depuis : {pretrained_model}")

    tokenizer = load_bert_tokenizer(pretrained_model)

    # Appliquer la tokenisation à chaque fichier
    for split_name, input_file in input_files.items():
        logger.info(f"Traitement du fichier : {input_file}")
        output_file = os.path.join(output_dir, f"{split_name}_bert.pkl")
        apply_bert_tokenization(input_file, tokenizer, output_file, text_column, label_column, max_length)
