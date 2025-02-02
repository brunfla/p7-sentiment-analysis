from transformers import BertTokenizer
from transformers import __version__ as transformers_version
from packaging import version

_bert_tokenizer_cache = None  # Cache pour le tokenizer BERT

def load_bert_tokenizer(pretrained_model_name_or_path="distilbert-base-uncased"):
    """
    Charge le tokenizer BERT dans une variable globale.

    Args:
        pretrained_model_name_or_path (str): Nom ou chemin du modèle pré-entraîné.

    Returns:
        BertTokenizer: Tokenizer BERT chargé.
    """

    global _bert_tokenizer_cache
    
    if _bert_tokenizer_cache is None:
        print(f"Chargement du tokenizer BERT depuis : {pretrained_model_name_or_path}")
        _bert_tokenizer_cache = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
    return _bert_tokenizer_cache

