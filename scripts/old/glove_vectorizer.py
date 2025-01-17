import logging
import numpy as np

logger = logging.getLogger(__name__)

def handle_glove(partitioned_data, cfg):
    """
    Gestion du vectoriseur GloVe (exemple avec moyenne d'embeddings).
    - Lit le fichier GloVe
    - Construit un embeddings_index {mot: vecteur}
    - Transforme X_train, X_val, X_test en matrices de taille (nb_samples, embedding_size)
    - Retourne (partitioned_data, glove_vectorizer) 
    """
    logger.info("Chargement des vecteurs GloVe...")

    # 1) Charger le fichier d'embeddings GloVe
    glove_file = cfg.vectorizer.embeddingFile  # ex: data/input/glove.6B.50d.txt
    embedding_size = cfg.vectorizer.embeddingSize
    logger.info(f"Lecture du fichier GloVe : {glove_file}")

    embeddings_index = {}
    with open(glove_file, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = [float(v) for v in values[1:]]  # vecteur
            embeddings_index[word] = coefs

    logger.info(f"GloVe: {len(embeddings_index)} mots chargés, dimension={embedding_size}")

    # 2) Limiter la taille du vocabulaire (vocabularySize) [Optionnel]
    vocab_size = cfg.vectorizer.vocabularySize
    logger.info(f"Vocabulaire limité à {vocab_size} mots (si besoin). [Non implémenté ici]")

    # 3) Fonction utilitaire pour transformer une liste de phrases en tableau de vecteurs
    def embed_texts(texts):
        """
        text : liste de strings (chaque string est une phrase).
        Retourne un np.array de shape (len(texts), embedding_size).
        """
        embedded_samples = []
        for sentence in texts:
            tokens = sentence.split()  # Tokenisation simple
            word_vectors = []
            for w in tokens:
                if w in embeddings_index:
                    word_vectors.append(embeddings_index[w])
                else:
                    # Mot hors vocab GloVe, vecteur nul ou random
                    word_vectors.append([0.0]*embedding_size)

            if len(word_vectors) == 0:
                # Phrase vide ou tokens inconnus => vecteur nul
                word_vectors = [[0.0]*embedding_size]

            # Moyenne des vecteurs de mots pour obtenir un embedding global de phrase
            sentence_embedding = np.mean(word_vectors, axis=0)
            embedded_samples.append(sentence_embedding)

        return np.array(embedded_samples, dtype=np.float32)

    # 4) Transformer X_train, X_val, X_test
    logger.info("Transformation des données avec GloVe (moyenne des embeddings).")

    if "X_train" in partitioned_data.data:
        X_train_texts = partitioned_data.data["X_train"]
        partitioned_data.data["X_train"] = embed_texts(X_train_texts)
        logger.info(f"X_train => shape : {partitioned_data.data['X_train'].shape}")

    if "X_val" in partitioned_data.data and partitioned_data.data["X_val"] is not None:
        X_val_texts = partitioned_data.data["X_val"]
        partitioned_data.data["X_val"] = embed_texts(X_val_texts)
        logger.info(f"X_val => shape : {partitioned_data.data['X_val'].shape}")

    if "X_test" in partitioned_data.data and partitioned_data.data["X_test"] is not None:
        X_test_texts = partitioned_data.data["X_test"]
        partitioned_data.data["X_test"] = embed_texts(X_test_texts)
        logger.info(f"X_test => shape : {partitioned_data.data['X_test'].shape}")

    # 5) Créer un objet "vectorizer" (un dict pour conserver quelques infos)
    glove_vectorizer = {
        "embeddings_index": embeddings_index,
        "embedding_size": embedding_size,
        "vocab_size": vocab_size
        # éventuellement d’autres infos (tokenizer, etc.)
    }

    return partitioned_data, glove_vectorizer

