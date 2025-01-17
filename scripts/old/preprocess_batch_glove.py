import os
from preprocess_common import preprocess_partition, preprocess_tweet, lemmatize_and_remove_stopwords
from gensim.models import KeyedVectors

def preprocess_with_glove(tweet, glove_model, threshold=0.6):
    """
    Nettoyage avec GloVe : suppression/correction des mots invalides.
    """
    def is_valid_word(word):
        if word in glove_model:
            return worda
        similar_words = glove_model.most_similar(word, topn=1) if word else []
        if similar_words and similar_words[0][1] > threshold:
            return similar_words[0][0]
        return None

    words = tweet.split()
    cleaned_words = [is_valid_word(word) for word in words if is_valid_word(word)]
    return " ".join(cleaned_words)


if __name__ == "__main__":
    # Charger les partitions
    partition_dir = "data/output/partitions/trainValTest"
    processed_dir = "data/output/partitions/trainValTest_glove"
    glove_path = "data/input/glove.twitter.27B.200d.txt"

    # Charger le modèle GloVe
    glove_model = KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)

    # Prétraitement avec GloVe
    for file_name in os.listdir(partition_dir):
        preprocess_partition(
            os.path.join(partition_dir, file_name),
            os.path.join(processed_dir, file_name),
            lambda tweet: preprocess_with_glove(
                lemmatize_and_remove_stopwords(preprocess_tweet(tweet)),
                glove_model,
            ),
        )

