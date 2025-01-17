from preprocess_common import preprocess_tweet, lemmatize_and_remove_stopwords

def preprocess_realtime_cleaning(tweet):
    """
    Nettoyage d'un tweet en temps réel.
    """
    return lemmatize_and_remove_stopwords(preprocess_tweet(tweet))

if __name__ == "__main__":
    example_tweets = [
        "This is a test tweet with #hashtag and @mention!",
        "Visit http://example.com for details."
    ]
    cleaned_tweets = [preprocess_realtime_cleaning(tweet) for tweet in example_tweets]
    print(cleaned_tweets)
