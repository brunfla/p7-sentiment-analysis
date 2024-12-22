#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import re
import nltk

# Chargement du fichier avec un encodage différent
df = pd.read_csv("../data/input/training.1600000.processed.noemoticon.utf-8.csv", 
                 header=None, 
                 names=["id", "timestamp", "date", "query", "user", "tweet"], 
                 )

# Supprimer les tweets vides (NaN ou chaînes vides)
df = df[~(df['tweet'].isna() | (df['tweet'] == ""))]

df['tweet'] = df['tweet'].str.lower()

# Suppression des URLs, mentions et hashtags
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'http\S+|www\S+|https\S+', '', x))  # Supprimer les URLs
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'@\w+', '', x))  # Supprimer les mentions
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'#\w+', '', x))  # Supprimer les hashtags

df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Supprimer la ponctuation
df['tweet'] = df['tweet'].apply(lambda x: re.sub(r'\s+', ' ', x))  # Supprimer les espaces multiples
df['tweet'] = df['tweet'].apply(lambda x: x.strip())  # Supprimer les espaces en début et fin de chaîne


from nltk.tokenize import word_tokenize

# Télécharger les ressources nécessaires (uniquement la première fois)
nltk.download('punkt')

# Tokeniser les tweets
df['tweet_tokenized'] = df['tweet'].apply(lambda x: word_tokenize(x))


# In[6]:


from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Si tes tweets sont en français

# Filtrer les stopwords
df['tweet_tokenized'] = df['tweet_tokenized'].apply(lambda x: [word for word in x if word not in stop_words])


# In[7]:

df.to_csv("../data/output/data_clean.csv", index=False)

