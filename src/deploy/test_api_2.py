import requests
import pandas as pd
import time

# --------------------------------------------------------
# 1. Charger x_test.csv et y_test.csv puis fusionner
# --------------------------------------------------------
x_test = pd.read_csv("data/output/preprocessed/clean_tweet_glove/x_test.csv")  # contient: id, feature (le tweet)
y_test = pd.read_csv("data/output/preprocessed/clean_tweet_glove/y_test.csv")  # contient: id, label (réel)

test_data = x_test.merge(y_test, on="id")

# Pour l'exemple, on prend 100 tweets, sinon tu peux enlever la ligne suivante
test_data = test_data.head(100)

# --------------------------------------------------------
# 2. Définir l'URL de l'endpoint de test
# --------------------------------------------------------
url = "http://localhost:8000/predict/test"  
# Adapte le port si besoin ou l'adresse si c'est sur un autre serveur

# --------------------------------------------------------
# 3. (Optionnel) Fonction de mapping label -> string
#    Si tes labels sont déjà "positif"/"negatif", enlève cette partie.
# --------------------------------------------------------
def map_label_to_text(label_value):
    """
    Convertit un label (par ex. 0, 1) en texte ('negatif', 'positif').
    Adapte si ton dataset stocke autrement (ex: 'negative', 'positive').
    """
    if label_value in [1, "1", "positive", "pos"]:
        return "positif"
    else:
        return "negatif"

# --------------------------------------------------------
# 4. Envoyer chaque tweet et label réel à l'API
# --------------------------------------------------------
responses = []

for i, row in test_data.iterrows():
    tweet_id = row["id"]
    tweet = row["feature"]
    real_label_raw = row["label"]  # selon comment c'est stocké
    real_label_str = map_label_to_text(real_label_raw)

    data = {
        "tweet": tweet,
        "true_label": real_label_str
    }

    try:
        response = requests.post(url, json=data)
    except Exception as e:
        print(f"❌ Erreur de requête pour id={tweet_id}: {e}")
        continue

    if response.status_code == 200:
        response_json = response.json()
        predicted_sentiment = response_json.get("predicted_sentiment", "")
        correctness = response_json.get("correctness", "")
        latency_ms = response_json.get("latency_ms", -1)

        responses.append({
            "id": tweet_id,
            "tweet": tweet,
            "true_label": real_label_str,
            "predicted_sentiment": predicted_sentiment,
            "correctness": correctness,
            "latency_ms": latency_ms
        })

        print(f"✅ Tweet {tweet_id} => Prédit: {predicted_sentiment} | Vrai: {real_label_str} | Correctness: {correctness}")
    else:
        print(f"❌ Erreur {response.status_code} pour id={tweet_id}: {response.text}")

    # Petit sleep pour éviter de spammer l'API
    time.sleep(0.5)

# --------------------------------------------------------
# 5. Analyser les résultats (accuracy, etc.)
# --------------------------------------------------------
if len(responses) > 0:
    correct_predictions = sum(1 for r in responses if r["correctness"] == "correct")
    accuracy = correct_predictions / len(responses) * 100
    print(f"\n🎯 Accuracy sur {len(responses)} tweets: {accuracy:.2f}%")
else:
    print("Aucune réponse valide n'a été reçue.")

