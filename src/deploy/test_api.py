import pandas as pd
import requests
import time

# Charger les données
x_test = pd.read_csv("x_test.csv")
y_test = pd.read_csv("y_test.csv")

#print(x_test.columns)
#print(y_test.columns)

# Fusionner les deux fichiers sur la clé 'id'
test_data = x_test.merge(y_test, on="id")
test_data = test_data.head(100)

# URL de l'API
url = "http://localhost:8000/predict"  # Adapter l'URL selon ton API

responses = []

# Envoyer chaque tweet à l'API
for i, row in test_data.iterrows():    
    tweet_id = row["id"]
    tweet = row["feature"]
    real_label = row["label"]

    data = {"tweet": tweet}
    response = requests.post(url, json=data)

    if response.status_code == 200:
        response_json = response.json()
        predicted_sentiment = response_json.get("sentiment", "")

        predicted_label = 0 if predicted_sentiment == "negatif" else 1
        responses.append((tweet_id, predicted_label))

        print(f"✅ Tweet {tweet_id} traité avec succès. Prédiction: {predicted_label}")
    else:
        print(f"❌ Erreur sur la requête {tweet_id}, code {response.status_code}: {response.text}")

    time.sleep(1)

# Convertir en DataFrame pour évaluation
df_predictions = pd.DataFrame(responses, columns=["id", "predicted_label"])

# Fusionner avec les labels réels pour comparaison
df_results = test_data[["id", "label"]].merge(df_predictions, on="id")

# Vérifier que toutes les réponses ont bien été obtenues
assert len(df_results) == len(test_data), "❌ Erreur : certaines prédictions sont manquantes."

# Calculer le taux de réussite
correct_predictions = sum(df_results["label"] == df_results["predicted_label"])
accuracy = correct_predictions / len(df_results) * 100

print(f"🎯 Précision du modèle : {accuracy:.2f}%")

