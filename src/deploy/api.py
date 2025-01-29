import os
from flask import Flask, request, jsonify
import tensorflow
import ktrain

# 1) Créer l'application Flask
app = Flask(__name__)

# 2) Charger le predictor ktrain au démarrage
#    Indiquez le chemin vers le dossier où se trouve votre "final_model"
MODEL_PATH = os.getenv("MODEL_PATH", "final_model")  # ou un chemin absolu
predictor = ktrain.load_predictor(MODEL_PATH)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint HTTP POST qui prend un tweet en JSON
    et renvoie si c'est positif ou négatif.
    Format attendu: { "tweet": "..." }
    """
    data = request.get_json(force=True)

    if "tweet" not in data:
        return jsonify({"error": "Le champ 'tweet' est requis"}), 400

    tweet = data["tweet"]
    # 3) Effectuer la prédiction
    #    Selon votre entraînement, predictor.predict(...) peut renvoyer "0"/"1" ou "negative"/"positive", etc.
    #    Adaptez la logique selon votre cas exact.
    pred = predictor.predict([tweet])[0]

    # Exemple: si le modèle renvoie "label", on l'interprète comme "1"
    #          sinon "0" (vous pouvez faire plus propre selon votre mapping de classes)
    # ou si vous avez un mapping direct: "0" => Négatif, "1" => Positif, etc.
    if pred in ["1", "label", "Positive_1"]:
        sentiment = "positif"
    else:
        sentiment = "negatif"

    # 4) Retourner la réponse au format JSON
    return jsonify({"tweet": tweet, "sentiment": sentiment})


if __name__ == "__main__":
    # Lancer le serveur en mode debug ou normal
    # Choisir le port qui vous convient, par ex. 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
