from flask import Flask, request, jsonify
import onnxruntime as ort

# Initialiser Flask
app = Flask(__name__)

# Charger le modèle ONNX
ort_session = ort.InferenceSession('logreg_model.onnx')  # Initialiser ONNX Runtime avec le modèle

# Route pour prédire le sentiment d'un tweet
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraire le tweet de la requête JSON
        tweet = request.json.get('tweet', '')

        # Vérifier que le tweet est fourni
        if not tweet:
            return jsonify({'error': 'Le champ "tweet" est requis.'}), 400

        # Passer le texte brut directement au modèle ONNX
        inputs = {ort_session.get_inputs()[0].name: [[tweet]]}  # Double crochets pour respecter le format attendu
        outputs = ort_session.run(None, inputs)

        # Extraire le label prédit (0 pour négatif, 1 pour positif)
        predicted_label = outputs[0][0]
        sentiment = 'positive' if predicted_label == 1 else 'negative'

        return jsonify({'sentiment': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Démarrer l'application
if __name__ == '__main__':
    app.run(debug=True, port=8000)

