import os
import time
from flask import Flask, request, jsonify
import ktrain
import logging

# OpenCensus pour Azure
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.trace_exporter import AzureExporter
from opencensus.trace.samplers import ProbabilitySampler
from opencensus.ext.flask.flask_middleware import FlaskMiddleware

# -----------------------------------------------------------------------------
# Configuration: clé / connection string Azure
# -----------------------------------------------------------------------------
#INSTRUMENTATION_KEY = os.getenv("APPINSIGHTS_INSTRUMENTATIONKEY", "<INSTRUMENTATION_KEY>")

INSTRUMENTATION_KEY = "450f998f-167f-44b4-af57-813bbd234d16"

if INSTRUMENTATION_KEY == "<INSTRUMENTATION_KEY>":
    print("⚠️  Attention: vous n'avez pas configuré la clé Azure. Les logs ne partiront pas.")

# -----------------------------------------------------------------------------
# Création de l'application Flask
# -----------------------------------------------------------------------------
app = Flask(__name__)

# -----------------------------------------------------------------------------
# Configuration de la journalisation (logging) vers Azure
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Handler pour envoyer les logs vers Azure
azure_log_handler = AzureLogHandler(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}")
logger.addHandler(azure_log_handler)

# Trace Exporter (pour voir les traces dans Azure : Application Map, etc.)
app.config["OPENCENSUS_TRACE"] = {
    "SAMPLER": ProbabilitySampler(rate=1.0),
    "EXPORTER": AzureExporter(connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}"),
    "PROPAGATOR": "opencensus.trace.propagation.trace_context.TraceContextPropagator",
}
middleware = FlaskMiddleware(app)

# -----------------------------------------------------------------------------
# Charger le predictor ktrain au démarrage
# -----------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "final_model")  # ou un chemin absolu
predictor = ktrain.load_predictor(MODEL_PATH)

# -----------------------------------------------------------------------------
# Route 1: /predict (usage standard : tweet => prediction)
# -----------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    data = request.get_json(force=True)
    if "tweet" not in data:
        logger.warning("Requête sans tweet.")
        return jsonify({"error": "Le champ 'tweet' est requis"}), 400

    tweet = data["tweet"]

    try:
        pred_raw = predictor.predict([tweet])[0]
    except Exception as e:
        logger.error(f"Erreur de prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    # Mapper la prédiction
    if pred_raw in ["1", "label", "Positive_1", "pos"]:
        sentiment = "positif"
    else:
        sentiment = "negatif"

    duration_ms = (time.time() - start_time) * 1000

    # Logguer l'info dans Azure
    logger.info(
        "PredictOnly",
        extra={
            "custom_dimensions": {
                "tweet_length": len(tweet),
                "predicted_sentiment": sentiment,
                "response_time_ms": duration_ms
            }
        }
    )

    return jsonify({
        "tweet": tweet,
        "sentiment": sentiment,
        "latency_ms": duration_ms
    })

# -----------------------------------------------------------------------------
# Route 2: /predict/test (pour évaluation / drift analysis)
# -----------------------------------------------------------------------------
@app.route("/predict/test", methods=["POST"])
def predict_test():
    """
    Endpoint de test : on s'attend à recevoir :
    {
      "tweet": "Du texte ...",
      "true_label": "positif" ou "negatif"
    }
    On compare la prédiction avec la vérité terrain, on loggue tout dans Azure.
    """
    start_time = time.time()

    data = request.get_json(force=True)
    if "tweet" not in data or "true_label" not in data:
        logger.warning("Requête incomplète. tweet ou true_label manquant.")
        return jsonify({"error": "Champs 'tweet' et 'true_label' requis"}), 400

    tweet = data["tweet"]
    true_label = data["true_label"].lower().strip()  # normaliser "Positif", "positif", etc.

    if true_label not in ["positif", "negatif"]:
        return jsonify({"error": "Le champ 'true_label' doit être 'positif' ou 'negatif'."}), 400

    try:
        pred_raw = predictor.predict([tweet])[0]
    except Exception as e:
        logger.error(f"Erreur de prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

    # Mapper la prédiction
    if pred_raw in ["1", "label", "Positive_1", "pos"]:
        sentiment = "positif"
    else:
        sentiment = "negatif"

    duration_ms = (time.time() - start_time) * 1000

    # Savoir si c'est correct ou pas
    correctness = "correct" if sentiment == true_label else "incorrect"

    # Logger un seul message "TestPrediction" pour l'analyse
    logger.info(
        "TestPrediction",
        extra={
            "custom_dimensions": {
                "tweet_length": len(tweet),
                "predicted_sentiment": sentiment,
                "true_label": true_label,
                "correctness": correctness,
                "response_time_ms": duration_ms
            }
        }
    )

    return jsonify({
        "tweet": tweet,
        "predicted_sentiment": sentiment,
        "true_label": true_label,
        "correctness": correctness,
        "latency_ms": duration_ms
    })

# -----------------------------------------------------------------------------
# Démarrage du serveur
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
