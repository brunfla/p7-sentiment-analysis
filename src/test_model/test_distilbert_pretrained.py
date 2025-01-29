import os
import json
import pandas as pd
import logging
from transformers import pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_distilbert(test_file, output_file, metrics=["accuracy", "precision", "recall", "f1"]):
    """
    Évalue DistilBERT pré-entraîné sur un ensemble de test.
    """
    # Charger les données
    logger.info(f"Chargement des données de test depuis {test_file}")
    test_data = pd.read_csv(test_file)

    if "tweet" not in test_data.columns or "id" not in test_data.columns:
        raise ValueError(f"Le fichier {test_file} doit contenir les colonnes 'tweet' et 'id'.")

    # Validation des données
    logger.info("Validation des données de test...")
    test_data = test_data.dropna(subset=["tweet", "id"])
    test_data = test_data[test_data["tweet"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
    logger.info(f"Données après validation : {len(test_data)} lignes.")

    texts = test_data["tweet"].tolist()
    labels = test_data["id"].tolist()

    # Charger le pipeline de classification
    logger.info("Chargement du pipeline DistilBERT...")
    classifier = pipeline("text-classification", model="distilbert-base-uncased", return_all_scores=False)

    # Prédictions
    logger.info("Génération des prédictions avec DistilBERT pré-entraîné...")
    predictions = []
    for text in texts:
        try:
            pred = classifier(text, truncation=True)[0]  # Utiliser uniquement le premier résultat
            predictions.append(pred)
        except Exception as e:
            logger.warning(f"Erreur lors de la prédiction pour le tweet : {text}. Erreur : {e}")
            predictions.append({"label": "LABEL_0", "score": 0})  # Valeur par défaut

    y_pred = [1 if pred["label"] == "LABEL_1" else 0 for pred in predictions]

    # Calcul des métriques
    logger.info("Calcul des métriques...")
    results = {}
    if "accuracy" in metrics:
        results["accuracy"] = accuracy_score(labels, y_pred)
    if "precision" in metrics:
        results["precision"] = precision_score(labels, y_pred)
    if "recall" in metrics:
        results["recall"] = recall_score(labels, y_pred)
    if "f1" in metrics:
        results["f1"] = f1_score(labels, y_pred)

    # Sauvegarder les résultats
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    logger.info(f"Métriques sauvegardées dans {output_file}")

if __name__ == "__main__":
    test_file = "data/output/partitions/trainvaltest/glove_cleaned/test.csv"
    output_file = "data/output/models/distilbert_pretrained/metrics.json"
    evaluate_distilbert(test_file, output_file)
