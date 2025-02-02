import os
import json
import pandas as pd
import logging
from tabulate import tabulate

# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_metrics(file_path):
    """
    Charger les métriques à partir d'un fichier JSON.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Fichier des métriques introuvable : {file_path}")
        return None

    with open(file_path, "r") as f:
        metrics = json.load(f)

    logger.info(f"Métriques chargées depuis : {file_path}")
    return metrics

def compare_models(metrics_files, output_path):
    """
    Comparer les performances de plusieurs modèles et générer un rapport.
    
    Args:
        metrics_files (dict): Dictionnaire avec les noms des modèles comme clés et leurs fichiers métriques comme valeurs.
        output_path (str): Chemin pour sauvegarder le rapport de comparaison.
    """
    comparison_data = []

    # Charger les métriques de chaque modèle
    for model_name, file_path in metrics_files.items():
        metrics = load_metrics(file_path)
        if metrics is not None:
            row = {"Model": model_name}
            row.update(metrics)
            comparison_data.append(row)

    # Créer un DataFrame pour organiser les données
    df = pd.DataFrame(comparison_data)

    # Trier les modèles par F1-score si disponible, sinon par précision
    sort_by = "f1" if "f1" in df.columns else "accuracy"
    df = df.sort_values(by=sort_by, ascending=False)

    # Sauvegarder le tableau en JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient="records", indent=4)
    logger.info(f"Rapport de comparaison sauvegardé dans : {output_path}")

    # Afficher un tableau lisible dans les logs
    logger.info("Résultats de la comparaison :\n")
    logger.info("\n" + tabulate(df, headers="keys", tablefmt="pretty"))

if __name__ == "__main__":
    # Configuration
    metrics_files = {
        "Logistic Regression (TF-IDF)": "data/output/models/logistic_tfidf/metrics.json",
        "Logistic Regression (TF-IDF + GloVe)": "data/output/models/logistic_tfidf_glove/metrics.json",
        "LSTM (GloVe)": "data/output/models/lstm_with_glove/metrics.json",
        "BERT (GloVe)": "data/output/models/bert_with_glove/metrics.json",
        "DistilBERT (Pretrained)": "data/output/models/distilbert_pretrained/metrics.json",
    }
    output_path = "data/output/reports/comparison.json"

    # Comparer les modèles
    compare_models(metrics_files, output_path)

