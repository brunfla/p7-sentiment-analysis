import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def load_params(params_file, section="data_plots"):
    """
    Lit la section data_plots de params.yaml pour connaître
    les chemins vers x_train.csv, x_val.csv, x_test.csv.
    """
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
    return params[section]

def load_dataset(csv_path):
    """
    Charge un dataset CSV ayant au moins les colonnes:
    - id
    - feature (le tweet)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def plot_word_distribution(df, title, output_dir):
    """
    Calcule la distribution du nombre de mots, affiche quelques stats,
    et génère un histogramme.

    df: DataFrame contenant la colonne 'feature'
    title: titre pour le graphique
    output_dir: répertoire où sauvegarder l'histogramme
    """
    # Calcul du nombre de mots par tweet
    df['num_words'] = df['feature'].apply(lambda x: len(x.split()))

    # Affiche un résumé statistique
    stats = df['num_words'].describe()
    print(f"\n--- Distribution du nb de mots pour {title} ---")
    print(stats)

    # Plot histogramme
    plt.figure(figsize=(8,5))
    sns.histplot(df['num_words'], bins=30, kde=True, color='blue')
    plt.title(f"Distribution du nombre de mots - {title}")
    plt.xlabel("Nombre de mots par tweet")
    plt.ylabel("Fréquence")

    # Sauvegarde du plot
    hist_path = os.path.join(output_dir, f"{title}_word_distribution.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"Histogramme sauvegardé: {hist_path}")

def compute_vocabulary_size(df):
    """
    Calcule la taille du vocabulaire,
    c'est-à-dire le nombre de mots uniques dans 'feature'.
    """
    vocab = set()
    for text in df['feature']:
        words = text.split()
        vocab.update(words)
    return len(vocab)

def plot_vocabulary_sizes(vocab_sizes, output_dir):
    """
    Reçoit un dict: { 'Train': 1234, 'Val': 567, 'Test': 890 }
    et génère un diagramme en barres comparant les tailles de vocabulaires.
    """
    plt.figure(figsize=(6,4))
    datasets = list(vocab_sizes.keys())
    sizes = list(vocab_sizes.values())

    sns.barplot(x=datasets, y=sizes, palette='Blues_r')
    plt.title("Comparaison de la taille du vocabulaire")
    plt.ylabel("Nombre de mots uniques")

    vocab_plot_path = os.path.join(output_dir, "vocab_size_comparison.png")
    plt.savefig(vocab_plot_path)
    plt.close()
    print(f"Diagramme de la taille des vocabulaires: {vocab_plot_path}")

def main():
    # 1) Chargement des paramètres
    params = load_params("params.yaml", section=sys.argv[1])
    input_dir = params["input_dir"]
    files = params["input_files"]  # ex: ["x_train.csv","x_val.csv","x_test.csv"]
    output_dir =  params["output_dir"]

    # 2) Chargement des CSV
    train_df = load_dataset(os.path.join(input_dir, files[0]))
    val_df   = load_dataset(os.path.join(input_dir, files[1]))
    test_df  = load_dataset(os.path.join(input_dir, files[2]))

    # 3) Création d'un dossier de sortie si besoin
    os.makedirs(output_dir, exist_ok=True)

    # 4) Distribution du nb de mots (histogrammes + stats)
    plot_word_distribution(train_df, "Train", output_dir)
    plot_word_distribution(val_df, "Validation", output_dir)
    plot_word_distribution(test_df, "Test", output_dir)

    # 5) Calcul et plot des tailles de vocabulaire
    vocab_train = compute_vocabulary_size(train_df)
    vocab_val   = compute_vocabulary_size(val_df)
    vocab_test  = compute_vocabulary_size(test_df)

    print(f"\nTaille vocabulaire TRAIN = {vocab_train}")
    print(f"Taille vocabulaire VAL   = {vocab_val}")
    print(f"Taille vocabulaire TEST  = {vocab_test}")

    vocab_sizes = {
        "Train": vocab_train,
        "Val":   vocab_val,
        "Test":  vocab_test
    }
    plot_vocabulary_sizes(vocab_sizes, output_dir)

if __name__ == "__main__":
    main()

