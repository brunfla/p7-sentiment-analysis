import argparse
import pickle
from sklearn.model_selection import train_test_split, KFold

def main(input_path, output_path, target, test_size=0.2, val_size=0.1, random_seed=42, folds=5):
    # Charger les données vectorisées
    with open(input_path, "rb") as f:
        X, y = pickle.load(f)

    if target == "trainValTest":
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=val_size, random_state=random_seed)
        with open(output_path, "wb") as f:
            pickle.dump((X_train, y_train, X_val, y_val, X_test, y_test), f)
    elif target == "trainTest":
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
        with open(output_path, "wb") as f:
            pickle.dump((X_train, y_train, X_test, y_test), f)
    elif target == "crossValidation":
        kfold = KFold(n_splits=folds, shuffle=True, random_state=random_seed)
        folds_data = list(kfold.split(X, y))
        with open(output_path, "wb") as f:
            pickle.dump((folds_data, X, y), f)
    else:
        raise ValueError(f"Partition target non reconnue : {target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier vectorisé d'entrée.")
    parser.add_argument("--output", type=str, required=True, help="Chemin pour sauvegarder le fichier partitionné.")
    parser.add_argument("--target", type=str, required=True, choices=["trainValTest", "trainTest", "crossValidation"], help="Type de partition.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Taille de la partition test.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Taille de la partition validation (pour trainValTest).")
    parser.add_argument("--random-seed", type=int, default=42, help="Seed aléatoire.")
    parser.add_argument("--folds", type=int, default=5, help="Nombre de folds pour cross-validation.")
    args = parser.parse_args()

    main(args.input, args.output, args.target, args.test_size, args.val_size, args.random_seed, args.folds)

