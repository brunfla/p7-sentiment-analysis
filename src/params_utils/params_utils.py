import os
import yaml

def load_params(params_file, section):
    """
    Charge les paramètres d'une section donnée depuis le fichier params.yaml.

    Args:
        params_file (str): Chemin vers le fichier params.yaml.
        section (str): Section des paramètres à charger.

    Returns:
        dict: Paramètres de la section.

    Raises:
        KeyError: Si la section ou une clé obligatoire est manquante.
        FileNotFoundError: Si le fichier params.yaml est introuvable.
    """
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"Le fichier des paramètres '{params_file}' est introuvable.")

    with open(params_file, "r") as file:
        params = yaml.safe_load(file)

    if section not in params:
        raise KeyError(f"La section '{section}' est absente dans '{params_file}'.")

    return params[section]
