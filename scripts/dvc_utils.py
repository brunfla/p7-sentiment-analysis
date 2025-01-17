import subprocess
import json

def get_stage_dependencies(stage_name):
    """
    Obtenir la liste des dépendances d'un stage DVC donné.

    Args:
        stage_name (str): Nom du stage DVC.

    Returns:
        list: Liste des chemins des dépendances pour le stage donné.
    """
    try:
        # Exécution de la commande DVC pour obtenir les détails du stage
        result = subprocess.run(
            ["dvc", "stages", "show", stage_name, "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        # Décoder la sortie JSON
        stage_info = json.loads(result.stdout)

        if stage_name not in stage_info:
            print(f"Stage '{stage_name}' introuvable.")
            return []

        # Extraire les dépendances
        dependencies = stage_info[stage_name].get("deps", [])
        return [dep["path"] for dep in dependencies]

    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution de DVC pour le stage '{stage_name}': {e.stderr}")
        return []
    except json.JSONDecodeError:
        print(f"Erreur lors du parsing JSON pour le stage '{stage_name}'.")
        return []
    except Exception as e:
        print(f"Erreur inattendue : {e}")
        return []

