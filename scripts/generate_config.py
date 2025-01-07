import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import os
import logging
import hydra.utils

# Lire la stratégie Hydra depuis une variable d'environnement
hydra_strategy = os.getenv("HYDRA_STRATEGY", "baseline")

@hydra.main(config_path="./config", config_name=hydra_strategy)
def main(cfg: DictConfig):
    """
    Génère des fichiers de configuration YAML pour chaque étape du pipeline DVC.

    Args:
        cfg (DictConfig): Configuration chargée par Hydra.
    """
    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Définir les configurations pour chaque étape
    configs = {
        "clean_data_config.yaml": OmegaConf.to_container(cfg.cleaner, resolve=True),
        "normalize_data_config.yaml": OmegaConf.to_container(cfg.normalizer, resolve=True),
        "vectorize_data_config.yaml": OmegaConf.to_container(cfg.vectorizer, resolve=True),
        "split_data_config.yaml": OmegaConf.to_container(cfg.partitioner, resolve=True),
        "model_config.yaml": OmegaConf.to_container(cfg.model, resolve=True),
        "train_config.yaml": OmegaConf.to_container(cfg.training, resolve=True),
        "tuning_config.yaml": {
            "tuning": OmegaConf.to_container(cfg.tuning, resolve=True),
            "type": OmegaConf.to_container(cfg.tuningType, resolve=True),
        },
        "test_config.yaml": OmegaConf.to_container(cfg.test, resolve=True),
    }

    # Utiliser le répertoire de travail original (avant que Hydra le change)
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = os.path.join(original_cwd, "data/output")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Répertoire de sortie pour les configurations : {output_dir}")

    # Écrire chaque configuration dans son fichier YAML respectif
    for filename, content in configs.items():
        file_path = os.path.join(output_dir, filename)
        with open(file_path, "w") as f:
            yaml.safe_dump(content, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Écrit {filename} avec succès dans {file_path}.")

    logger.info("Toutes les configurations ont été générées avec succès.")

if __name__ == "__main__":
    main()
