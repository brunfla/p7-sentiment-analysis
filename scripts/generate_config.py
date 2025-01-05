# scripts/generate_config.py
import hydra
from omegaconf import DictConfig, OmegaConf
import yaml
import os
import logging
import hydra.utils

# Lire la stratégie Hydra depuis une variable d'environnement
hydra_strategy = os.getenv('HYDRA_STRATEGY', 'validation-quick')

@hydra.main(config_path='./config', config_name=hydra_strategy)
def main(cfg: DictConfig):
    """
    Génère des fichiers de configuration YAML pour chaque étape du pipeline DVC.

    Args:
        cfg (DictConfig): Configuration chargée par Hydra.
    """
    # Définir les configurations pour chaque étape en convertissant les objets Hydra en dictionnaires
    configs = {
        "clean_data_config.yaml": OmegaConf.to_container(cfg.cleaner, resolve=True),
        "normalize_data_config.yaml": OmegaConf.to_container(cfg.normalizer, resolve=True),
        "vectorize_data_config.yaml": OmegaConf.to_container(cfg.vectorizer, resolve=True),
        "split_data_config.yaml": OmegaConf.to_container(cfg.partitioner, resolve=True),
        "model_config.yaml": OmegaConf.to_container(cfg.model, resolve=True),
        "train_config.yaml": OmegaConf.to_container(cfg.train, resolve=True),
        "test_config.yaml": OmegaConf.to_container(cfg.test, resolve=True),
    }

    # Utiliser le répertoire de travail original (avant que Hydra le change)
    original_cwd = hydra.utils.get_original_cwd()
    output_dir = os.path.join(original_cwd, "data/output")
    os.makedirs(output_dir, exist_ok=True)

    # Configurer le logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Écrire chaque configuration dans son fichier YAML respectif
    for filename, content in configs.items():
        with open(os.path.join(output_dir, filename), "w") as f:
            yaml.safe_dump(content, f)
        logger.info(f"Écrit {filename} avec succès.")

    logger.info("Configurations générées avec succès.")

if __name__ == "__main__":
    main()