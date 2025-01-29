import psutil
import logging

# ------------------------------------------------
# Fonction pour afficher les métriques système
# ------------------------------------------------

def get_logger(name):
    """
    Configure et retourne un logger.

    Args:
        name (str): Nom du logger.

    Returns:
        logging.Logger: Logger configuré.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_system_metrics(logger):
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    logger.info(f"CPU Usage: {cpu_usage}%")
    logger.info(
        f"Memory Usage: {memory.percent}% "
        f"(Total: {memory.total / (1024**3):.2f} GB, "
        f"Available: {memory.available / (1024**3):.2f} GB)"
    )
