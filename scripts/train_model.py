# ------------------------------------------------
# Orchestration principale
# ------------------------------------------------
def main():
    config_path = os.getenv('HYDRA_CONFIG_PATH', './config')
    strategy = os.getenv('HYDRA_STRATEGY', 'baseline')

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    initialize(config_path=config_path, version_base=None)

    # Charger les configurations principales
    cfg = compose(config_name=strategy)

    # Charger les configurations spécifiques
    model_cfg = compose(config_name=cfg.model)
    training_cfg = compose(config_name=cfg.training)
    tuning_cfg = {
        "type": compose(config_name=cfg.tuningType),
        "parameters": compose(config_name=cfg.tuning),
    }

    logger.info(f"Stratégie sélectionnée : {strategy}")

    # Charger les données
    data = load_partitioned_data("data/output/train_ready_data.pkl", cfg.partitioner._target_)

    # Entraîner le modèle en fonction du type
    if model_cfg.type == "logistic_regression":
        model = train_logistic_regression(model_cfg, tuning_cfg['type'], tuning_cfg['parameters'], data)
    elif model_cfg.type == "bidirectional_lstm":
        model = train_lstm(model_cfg, training_cfg, tuning_cfg['type'], tuning_cfg['parameters'], data)
    else:
        raise ValueError(f"Type de modèle non supporté : {model_cfg.type}")

    # Sauvegarder le modèle dans MLflow
    mlflow.log_params(model_cfg.parameters)
    mlflow.sklearn.log_model(model, artifact_path="model")
    logger.info("Modèle enregistré dans MLflow.")

    # Sauvegarder le run ID dans MLflow
    run_id_file = "data/output/mlflow_id.json"
    mlflow_run_id = mlflow.active_run().info.run_id
    with open(run_id_file, "w") as f:
        json.dump({"id": mlflow_run_id}, f)
    logger.info(f"MLflow run_id sauvegardé dans {run_id_file}")


if __name__ == "__main__":
    main()