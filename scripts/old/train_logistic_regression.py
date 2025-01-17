import mlflow
import mlflow.sklearn
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def train_logistic_regression(cfg, tuning_cfg, mlflow_cfg, partitioned_data):
    """
    Entraîne un modèle de régression logistique avec des données encapsulées dans PartitionedData.
    """
    X_train, y_train = partitioned_data.data["X_train"], partitioned_data.data["y_train"]
    X_val, y_val = partitioned_data.data.get("X_val"), partitioned_data.data.get("y_val")

    logger.info(f"Classes dans y_train : {set(y_train)}")
    if y_val is not None:
        logger.info(f"Classes dans y_val : {set(y_val)}")
    else:
        logger.info("y_val est None. Pas de données de validation disponibles.")

    model = LogisticRegression(**cfg.parameters)

    mlflow.set_tracking_uri(mlflow_cfg.trackingUri)
    mlflow.set_experiment(mlflow_cfg.experiment.name)

    run_name = mlflow_cfg.experiment.run.get("name", None)
    run_description = mlflow_cfg.experiment.run.get("description", None)
    run_tags = mlflow_cfg.experiment.run.get("tags", {})

    with mlflow.start_run(run_name=run_name, description=run_description) as run:
        if run_tags:
            mlflow.set_tags(run_tags)

        if tuning_cfg.get("type") == "grid_search":
            logger.info("Tuning des hyperparamètres activé avec GridSearchCV.")
            param_grid = dict(tuning_cfg.get('paramGrid', {}))
            
            logger.info("Initialisation de GridSearchCV...")
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=tuning_cfg.get('cv', 5),
                verbose=3,
                n_jobs=-1
            )
            logger.info("Démarrage du fit de GridSearchCV (peut être long) ...")
            grid_search.fit(X_train, y_train)
            logger.info("GridSearchCV terminé.")

            model = grid_search.best_estimator_
            mlflow.log_params(grid_search.best_params_)

            # Logger tous les résultats (optionnel)
            # ... (idem code précédent)

        else:
            logger.info("Tuning désactivé. Entraînement avec les paramètres par défaut.")
            model.fit(X_train, y_train)

        # Évaluation train
        train_score = model.score(X_train, y_train)
        mlflow.log_metric("train_accuracy", train_score)
        logger.info(f"Score sur l'ensemble d'entraînement : {train_score}")

        # Évaluation val
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            mlflow.log_metric("val_accuracy", val_score)
            logger.info(f"Score sur l'ensemble de validation : {val_score}")

        # log model
        mlflow.sklearn.log_model(model, "logistic_regression_model")

    return model

