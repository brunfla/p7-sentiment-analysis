import optuna
from sklearn.model_selection import GridSearchCV
import logging

logger = logging.getLogger(__name__)

# ------------------------------------------------
# Appliquer GridSearchCV
# ------------------------------------------------
def apply_grid_search(model, tuning_parameters, X_train, y_train):
    logger.info("Démarrage de GridSearchCV...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=tuning_parameters.paramGrid,
        cv=tuning_parameters.crossValidationFolds,
        verbose=tuning_parameters.verbosityLevel,
        n_jobs=tuning_parameters.parallelJobs
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Meilleurs paramètres : {grid_search.best_params_}")
    return grid_search.best_estimator_

# ------------------------------------------------
# Appliquer Optuna
# ------------------------------------------------
def apply_optuna(train_func, tuning_parameters, X_train, y_train, X_val, y_val):
    logger.info("Démarrage d'Optuna...")

    def objective(trial):
        params = {key: trial.suggest_categorical(key, values) for key, values in tuning_parameters.paramGrid.items()}
        return train_func(params, X_train, y_train, X_val, y_val)

    study = optuna.create_study(direction=tuning_parameters.optimizationDirection)
    study.optimize(objective, n_trials=tuning_parameters.trialCount, timeout=tuning_parameters.timeLimitSeconds)
    logger.info(f"Meilleurs hyperparamètres : {study.best_params}")
    return study.best_params

# ------------------------------------------------
# Gérer le type de tuning
# ------------------------------------------------
def apply_tuning(tuning_type_cfg, tuning_parameters_cfg, model, train_func, X_train, y_train, X_val, y_val):
    if tuning_type_cfg.type == "gridSearchCV":
        return apply_grid_search(model, tuning_parameters_cfg, X_train, y_train)
    elif tuning_type_cfg.type == "optuna":
        return apply_optuna(train_func, tuning_parameters_cfg, X_train, y_train, X_val, y_val)
    elif tuning_type_cfg.type == "none":
        logger.info("Pas de tuning sélectionné.")
        return model
    else:
        raise ValueError(f"Type de tuning non supporté : {tuning_type_cfg.type}")
