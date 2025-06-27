from datetime import datetime
import numpy as np
from numpy.random import default_rng
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import mlflow
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
import optuna


_ENGINE_FUNCS = {
    "random_search": "_random_search",
    "hyperopt"     : "_hyperopt",
    "optuna"       : "_optuna",
}


def hyperparameter_tuning(train_X, val_X,
                          train_y, val_y,
                          preprocessor,
                          n_iterations: int = 30,
                          framework: str = "random_search",
                          experiment_name: str = "XGBoost_Hyperparameter_Tuning"):

    framework = framework.lower()
    if framework not in {"random_search", "hyperopt", "optuna"}:
        raise ValueError("framework must be 'random_search', 'hyperopt', or 'optuna'")

    # ——— dispatch ———
    if framework == "random_search":
        best_score, best_params = _random_search(
            train_X, val_X, train_y, val_y,
            preprocessor, n_iterations, experiment_name
        )
    elif framework == "hyperopt":
        best_score, best_params = _hyperopt(
            train_X, val_X, train_y, val_y,
            preprocessor, n_iterations, experiment_name
        )
    else:  # optuna
        best_score, best_params = _optuna(
            train_X, val_X, train_y, val_y,
            preprocessor, n_iterations, experiment_name
        )

    # ——— ALWAYS refit a fresh pipeline & RETURN it ———
    best_pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", xgb.XGBClassifier(
            random_state=42,
            enable_categorical=True,
            **best_params
        ))
    ]).fit(train_X, train_y)

    return best_pipe, best_params  
    

def _eval_trial(params, preprocessor,
                train_X, val_X, train_y, val_y):
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", xgb.XGBClassifier(
            random_state=42,
            enable_categorical=True,
            **params
        ))
    ]).fit(train_X, train_y)

    val_proba = pipe.predict_proba(val_X)[:, 1]
    return roc_auc_score(val_y, val_proba)


# ---- RANDOM SEARCH -------------------------------------------------
def _random_search(train_X, val_X, train_y, val_y,
                   preprocessor, n_iter, experiment):
    mlflow.set_experiment(experiment)
    best_score, best_params = -np.inf, None

    with mlflow.start_run(run_name=f"random_{datetime.now():%Y%m%d_%H%M%S}"):
        for i in range(n_iter):
            params = {                         # sample
                "n_estimators"     : np.random.randint(50, 1000),
                "learning_rate"    : np.random.uniform(0.01, 0.3),
                "max_depth"        : np.random.randint(1, 10),
                "min_child_weight" : np.random.uniform(0.1, 10),
                "subsample"        : np.random.uniform(0.5, 1.0),
                "colsample_bytree" : np.random.uniform(0.5, 1.0),
                "gamma"            : np.random.uniform(0, 10),
                "reg_alpha"        : np.random.uniform(0, 10),
                "reg_lambda"       : np.random.uniform(0, 10),
                "scale_pos_weight" : np.random.uniform(0.1, 10),
                "max_delta_step"   : np.random.randint(0, 10)
            }
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                score = _eval_trial(params, preprocessor,
                                    train_X, val_X, train_y, val_y)
                mlflow.log_metric("val_roc_auc", score)

            if score > best_score:
                best_score, best_params = score, params

        mlflow.log_metric("best_val_roc_auc", best_score)
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    return best_score, best_params

# ---- HYPEROPT --------------------------------------------------------
def _hyperopt(train_X, val_X, train_y, val_y,
              preprocessor, n_iter, experiment):

    mlflow.set_experiment(experiment)

    search_space = {
        "n_estimators"     : hp.quniform("n_estimators",     50, 1000, 1),
        "learning_rate"    : hp.uniform ("learning_rate",    0.01, 0.30),
        "max_depth"        : hp.quniform("max_depth",        1, 10, 1),
        "min_child_weight" : hp.uniform ("min_child_weight", 0.1, 10.0),
        "subsample"        : hp.uniform ("subsample",        0.5, 1.0),
        "colsample_bytree" : hp.uniform ("colsample_bytree", 0.5, 1.0),
        "gamma"            : hp.uniform ("gamma",            0.0, 10.0),
        "reg_alpha"        : hp.uniform ("reg_alpha",        0.0, 10.0),
        "reg_lambda"       : hp.uniform ("reg_lambda",       0.0, 10.0),
        "scale_pos_weight" : hp.uniform ("scale_pos_weight", 0.1, 10.0),
        "max_delta_step"   : hp.quniform("max_delta_step",   0, 10, 1),
    }

    def _objective(params):
        # cast integer-ish floats
        for k in ("n_estimators", "max_depth", "max_delta_step"):
            params[k] = int(params[k])
        auc = _eval_trial(params, preprocessor,
                          train_X, val_X, train_y, val_y)
        return {"loss": -auc, "status": STATUS_OK}

    trials = Trials()
    rng    = default_rng(42)                   
    best_params = fmin(
        fn=_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=n_iter,
        trials=trials,
        rstate=rng                           
    )

    # convert ints
    for k in ("n_estimators", "max_depth", "max_delta_step"):
        best_params[k] = int(best_params[k])

    best_score = -min(trials.losses())
    return best_score, best_params

    # ---- OPTUNA --------------------------------------------------------
def _optuna(train_X, val_X, train_y, val_y,
            preprocessor, n_iter, experiment):

    mlflow.set_experiment(experiment)

    def _objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int   ("n_estimators",     50, 1000),
            "learning_rate"    : trial.suggest_float ("learning_rate",    0.01, 0.30),
            "max_depth"        : trial.suggest_int   ("max_depth",        1, 10),
            "min_child_weight" : trial.suggest_float ("min_child_weight", 0.1, 10.0),
            "subsample"        : trial.suggest_float ("subsample",        0.5, 1.0),
            "colsample_bytree" : trial.suggest_float ("colsample_bytree", 0.5, 1.0),
            "gamma"            : trial.suggest_float ("gamma",            0.0, 10.0),
            "reg_alpha"        : trial.suggest_float ("reg_alpha",        0.0, 10.0),
            "reg_lambda"       : trial.suggest_float ("reg_lambda",       0.0, 10.0),
            "scale_pos_weight" : trial.suggest_float ("scale_pos_weight", 0.1, 10.0),
            "max_delta_step"   : trial.suggest_int   ("max_delta_step",   0, 10),
        }
        return _eval_trial(params, preprocessor,
                           train_X, val_X, train_y, val_y)

    study = optuna.create_study(direction="maximize")
    study.optimize(_objective, n_trials=n_iter, show_progress_bar=False)
    return study.best_value, study.best_params

def get_predictions_and_probabilities(model, train_X, val_X):
    """
    Returns predicted labels and probabilities for train and validation sets.
    """
    train_pred = model.predict(train_X)
    val_pred = model.predict(val_X)

    train_proba = model.predict_proba(train_X)[:, 1]
    val_proba   = model.predict_proba(val_X)[:, 1]

    return train_pred, val_pred, train_proba, val_proba