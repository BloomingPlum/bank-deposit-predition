import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from mlflow.models.signature import infer_signature
from utils.eval import plot_confusion_matrix, _plot_single_roc


def run_experiment(
        model,                      # any fitted or unfitted estimator / pipeline
        train_X, train_y,
        val_X,   val_y,
        run_name: str,
        registered_model_name: str | None = None,
        log_roc: bool = True,
        log_cm:  bool = True):
    """
    Train (if necessary), evaluate, and log EVERYTHING to MLflow
    ------------------------------------------------------------------
    Returns a dict of main metrics for quick inspection.
    """

    with mlflow.start_run(run_name=run_name):
        # -- fit if the model isn't trained yet -----------------------
        if not hasattr(model, "classes_"):
            model.fit(train_X, train_y)

        # -- Params ---------------------------------------------------
        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        # -- Probabilities & labels -----------------------------------
        train_proba = model.predict_proba(train_X)[:, 1]
        val_proba   = model.predict_proba(val_X)[:, 1]

        train_pred  = (train_proba >= 0.5).astype(int)  # faster than predict()
        val_pred    = (val_proba   >= 0.5).astype(int)

        # -- Basic metrics -------------------------------------------
        metrics = {
            "accuracy_train":  accuracy_score(train_y, train_pred),
            "accuracy_val":    accuracy_score(val_y,   val_pred),
            "precision_train": precision_score(train_y, train_pred),
            "precision_val":   precision_score(val_y,   val_pred),
            "recall_train":    recall_score(train_y, train_pred),
            "recall_val":      recall_score(val_y,   val_pred),
            "f1_train":        f1_score(train_y, train_pred),
            "f1_val":          f1_score(val_y,   val_pred),
            "roc_auc_train":   roc_auc_score(train_y, train_proba),
            "roc_auc_val":     roc_auc_score(val_y,   val_proba),
        }
        mlflow.log_metrics(metrics)

        # -- Confusion matrices --------------------------------------
        if log_cm:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            plot_confusion_matrix(train_y, train_pred, "Training", ax1)
            plot_confusion_matrix(val_y,   val_pred,   "Validation", ax2)
            fig.suptitle("Confusion Matrices", fontsize=14)
            plt.tight_layout()
            mlflow.log_figure(fig, "plots/confusion_matrices.png")
            plt.close(fig)

        # -- ROC curves ----------------------------------------------
        if log_roc:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            _plot_single_roc(ax1, train_y, train_proba, "Training")
            _plot_single_roc(ax2, val_y,   val_proba,   "Validation")
            fig.suptitle("ROC Curves", fontsize=14)
            plt.tight_layout()
            mlflow.log_figure(fig, "plots/roc_curves.png")
            plt.close(fig)

        # -- Classification reports ----------------------------------
        mlflow.log_text(
            classification_report(train_y, train_pred, digits=3),
            "reports/classification_report_train.txt")
        mlflow.log_text(
            classification_report(val_y, val_pred, digits=3),
            "reports/classification_report_val.txt")

        # -- Model + signature ---------------------------------------
        signature = infer_signature(val_X, val_pred)
        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            registered_model_name=registered_model_name)

    return metrics, train_pred, val_pred, train_proba, val_proba