import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow  # only needed for logging in create_and_log_roc_curves

from typing import Tuple
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

def plot_confusion_matrix(y_true, y_pred, dataset_name="Dataset", ax=None, 
                                   figsize=(5, 4), class_names=None, cmap="Blues", annot_fontsize=10):
    """
    Plot an enhanced confusion matrix with both counts and percentages.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    dataset_name : str, optional
        Name of the dataset (e.g., "Training", "Validation").
    ax : matplotlib axis, optional
        Axis to plot on. If None, a new figure and axis will be created.
    figsize : tuple, optional
        Figure size (only used if ax is None). Default is (5, 4).
    class_names : list, optional
        List of class labels to use for the tick labels. If None, defaults to sorted unique labels.
    cmap : str, optional
        Colormap for the heatmap. Default is "RdPu".
    annot_fontsize : int, optional
        Font size for the annotation text. Default is 10.
    
    Returns:
    --------
    f1 : float
        F1 score for the predictions.
    fig : matplotlib.figure.Figure (optional)
        The figure object, returned only if a new figure was created.
    """
    # Calculate confusion matrix and percentages
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm)
    if cm_sum == 0:
        cm_sum = 1  # Prevent division by zero
    cm_percentages = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    # Compute F1 score
    f1 = f1_score(y_true, y_pred)
    
    # Create annotation text with both percentage and count
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm_percentages[i, j]:.3f}\n({cm[i, j]})"
    
    # Determine class names if not provided
    if class_names is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = [str(cls) for cls in sorted(classes)]
    
    # Create a new figure if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    
    # Plot the heatmap
    sns.heatmap(cm_percentages, annot=annot, fmt="", cmap=cmap, cbar=True,
                ax=ax, vmin=0, vmax=1, annot_kws={"fontsize": annot_fontsize}, square=True)
    
    # Set titles and labels
    ax.set_title(f"{dataset_name} Confusion Matrix\nF1 Score: {f1:.3f}", fontsize=annot_fontsize+2)
    ax.set_xlabel("Predicted", fontsize=annot_fontsize)
    ax.set_ylabel("Actual", fontsize=annot_fontsize)
    ax.set_xticklabels(class_names, fontsize=annot_fontsize)
    ax.set_yticklabels(class_names, fontsize=annot_fontsize, rotation=0)
    
    if fig is not None:
        fig.tight_layout()
        return f1, fig
    else:
        return f1


def display_confusion_matrices(
    train_y, train_pred,
    val_y,   val_pred,
    figsize: Tuple[int, int] = (12, 5),
    **plot_kwargs,                    # forwarded to plot_confusion_matrix
):
    """
    Display training & validation confusion matrices side-by-side.

    Parameters
    ----------
    train_y, train_pred : array-like
        True and predicted labels for the training set.
    val_y, val_pred : array-like
        True and predicted labels for the validation set.
    figsize : tuple(int, int), default (12, 5)
        Figure width and height in inches.
    plot_kwargs :
        Extra keyword arguments passed straight to your
        `plot_confusion_matrix` function (e.g. cmap, class_names…).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_confusion_matrix(train_y, train_pred,
                          "Training", ax=axes[0], **plot_kwargs)
    plot_confusion_matrix(val_y,   val_pred,
                          "Validation", ax=axes[1], **plot_kwargs)

    plt.tight_layout()
    plt.show()
    plt.close(fig)   

def print_metrics(train_targets, train_preds, val_targets, val_preds, train_probs, val_probs):
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score

    print("Train vs Validation\n")
    print(f"{'Metric':<12} {'Train':>10} {'Validation':>15}")
    print("-" * 40)
    print(f"{'F1 Score':<12} {f1_score(train_targets, train_preds):>10.3f} {f1_score(val_targets, val_preds):>15.3f}")
    print(f"{'Precision':<12} {precision_score(train_targets, train_preds):>10.3f} {precision_score(val_targets, val_preds):>15.3f}")
    print(f"{'Recall':<12} {recall_score(train_targets, train_preds):>10.3f} {recall_score(val_targets, val_preds):>15.3f}")
    print(f"{'Accuracy':<12} {accuracy_score(train_targets, train_preds):>10.3f} {accuracy_score(val_targets, val_preds):>15.3f}")
    print(f"{'ROC AUC':<12} {roc_auc_score(train_targets, train_probs):>10.3f} {roc_auc_score(val_targets, val_probs):>15.3f}")



def find_best_threshold(y_true, y_proba, metric=f1_score, thresholds=np.linspace(0, 1, 101)):
    """
    Finds the best threshold for converting probabilities into class predictions
    that maximizes a given metric.
    
    Parameters:
    -----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities.
    metric : function, optional
        A performance metric function that takes (y_true, y_pred) as input.
        Default is f1_score.
    thresholds : array-like, optional
        List of thresholds to evaluate. Default is 0 to 1 in 0.01 increments.
    
    Returns:
    --------
    best_threshold : float
        The threshold that maximizes the metric.
    best_metric : float
        The corresponding metric value.
    """
    best_metric = -np.inf
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        score = metric(y_true, y_pred)
        if score > best_metric:
            best_metric = score
            best_threshold = thresh
    return best_threshold, best_metric



def _plot_single_roc(ax, y_true, y_proba, title):
    """Draw one ROC curve on the supplied axis and return the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")          # random-guess line
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return auc


def create_and_log_roc_curves(
        train_y, train_proba, val_y, val_proba,
        artifact_path="plots/roc_curves.png", log_to_mlflow=True):
    """
    Build the two-panel ROC figure, log it directly (no saving to disk),
    and log AUC values to MLflow. Returns (train_auc, val_auc).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    train_auc = _plot_single_roc(ax1, train_y, train_proba, "Training")
    val_auc   = _plot_single_roc(ax2, val_y,  val_proba,   "Validation")

    fig.suptitle("ROC Curves", fontsize=14)
    plt.tight_layout()

    if log_to_mlflow:
        mlflow.log_figure(fig, artifact_path)
        mlflow.log_metric("roc_auc_train", train_auc)
        mlflow.log_metric("roc_auc_val",   val_auc)

    plt.close(fig)
    return train_auc, val_auc




# ---- internal helper -------------------------------------------------
def _plot_single_roc(ax, y_true, y_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlim=(0, 1), ylim=(0, 1.05),
           xlabel="False Positive Rate",
           ylabel="True Positive Rate",
           title=title)
    ax.legend(loc="lower right")
    return auc

