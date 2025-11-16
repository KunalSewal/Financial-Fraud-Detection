"""
Evaluation metrics and visualization utilities.
"""

import os
from typing import Dict, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    
    # Add probabilistic metrics if probabilities provided
    if y_prob is not None:
        if y_prob.ndim > 1:
            # Convert logits to probabilities
            y_prob_pos = np.exp(y_prob[:, 1]) / np.exp(y_prob).sum(axis=1)
        else:
            y_prob_pos = y_prob
        
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob_pos)
            metrics["avg_precision"] = average_precision_score(y_true, y_prob_pos)
        except ValueError:
            # Handle case where only one class present
            metrics["roc_auc"] = 0.0
            metrics["avg_precision"] = 0.0
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    class_names: List[str] = ["Normal", "Fraud"]
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure
        class_names: Names of classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True
    )
    plt.title("Confusion Matrix", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved confusion matrix to {save_path}")
    
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    # Convert to probability of positive class
    if y_prob.ndim > 1:
        y_prob_pos = np.exp(y_prob[:, 1]) / np.exp(y_prob).sum(axis=1)
    else:
        y_prob_pos = y_prob
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob_pos)
    roc_auc = roc_auc_score(y_true, y_prob_pos)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved ROC curve to {save_path}")
    
    plt.show()


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
):
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    """
    # Convert to probability of positive class
    if y_prob.ndim > 1:
        y_prob_pos = np.exp(y_prob[:, 1]) / np.exp(y_prob).sum(axis=1)
    else:
        y_prob_pos = y_prob
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob_pos)
    avg_precision = average_precision_score(y_true, y_prob_pos)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="darkorange", lw=2, 
             label=f"PR curve (AP = {avg_precision:.3f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved PR curve to {save_path}")
    
    plt.show()


def plot_training_curves(
    history: Dict,
    save_path: Optional[str] = None
):
    """
    Plot training and validation loss/metrics curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Loss curves
    axes[0, 0].plot(epochs, history["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0, 0].set_xlabel("Epoch", fontsize=11)
    axes[0, 0].set_ylabel("Loss", fontsize=11)
    axes[0, 0].set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # F1 score curves
    train_f1 = [m["f1"] for m in history["train_metrics"]]
    val_f1 = [m["f1"] for m in history["val_metrics"]]
    axes[0, 1].plot(epochs, train_f1, "b-", label="Train F1", linewidth=2)
    axes[0, 1].plot(epochs, val_f1, "r-", label="Val F1", linewidth=2)
    axes[0, 1].set_xlabel("Epoch", fontsize=11)
    axes[0, 1].set_ylabel("F1 Score", fontsize=11)
    axes[0, 1].set_title("F1 Score", fontsize=12, fontweight="bold")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # Precision and Recall
    train_precision = [m["precision"] for m in history["train_metrics"]]
    val_precision = [m["precision"] for m in history["val_metrics"]]
    train_recall = [m["recall"] for m in history["train_metrics"]]
    val_recall = [m["recall"] for m in history["val_metrics"]]
    
    axes[1, 0].plot(epochs, train_precision, "b-", label="Train Precision", linewidth=2)
    axes[1, 0].plot(epochs, val_precision, "r-", label="Val Precision", linewidth=2)
    axes[1, 0].set_xlabel("Epoch", fontsize=11)
    axes[1, 0].set_ylabel("Precision", fontsize=11)
    axes[1, 0].set_title("Precision", fontsize=12, fontweight="bold")
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].plot(epochs, train_recall, "b-", label="Train Recall", linewidth=2)
    axes[1, 1].plot(epochs, val_recall, "r-", label="Val Recall", linewidth=2)
    axes[1, 1].set_xlabel("Epoch", fontsize=11)
    axes[1, 1].set_ylabel("Recall", fontsize=11)
    axes[1, 1].set_title("Recall", fontsize=12, fontweight="bold")
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved training curves to {save_path}")
    
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ["Normal", "Fraud"]
):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
    """
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)
    print(classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    ))
    print("=" * 60 + "\n")


def compare_models(
    results: Dict[str, Dict],
    save_path: Optional[str] = None
):
    """
    Compare multiple models side by side.
    
    Args:
        results: Dictionary mapping model names to their metrics
        save_path: Path to save comparison figure
    """
    models = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in models]
        
        bars = axes[i].bar(models, values, color="skyblue", edgecolor="navy", linewidth=1.5)
        axes[i].set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        axes[i].set_title(metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
        axes[i].set_ylim([0, 1.0])
        axes[i].grid(axis="y", alpha=0.3)
        axes[i].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[i].text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved model comparison to {save_path}")
    
    plt.show()


def save_results_to_markdown(
    results: Dict,
    save_path: str,
    model_name: str,
    config: Dict
):
    """
    Save results to markdown file.
    
    Args:
        results: Results dictionary
        save_path: Path to save markdown file
        model_name: Name of the model
        config: Model configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, "w") as f:
        f.write(f"# {model_name} Results\n\n")
        
        f.write("## Configuration\n\n")
        for key, value in config.items():
            f.write(f"- **{key}**: {value}\n")
        
        f.write("\n## Test Set Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        
        test_metrics = results.get("test_metrics", {})
        for metric, value in test_metrics.items():
            f.write(f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n")
        
        f.write(f"\n**Best Epoch**: {results.get('best_epoch', 'N/A')}\n")
        f.write(f"**Training Time**: {sum(results.get('epoch_times', [])):.2f}s\n")
    
    print(f"✓ Saved results to {save_path}")

