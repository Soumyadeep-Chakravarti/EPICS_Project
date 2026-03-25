"""
Visualization module.

This module provides functions to generate and save
plots for model evaluation and dataset analysis,
including ROC curves.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

from src.config import config, logger

plt.rcParams["figure.dpi"] = 300


class Visualizer:
    """
    Visualizer class for generating plots.

    Provides methods for creating confusion matrices,
    model comparisons, feature importance, ROC curves,
    and dataset distribution visualizations.
    """

    def __init__(self) -> None:
        """Initialize Visualizer."""
        logger.info("Visualizer initialized")

    def save_confusion_matrix(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: pd.Series,
        path: Optional[str] = None,
    ) -> None:
        """
        Save confusion matrix plot.

        Parameters
        ----------
        model : BaseEstimator
            Trained model.
        X_test : numpy.ndarray
            Test data.
        y_test : pandas.Series
            True labels.
        path : str, optional
            File path to save image. Defaults to config path.
        """
        save_path = path or config.paths.confusion_matrix_img

        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
        )

        plt.title("Confusion Matrix (Ensemble Model)")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Confusion matrix saved to {save_path}")

    def save_model_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        path: Optional[str] = None,
    ) -> None:
        """
        Save model comparison bar chart for multiple metrics.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary of model metrics.
        path : str, optional
            File path to save image. Defaults to config path.
        """
        save_path = path or config.paths.model_comparison_img

        df = pd.DataFrame(metrics_dict).T

        # Select metrics to plot (exclude None values)
        plot_metrics = ["accuracy", "precision", "recall", "f1_score"]
        if "roc_auc" in df.columns and df["roc_auc"].notna().all():
            plot_metrics.append("roc_auc")

        fig, axes = plt.subplots(1, len(plot_metrics), figsize=(15, 5))

        for idx, metric in enumerate(plot_metrics):
            ax = axes[idx] if len(plot_metrics) > 1 else axes
            df[metric].plot(kind="bar", ax=ax, color="steelblue")
            ax.set_title(metric.replace("_", " ").title())
            ax.set_ylabel("Score")
            ax.set_ylim(0, 1)
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle("Model Performance Comparison", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        logger.info(f"Model comparison saved to {save_path}")

    def save_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        path: Optional[str] = None,
        top_n: int = 10,
    ) -> None:
        """
        Save feature importance plot.

        Parameters
        ----------
        model : BaseEstimator
            Model with feature_importances_ attribute.
        feature_names : list of str
            Names of features.
        path : str, optional
            File path to save image. Defaults to config path.
        top_n : int
            Number of top features to display (default: 10).
        """
        save_path = path or config.paths.feature_importance_img

        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature_importances_ attribute")
            return

        importance: np.ndarray = model.feature_importances_
        indices = importance.argsort()[-top_n:]

        plt.figure(figsize=(8, 6))
        plt.barh(range(top_n), importance[indices], color="steelblue")
        plt.yticks(
            range(top_n),
            [feature_names[i] for i in indices],
        )

        plt.title(f"Top {top_n} Feature Importances")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Feature importance saved to {save_path}")

    def save_dataset_distribution(
        self,
        y: pd.Series,
        path: Optional[str] = None,
    ) -> None:
        """
        Save dataset class distribution plot.

        Parameters
        ----------
        y : pandas.Series
            Target labels.
        path : str, optional
            File path to save image. Defaults to config path.
        """
        save_path = path or config.paths.dataset_distribution_img

        counts = y.value_counts()

        plt.figure(figsize=(6, 5))
        colors = ["#66b3ff", "#ff9999"]
        counts.plot(kind="bar", color=colors)

        plt.title("Dataset Class Distribution")
        plt.xticks([0, 1], ["Benign (0)", "Malignant (1)"], rotation=0)
        plt.ylabel("Count")
        plt.xlabel("Class")

        # Add count labels on bars
        for i, count in enumerate(counts):
            plt.text(i, count + 5, str(count), ha="center", fontweight="bold")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"Dataset distribution saved to {save_path}")

    def save_roc_curves(
        self,
        roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
        path: Optional[str] = None,
    ) -> None:
        """
        Save ROC curves for multiple models.

        Parameters
        ----------
        roc_data : dict
            Dictionary mapping model names to (fpr, tpr, auc) tuples.
        path : str, optional
            File path to save image. Defaults to config path.
        """
        save_path = path or config.paths.roc_curve_img

        plt.figure(figsize=(8, 6))

        colors = plt.cm.Set1(np.linspace(0, 1, len(roc_data)))

        for (name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
            plt.plot(
                fpr,
                tpr,
                color=color,
                lw=2,
                label=f"{name} (AUC = {auc_score:.3f})",
            )

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves - Model Comparison")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        logger.info(f"ROC curves saved to {save_path}")

    def save_cv_scores(
        self,
        cv_results: Dict[str, Dict[str, float]],
        path: str = "results/cv_scores.png",
    ) -> None:
        """
        Save cross-validation scores comparison.

        Parameters
        ----------
        cv_results : dict
            Dictionary mapping model names to CV results.
        path : str
            File path to save image.
        """
        models = list(cv_results.keys())
        means = [cv_results[m]["mean"] for m in models]
        stds = [cv_results[m]["std"] for m in models]

        plt.figure(figsize=(8, 5))
        x = np.arange(len(models))

        plt.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
        plt.xticks(x, models, rotation=30, ha="right")
        plt.ylabel("Accuracy")
        plt.title("Cross-Validation Scores (5-Fold)")
        plt.ylim(0, 1)

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(
                i,
                mean + std + 0.02,
                f"{mean:.3f}",
                ha="center",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        logger.info(f"CV scores plot saved to {path}")

    def get_top_features(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        top_n: int = 5,
    ) -> List[str]:
        """
        Get top important features from model.

        Parameters
        ----------
        model : BaseEstimator
            Trained model with feature_importances_.
        feature_names : list of str
            Feature names.
        top_n : int
            Number of top features to return.

        Returns
        -------
        list of str
            Top feature names.
        """
        if hasattr(model, "feature_importances_"):
            importance: np.ndarray = model.feature_importances_
            indices = importance.argsort()[-top_n:][::-1]

            return [feature_names[i] for i in indices]

        return []
