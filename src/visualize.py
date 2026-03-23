"""
Visualization module.

This module provides functions to generate and save
plots for model evaluation and dataset analysis.
"""

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.dpi"] = 300


class Visualizer:
    """
    Visualizer class for generating plots.
    """

    def save_confusion_matrix(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: pd.Series,
        path: str,
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
        path : str
            File path to save image.
        """
        predictions = model.predict(X_test)
        cm = confusion_matrix(y_test, predictions)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

        plt.title("Confusion Matrix (Ensemble Model)")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_model_comparison(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        path: str,
    ) -> None:
        """
        Save model accuracy comparison bar chart.

        Parameters
        ----------
        metrics_dict : dict
            Dictionary of model metrics.
        path : str
            File path to save image.
        """
        df = pd.DataFrame(metrics_dict).T

        plt.figure(figsize=(6, 4))
        df["accuracy"].plot(kind="bar")

        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=30)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def save_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: List[str],
        path: str,
    ) -> None:
        """
        Save feature importance plot.

        Parameters
        ----------
        model : BaseEstimator
            Model with feature_importances_ attribute.
        feature_names : list of str
            Names of features.
        path : str
            File path to save image.
        """
        if hasattr(model, "feature_importances_"):
            importance: np.ndarray = model.feature_importances_
            indices = importance.argsort()[-10:]

            plt.figure(figsize=(6, 4))
            plt.barh(range(10), importance[indices])
            plt.yticks(
                range(10),
                [feature_names[i] for i in indices],
            )

            plt.title("Top 10 Feature Importances")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

    def save_dataset_distribution(
        self,
        y: pd.Series,
        path: str,
    ) -> None:
        """
        Save dataset class distribution plot.

        Parameters
        ----------
        y : pandas.Series
            Target labels.
        path : str
            File path to save image.
        """
        counts = y.value_counts()

        plt.figure(figsize=(4, 4))
        counts.plot(kind="bar")

        plt.title("Dataset Class Distribution")
        plt.xticks([0, 1], ["Benign", "Malignant"], rotation=0)
        plt.ylabel("Count")

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

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
