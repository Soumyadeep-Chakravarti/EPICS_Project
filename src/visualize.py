"""
Visualization module.

This module provides functions to generate and save
plots for model evaluation and dataset analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

plt.rcParams["figure.dpi"] = 300


class Visualizer:
    """
    Visualizer class for generating plots.

    Methods
    -------
    save_confusion_matrix(...)
    save_model_comparison(...)
    save_feature_importance(...)
    save_dataset_distribution(...)
    """

    def save_confusion_matrix(self, model, X_test, y_test, path):
        """
        Save confusion matrix plot.

        Parameters
        ----------
        model : estimator
            Trained model.
        X_test : array-like
            Test data.
        y_test : array-like
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

    def save_model_comparison(self, metrics_dict, path):
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

    def save_feature_importance(self, model, feature_names, path):
        """
        Save feature importance plot.

        Parameters
        ----------
        model : estimator
            Model with feature_importances_ attribute.
        feature_names : list
            Names of features.
        path : str
            File path to save image.
        """
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            indices = importance.argsort()[-10:]

            plt.figure(figsize=(6, 4))
            plt.barh(range(10), importance[indices])
            plt.yticks(range(10), [feature_names[i] for i in indices])

            plt.title("Top 10 Feature Importances")
            plt.tight_layout()
            plt.savefig(path)
            plt.close()

    def save_dataset_distribution(self, y, path):
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
