"""
Main pipeline module.

This module integrates all components and executes
the full machine learning workflow, including
data loading, preprocessing, model training,
evaluation, visualization, and result display.
"""

import os
from typing import Dict

import pandas as pd

from src.data_loader import DataLoader
from src.ensemble import EnsembleBuilder
from src.evaluate import Evaluator
from src.models import ModelFactory
from src.preprocess import Preprocessor
from src.visualize import Visualizer


class Pipeline:
    """
    Pipeline class for executing the ML workflow.

    Methods
    -------
    run()
        Executes the full pipeline.
    display_results(metrics)
        Displays evaluation results in terminal.
    """

    def __init__(self) -> None:
        """Initialize pipeline components."""
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.model_factory = ModelFactory()
        self.ensemble_builder = EnsembleBuilder()
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()

        os.makedirs("results", exist_ok=True)

    def display_results(
        self,
        metrics: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Display evaluation results in a tabular format.

        Parameters
        ----------
        metrics : dict
            Dictionary containing model evaluation metrics.
        """
        print("\n=== MODEL PERFORMANCE SUMMARY ===\n")

        df = pd.DataFrame(metrics).T
        df = df.sort_values(by="accuracy", ascending=False)

        best_model: str = df.index[0]

        header = (
            f"{'Model':<20}"
            f"{'Accuracy':<12}"
            f"{'Precision':<12}"
            f"{'Recall':<12}"
            f"{'F1 Score':<12}"
        )
        print(header)
        print("-" * len(header))

        for model, row in df.iterrows():
            marker = " <-- BEST" if model == best_model else ""

            print(
                f"{model:<20}"
                f"{row['accuracy']:<12.4f}"
                f"{row['precision']:<12.4f}"
                f"{row['recall']:<12.4f}"
                f"{row['f1_score']:<12.4f}"
                f"{marker}"
            )
            print()

    def display_confusion_matrix(self, cm):
        """
        Display confusion matrix in ASCII format.

        Parameters
        ----------
        cm : np.ndarray
          Confusion matrix.
        """
        print("\n=== CONFUSION MATRIX (Ensemble Model) ===\n")
        print("Labels: 0 = Benign, 1 = Malignant\n")

        print("+----------------------+------------+------------+")
        print("|                      |  Pred 0    |  Pred 1    |")
        print("+----------------------+------------+------------+")

        print(
            f"| Actual 0 (Benign)    |"
            f" {cm[0][0]:>8}   |"
            f" {cm[0][1]:>8}   |"
        )

        print(
            f"| Actual 1 (Malignant) |"
            f" {cm[1][0]:>8}   |"
            f" {cm[1][1]:>8}   |"
        )

        print("+----------------------+------------+------------+")
        print()

    def run(self) -> None:
        """Execute the complete machine learning pipeline."""
        # Load data
        X, y = self.loader.load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test = (
            self.preprocessor.split_and_scale(X, y)
        )

        # Initialize models
        models = self.model_factory.get_models()
        metrics: Dict[str, Dict[str, float]] = {}

        # Train and evaluate individual models
        for name, model in models.items():
            model.fit(X_train, y_train)
            metrics[name] = self.evaluator.evaluate(
                model,
                X_test,
                y_test,
            )

        # Build and evaluate ensemble
        ensemble = self.ensemble_builder.build(models)
        ensemble.fit(X_train, y_train)

        metrics["Ensemble"] = self.evaluator.evaluate(
            ensemble,
            X_test,
            y_test,
        )

        # Sample prediction with confidence
        sample_pred, sample_conf = (
            self.evaluator.predict_with_confidence(
                ensemble,
                X_test[:1],
            )
        )

        risk_level: str = "Low"
        if sample_conf is not None:
            if sample_conf[0] > 0.7:
                risk_level = "High"
            elif sample_conf[0] > 0.4:
                risk_level = "Medium"

        print("=== SAMPLE PREDICTION ===")

        label: str = (
            "Malignant" if sample_pred[0] == 1 else "Benign"
        )
        print(f"Prediction : {label}")

        if sample_conf is not None:
            print(f"Confidence : {sample_conf[0]:.4f}")
            print(f"Risk Level : {risk_level}")
        print()

        # Save metrics to CSV
        df = pd.DataFrame(metrics).T
        df.to_csv("results/metrics.csv")

        # Save human-readable metrics
        with open("results/metrics_pretty.txt", "w") as file:
            for model, vals in metrics.items():
                file.write(f"{model}\n")
                file.write(f"Accuracy: {vals['accuracy']:.4f}\n")
                file.write(f"Precision: {vals['precision']:.4f}\n")
                file.write(f"Recall: {vals['recall']:.4f}\n")
                file.write(f"F1 Score: {vals['f1_score']:.4f}\n\n")

        cm = self.evaluator.get_confusion_matrix(
            ensemble,
            X_test,
            y_test,
        )
        self.display_confusion_matrix(cm)

        # Generate visualizations
        self.visualizer.save_confusion_matrix(
            ensemble,
            X_test,
            y_test,
            "results/confusion_matrix.png",
        )

        self.visualizer.save_model_comparison(
            metrics,
            "results/model_comparison.png",
        )

        self.visualizer.save_feature_importance(
            models["Random Forest"],
            X.columns,
            "results/feature_importance.png",
        )

        top_features = self.visualizer.get_top_features(
            models["Random Forest"],
            X.columns,
        )

        print("Top Influential Features:")
        for feat in top_features:
            clean_feat = feat.replace("_", " ").title()
            print(f" - {clean_feat}")
        print()

        self.visualizer.save_dataset_distribution(
            y,
            "results/dataset_distribution.png",
        )

        # Display results in terminal
        self.display_results(metrics)

        best_model = max(metrics, key=lambda m: metrics[m]["accuracy"])
        print(f"\nFinal Selected Model: {best_model}")

        print("\nPipeline complete. Results saved in /results")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
