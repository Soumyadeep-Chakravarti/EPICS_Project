"""
Main pipeline module.

This module integrates all components and executes
the full machine learning workflow, including
data loading, preprocessing, model training,
evaluation, visualization, and result display.
"""

import os

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

    def __init__(self):
        """
        Initialize pipeline components.
        """
        self.loader = DataLoader()
        self.preprocessor = Preprocessor()
        self.model_factory = ModelFactory()
        self.ensemble_builder = EnsembleBuilder()
        self.evaluator = Evaluator()
        self.visualizer = Visualizer()

        os.makedirs("results", exist_ok=True)

    def display_results(self, metrics):
        """
        Display evaluation results in a formatted manner.

        Parameters
        ----------
        metrics : dict
            Dictionary containing model evaluation metrics.
        """
        print("\n=== MODEL PERFORMANCE SUMMARY ===\n")

        best_model = None
        best_accuracy = 0.0

        for model_name, values in metrics.items():
            print(f"{model_name}")
            print(f"  Accuracy : {values['accuracy']:.4f}")
            print(f"  Precision: {values['precision']:.4f}")
            print(f"  Recall   : {values['recall']:.4f}")
            print(f"  F1 Score : {values['f1_score']:.4f}\n")

            if values["accuracy"] > best_accuracy:
                best_accuracy = values["accuracy"]
                best_model = model_name

        print(
            f"Best Model: {best_model} "
            f"(Accuracy: {best_accuracy:.4f})\n"
        )

    def run(self):
        """
        Execute the complete machine learning pipeline.
        """
        # Load data
        X, y = self.loader.load_data()

        # Preprocess data
        X_train, X_test, y_train, y_test = (
            self.preprocessor.split_and_scale(X, y)
        )

        # Initialize models
        models = self.model_factory.get_models()
        metrics = {}

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

        self.visualizer.save_dataset_distribution(
            y,
            "results/dataset_distribution.png",
        )

        # Display results in terminal
        self.display_results(metrics)

        print("Pipeline complete. Results saved in /results")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
