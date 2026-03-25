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

from src.config import config, logger
from src.data_loader import DataLoader
from src.ensemble import EnsembleBuilder
from src.evaluate import Evaluator
from src.models import ModelFactory
from src.preprocess import Preprocessor
from src.visualize import Visualizer


class Pipeline:
    """
    Pipeline class for executing the ML workflow.

    This pipeline performs:
    - Data loading and preprocessing
    - Model training with cross-validation
    - Ensemble model creation
    - Comprehensive evaluation (metrics, ROC-AUC)
    - Visualization generation
    - Model persistence

    Methods
    -------
    run()
        Executes the full pipeline.
    display_results(metrics, cv_results)
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

        # Create directories from config
        os.makedirs(config.paths.results_dir, exist_ok=True)
        os.makedirs(config.paths.models_dir, exist_ok=True)

        logger.info("Pipeline initialized")

    def display_results(
        self,
        metrics: Dict[str, Dict[str, float]],
        cv_results: Dict[str, Dict[str, float]] = None,
    ) -> None:
        """
        Display evaluation results in a tabular format.

        Parameters
        ----------
        metrics : dict
            Dictionary containing model evaluation metrics.
        cv_results : dict, optional
            Cross-validation results for each model.
        """
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE SUMMARY")
        print("=" * 70 + "\n")

        df = pd.DataFrame(metrics).T
        df = df.sort_values(by="accuracy", ascending=False)

        best_model: str = df.index[0]

        # Header
        header = (
            f"{'Model':<20}"
            f"{'Accuracy':<10}"
            f"{'Precision':<10}"
            f"{'Recall':<10}"
            f"{'F1':<10}"
            f"{'ROC-AUC':<10}"
        )
        print(header)
        print("-" * 70)

        for model, row in df.iterrows():
            marker = " *BEST*" if model == best_model else ""

            roc_auc = row.get("roc_auc")
            roc_str = f"{roc_auc:.4f}" if roc_auc else "N/A"

            print(
                f"{model:<20}"
                f"{row['accuracy']:<10.4f}"
                f"{row['precision']:<10.4f}"
                f"{row['recall']:<10.4f}"
                f"{row['f1_score']:<10.4f}"
                f"{roc_str:<10}"
                f"{marker}"
            )

        # Display CV results if available
        if cv_results:
            print("\n" + "-" * 70)
            print("CROSS-VALIDATION RESULTS (5-Fold)")
            print("-" * 70)

            print(f"{'Model':<20}{'Mean Accuracy':<15}{'Std Dev':<10}")
            print("-" * 45)

            for model, cv in cv_results.items():
                print(f"{model:<20}{cv['mean']:<15.4f}{cv['std']:<10.4f}")

        print()

    def display_confusion_matrix(self, cm) -> None:
        """
        Display confusion matrix in ASCII format.

        Parameters
        ----------
        cm : np.ndarray
            Confusion matrix.
        """
        print("\n" + "=" * 50)
        print("CONFUSION MATRIX (Ensemble Model)")
        print("=" * 50)
        print("\nLabels: 0 = Benign, 1 = Malignant\n")

        print("+----------------------+------------+------------+")
        print("|                      |  Pred 0    |  Pred 1    |")
        print("+----------------------+------------+------------+")

        print(f"| Actual 0 (Benign)    | {cm[0][0]:>8}   | {cm[0][1]:>8}   |")

        print(f"| Actual 1 (Malignant) | {cm[1][0]:>8}   | {cm[1][1]:>8}   |")

        print("+----------------------+------------+------------+")
        print()

    def run(self) -> None:
        """Execute the complete machine learning pipeline."""
        logger.info("Starting pipeline execution")

        # Load data
        print("\n[1/8] Loading data...")
        X, y = self.loader.load_data()
        logger.info(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")

        # Preprocess data
        print("[2/8] Preprocessing data...")
        X_train, X_test, y_train, y_test = self.preprocessor.split_and_scale(X, y)

        # Save scaler for future use
        self.preprocessor.save_scaler()

        # Initialize models
        print("[3/8] Initializing models...")
        models = self.model_factory.get_models()
        metrics: Dict[str, Dict[str, float]] = {}
        cv_results: Dict[str, Dict[str, float]] = {}
        roc_data: Dict[str, tuple] = {}

        # Train and evaluate individual models
        print("[4/8] Training and evaluating models...")
        for name, model in models.items():
            print(f"  - Training {name}...")

            # Train model
            model.fit(X_train, y_train)

            # Evaluate on test set
            metrics[name] = self.evaluator.evaluate(model, X_test, y_test)

            # Cross-validation
            cv_results[name] = self.evaluator.cross_validate(model, X_train, y_train)

            # Get ROC data for plotting
            roc = self.evaluator.get_roc_data(model, X_test, y_test)
            if roc:
                roc_data[name] = roc

        # Build and evaluate ensemble
        print("[5/8] Building ensemble model...")
        ensemble = self.ensemble_builder.build(models)
        ensemble.fit(X_train, y_train)

        metrics["Ensemble"] = self.evaluator.evaluate(ensemble, X_test, y_test)
        cv_results["Ensemble"] = self.evaluator.cross_validate(
            ensemble, X_train, y_train
        )

        roc = self.evaluator.get_roc_data(ensemble, X_test, y_test)
        if roc:
            roc_data["Ensemble"] = roc

        # Save ensemble model
        self.ensemble_builder.save(ensemble)

        # Sample prediction with confidence
        print("[6/8] Running sample prediction...")
        sample_pred, sample_conf = self.evaluator.predict_with_confidence(
            ensemble, X_test[:1]
        )

        risk_level: str = "Low"
        if sample_conf is not None:
            if sample_conf[0] > 0.7:
                risk_level = "High"
            elif sample_conf[0] > 0.4:
                risk_level = "Medium"

        print("\n" + "=" * 50)
        print("SAMPLE PREDICTION")
        print("=" * 50)

        label: str = "Malignant" if sample_pred[0] == 1 else "Benign"
        print(f"Prediction : {label}")

        if sample_conf is not None:
            print(f"Confidence : {sample_conf[0]:.4f}")
            print(f"Risk Level : {risk_level}")

        # Save metrics to CSV
        print("\n[7/8] Saving results...")
        df = pd.DataFrame(metrics).T
        df.to_csv(config.paths.metrics_csv)

        # Save human-readable metrics
        with open(config.paths.metrics_txt, "w") as file:
            file.write("=" * 50 + "\n")
            file.write("MODEL EVALUATION RESULTS\n")
            file.write("=" * 50 + "\n\n")

            for model, vals in metrics.items():
                file.write(f"{model}\n")
                file.write("-" * len(model) + "\n")
                file.write(f"  Accuracy:  {vals['accuracy']:.4f}\n")
                file.write(f"  Precision: {vals['precision']:.4f}\n")
                file.write(f"  Recall:    {vals['recall']:.4f}\n")
                file.write(f"  F1 Score:  {vals['f1_score']:.4f}\n")
                if vals.get("roc_auc"):
                    file.write(f"  ROC-AUC:   {vals['roc_auc']:.4f}\n")
                file.write("\n")

            file.write("=" * 50 + "\n")
            file.write("CROSS-VALIDATION RESULTS (5-Fold)\n")
            file.write("=" * 50 + "\n\n")

            for model, cv in cv_results.items():
                file.write(f"{model}: {cv['mean']:.4f} (+/- {cv['std']:.4f})\n")

        # Classification report for ensemble
        report = self.evaluator.get_classification_report(ensemble, X_test, y_test)
        with open("results/classification_report.txt", "w") as f:
            f.write("CLASSIFICATION REPORT (Ensemble Model)\n")
            f.write("=" * 50 + "\n\n")
            f.write(report)

        # Confusion matrix
        cm = self.evaluator.get_confusion_matrix(ensemble, X_test, y_test)
        self.display_confusion_matrix(cm)

        # Generate visualizations
        print("[8/8] Generating visualizations...")

        self.visualizer.save_confusion_matrix(ensemble, X_test, y_test)
        self.visualizer.save_model_comparison(metrics)
        self.visualizer.save_feature_importance(models["Random Forest"], X.columns)
        self.visualizer.save_dataset_distribution(y)
        self.visualizer.save_roc_curves(roc_data)
        self.visualizer.save_cv_scores(cv_results)

        # Top features
        top_features = self.visualizer.get_top_features(
            models["Random Forest"], X.columns
        )

        print("\n" + "=" * 50)
        print("TOP INFLUENTIAL FEATURES")
        print("=" * 50)
        for i, feat in enumerate(top_features, 1):
            clean_feat = feat.replace("_", " ").title()
            print(f"  {i}. {clean_feat}")

        # Display results in terminal
        self.display_results(metrics, cv_results)

        best_model = max(metrics, key=lambda m: metrics[m]["accuracy"])
        print(f"Best Model: {best_model}")
        print(f"Best Accuracy: {metrics[best_model]['accuracy']:.4f}")

        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
        print(f"\nOutputs saved to:")
        print(f"  - Metrics:        {config.paths.results_dir}/")
        print(f"  - Models:         {config.paths.models_dir}/")
        print(f"  - Visualizations: {config.paths.results_dir}/*.png")

        logger.info("Pipeline execution completed successfully")


if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()
