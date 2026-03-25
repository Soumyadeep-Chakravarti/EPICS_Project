"""
Configuration module.

This module provides centralized configuration for the ML pipeline
using dataclasses for type safety and easy modification.
"""

import logging
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5

    # Logistic Regression
    lr_max_iter: int = 5000
    lr_C: float = 1.0

    # Decision Tree
    dt_max_depth: int | None = None
    dt_min_samples_split: int = 2

    # SVM
    svm_C: float = 1.0
    svm_kernel: str = "rbf"

    # Random Forest
    rf_n_estimators: int = 100
    rf_max_depth: int | None = None


@dataclass
class PathConfig:
    """Configuration for file paths."""

    results_dir: str = "results"
    models_dir: str = "models"
    metrics_csv: str = "results/metrics.csv"
    metrics_txt: str = "results/metrics_pretty.txt"
    confusion_matrix_img: str = "results/confusion_matrix.png"
    model_comparison_img: str = "results/model_comparison.png"
    feature_importance_img: str = "results/feature_importance.png"
    dataset_distribution_img: str = "results/dataset_distribution.png"
    roc_curve_img: str = "results/roc_curves.png"
    scaler_path: str = "models/scaler.joblib"
    ensemble_path: str = "models/ensemble.joblib"


@dataclass
class Config:
    """Main configuration class combining all configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    # Feature selection
    use_feature_selection: bool = False
    top_k_features: int = 15

    # Evaluation
    metrics: List[str] = field(
        default_factory=lambda: [
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "roc_auc",
        ]
    )


# Global config instance
config = Config()


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.

    Parameters
    ----------
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("breast_cancer_ml")


# Global logger instance
logger = setup_logging()
