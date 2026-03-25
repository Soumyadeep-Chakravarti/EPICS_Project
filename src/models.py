"""
Model factory module.

This module defines and provides machine learning models
used for classification tasks.
"""

from typing import Dict

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import config, logger


class ModelFactory:
    """
    Factory class for creating machine learning models.

    Uses configuration from config module for model parameters.

    Methods
    -------
    get_models()
        Returns dictionary of initialized models.
    """

    def __init__(self) -> None:
        """Initialize ModelFactory with config."""
        self.config = config.model
        logger.info("ModelFactory initialized with config")

    def get_models(self) -> Dict[str, BaseEstimator]:
        """
        Get all machine learning models.

        Returns
        -------
        dict
            Dictionary mapping model names to model instances.
        """
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=self.config.lr_max_iter,
                C=self.config.lr_C,
                random_state=self.config.random_state,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=self.config.dt_max_depth,
                min_samples_split=self.config.dt_min_samples_split,
                random_state=self.config.random_state,
            ),
            "SVM": SVC(
                C=self.config.svm_C,
                kernel=self.config.svm_kernel,
                probability=True,
                random_state=self.config.random_state,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=self.config.rf_n_estimators,
                max_depth=self.config.rf_max_depth,
                random_state=self.config.random_state,
            ),
        }

        logger.info(f"Created {len(models)} models: {list(models.keys())}")
        return models
