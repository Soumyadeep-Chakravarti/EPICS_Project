"""
Evaluation module.

This module provides methods to evaluate machine learning
models using standard performance metrics including
cross-validation and ROC-AUC analysis.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score

from src.config import config, logger


class Evaluator:
    """
    Evaluator class for model performance assessment.

    Provides comprehensive evaluation including basic metrics,
    cross-validation, ROC-AUC, and detailed classification reports.

    Methods
    -------
    evaluate(model, X_test, y_test)
        Computes evaluation metrics.
    cross_validate(model, X, y)
        Performs k-fold cross-validation.
    get_classification_report(model, X_test, y_test)
        Generates detailed classification report.
    """

    def __init__(self) -> None:
        """Initialize Evaluator with config."""
        self.config = config.model
        logger.info("Evaluator initialized")

    def evaluate(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: Any,
    ) -> Dict[str, float]:
        """
        Evaluate model performance.

        Parameters
        ----------
        model : estimator
            Trained machine learning model.
        X_test : array-like
            Test feature data.
        y_test : array-like
            True labels.

        Returns
        -------
        dict
            Dictionary containing evaluation metrics including ROC-AUC.
        """
        predictions = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }

        # Add ROC-AUC if model supports probability prediction
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        logger.info(f"Evaluation complete: accuracy={metrics['accuracy']:.4f}")
        return metrics

    def cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        scoring: str = "accuracy",
    ) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.

        Parameters
        ----------
        model : estimator
            Machine learning model to evaluate.
        X : array-like
            Feature matrix.
        y : array-like
            Target labels.
        scoring : str
            Scoring metric (default: accuracy).

        Returns
        -------
        dict
            Dictionary with mean, std, and individual fold scores.
        """
        logger.info(f"Running {self.config.cv_folds}-fold cross-validation")

        scores = cross_val_score(
            model,
            X,
            y,
            cv=self.config.cv_folds,
            scoring=scoring,
        )

        result = {
            "mean": scores.mean(),
            "std": scores.std(),
            "scores": scores.tolist(),
        }

        logger.info(f"CV {scoring}: {result['mean']:.4f} (+/- {result['std']:.4f})")
        return result

    def get_classification_report(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: Any,
        target_names: Optional[List[str]] = None,
    ) -> str:
        """
        Generate detailed classification report.

        Parameters
        ----------
        model : estimator
            Trained model.
        X_test : array-like
            Test features.
        y_test : array-like
            True labels.
        target_names : list of str, optional
            Names for target classes.

        Returns
        -------
        str
            Formatted classification report.
        """
        if target_names is None:
            target_names = ["Benign", "Malignant"]

        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            target_names=target_names,
        )

        logger.info("Classification report generated")
        return report

    def predict_with_confidence(
        self,
        model: BaseEstimator,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict with confidence scores.

        Parameters
        ----------
        model : estimator
            Trained model.
        X : array-like
            Input data.

        Returns
        -------
        tuple
            Predictions and confidence scores.
        """
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            confidence = probs.max(axis=1)
            preds = probs.argmax(axis=1)
            return preds, confidence

        preds = model.predict(X)
        return preds, None

    def get_confusion_matrix(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: Any,
    ) -> np.ndarray:
        """
        Compute confusion matrix.

        Parameters
        ----------
        model : estimator
            Trained model.
        X_test : array-like
            Test features.
        y_test : array-like
            True labels.

        Returns
        -------
        np.ndarray
            Confusion matrix.
        """
        predictions = model.predict(X_test)
        return confusion_matrix(y_test, predictions)

    def get_roc_data(
        self,
        model: BaseEstimator,
        X_test: np.ndarray,
        y_test: Any,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Get ROC curve data for plotting.

        Parameters
        ----------
        model : estimator
            Trained model with predict_proba method.
        X_test : array-like
            Test features.
        y_test : array-like
            True labels.

        Returns
        -------
        tuple or None
            (fpr, tpr, auc_score) or None if not supported.
        """
        if not hasattr(model, "predict_proba"):
            return None

        from sklearn.metrics import roc_curve, auc

        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)

        return fpr, tpr, auc_score
