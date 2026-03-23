"""
Evaluation module.

This module provides methods to evaluate machine learning
models using standard performance metrics.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)


class Evaluator:
    """
    Evaluator class for model performance assessment.

    Methods
    -------
    evaluate(model, X_test, y_test)
        Computes evaluation metrics.
    """

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
            Dictionary containing evaluation metrics.
        """
        predictions = model.predict(X_test)

        return {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions),
            "recall": recall_score(y_test, predictions),
            "f1_score": f1_score(y_test, predictions),
        }

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
            model,
            X_test,
            y_test,
    ):
        """
        Compute confusion matrix.

        Returns
        -------
        np.ndarray
          Confusion matrix.
        """
        from sklearn.metrics import confusion_matrix

        predictions = model.predict(X_test)
        return confusion_matrix(y_test, predictions)
