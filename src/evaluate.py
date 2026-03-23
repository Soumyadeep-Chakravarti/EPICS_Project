"""
Evaluation module.

This module provides methods to evaluate machine learning
models using standard performance metrics.
"""

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

    def evaluate(self, model, X_test, y_test):
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
