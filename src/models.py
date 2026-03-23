"""
Model factory module.

This module defines and provides machine learning models
used for classification tasks.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


class ModelFactory:
    """
    Factory class for creating machine learning models.

    Methods
    -------
    get_models()
        Returns dictionary of initialized models.
    """

    def get_models(self):
        """
        Get all machine learning models.

        Returns
        -------
        dict
            Dictionary mapping model names to model instances.
        """
        return {
            "Logistic Regression": LogisticRegression(max_iter=5000),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC(probability=True),
            "Random Forest": RandomForestClassifier(),
        }
