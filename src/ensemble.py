"""
Ensemble module.

This module provides functionality to combine multiple
machine learning models into an ensemble model.
"""

from sklearn.ensemble import VotingClassifier


class EnsembleBuilder:
    """
    Builder class for creating ensemble models.

    Methods
    -------
    build(models)
        Combines models into a voting classifier.
    """

    def build(self, models):
        """
        Build ensemble model using voting classifier.

        Parameters
        ----------
        models : dict
            Dictionary of model names and instances.

        Returns
        -------
        VotingClassifier
            Combined ensemble model.
        """
        estimators = [(name, model) for name, model in models.items()]

        ensemble = VotingClassifier(
            estimators=estimators,
            voting="hard",
        )

        return ensemble
