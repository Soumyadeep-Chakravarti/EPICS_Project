"""
Ensemble module.

This module provides functionality to combine multiple
machine learning models into an ensemble model with
persistence capabilities.
"""

import os
from typing import Dict, Optional

import joblib
from sklearn.base import BaseEstimator
from sklearn.ensemble import VotingClassifier

from src.config import config, logger


class EnsembleBuilder:
    """
    Builder class for creating and managing ensemble models.

    Methods
    -------
    build(models)
        Combines models into a voting classifier.
    save(ensemble, path)
        Saves ensemble model to disk.
    load(path)
        Loads ensemble model from disk.
    """

    def __init__(self) -> None:
        """Initialize EnsembleBuilder."""
        logger.info("EnsembleBuilder initialized")

    def build(
        self,
        models: Dict[str, BaseEstimator],
        voting: str = "soft",
    ) -> VotingClassifier:
        """
        Build ensemble model using voting classifier.

        Parameters
        ----------
        models : dict
            Dictionary of model names and instances.
        voting : str
            Voting strategy: 'hard' or 'soft' (default: soft).

        Returns
        -------
        VotingClassifier
            Combined ensemble model.

        Raises
        ------
        ValueError
            If models dictionary is empty.
        """
        if not models:
            raise ValueError("Cannot build ensemble with empty models dict")

        estimators = [(name, model) for name, model in models.items()]

        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting,
        )

        logger.info(f"Built ensemble with {len(models)} models, voting={voting}")
        return ensemble

    def save(
        self,
        ensemble: VotingClassifier,
        path: Optional[str] = None,
    ) -> None:
        """
        Save ensemble model to disk.

        Parameters
        ----------
        ensemble : VotingClassifier
            Trained ensemble model.
        path : str, optional
            File path to save model. Defaults to config path.
        """
        save_path = path or config.paths.ensemble_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        joblib.dump(ensemble, save_path)
        logger.info(f"Ensemble saved to {save_path}")

    def load(self, path: Optional[str] = None) -> VotingClassifier:
        """
        Load ensemble model from disk.

        Parameters
        ----------
        path : str, optional
            File path to load model from. Defaults to config path.

        Returns
        -------
        VotingClassifier
            Loaded ensemble model.

        Raises
        ------
        FileNotFoundError
            If model file does not exist.
        """
        load_path = path or config.paths.ensemble_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model not found at {load_path}")

        ensemble = joblib.load(load_path)
        logger.info(f"Ensemble loaded from {load_path}")
        return ensemble
