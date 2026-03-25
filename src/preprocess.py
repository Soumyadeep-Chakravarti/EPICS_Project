"""
Preprocessing module for breast cancer dataset.

This module handles dataset splitting, feature scaling,
and provides persistence for the scaler object.
"""

import os
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import config, logger


class Preprocessor:
    """
    Preprocessor class for handling data splitting and scaling.

    Attributes
    ----------
    scaler : StandardScaler
        Fitted scaler object for feature normalization.

    Methods
    -------
    split_and_scale(X, y)
        Splits dataset into train/test sets and applies standard scaling.
    save_scaler(path)
        Saves the fitted scaler to disk.
    load_scaler(path)
        Loads a previously saved scaler from disk.
    transform(X)
        Transforms new data using the fitted scaler.
    """

    def __init__(self) -> None:
        """Initialize Preprocessor."""
        self.scaler: Optional[StandardScaler] = None
        self.config = config.model
        logger.info("Preprocessor initialized")

    def split_and_scale(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series]:
        """
        Split dataset and apply standard scaling.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Feature matrix.
        y : pandas.Series or array-like
            Target labels.

        Returns
        -------
        tuple
            Scaled training and testing datasets:
            (X_train, X_test, y_train, y_test)

        Raises
        ------
        ValueError
            If X and y have mismatched lengths.
        """
        # Input validation
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same length. Got X={len(X)}, y={len(y)}"
            )

        if len(X) == 0:
            raise ValueError("Dataset cannot be empty")

        logger.info(f"Splitting dataset: {len(X)} samples")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,  # Ensure balanced split
        )

        logger.info(f"Split complete: train={len(X_train)}, test={len(X_test)}")

        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        logger.info("Feature scaling applied")

        return X_train, X_test, y_train, y_test

    def save_scaler(self, path: Optional[str] = None) -> None:
        """
        Save the fitted scaler to disk.

        Parameters
        ----------
        path : str, optional
            File path to save scaler. Defaults to config path.

        Raises
        ------
        ValueError
            If scaler has not been fitted yet.
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call split_and_scale first.")

        save_path = path or config.paths.scaler_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        joblib.dump(self.scaler, save_path)
        logger.info(f"Scaler saved to {save_path}")

    def load_scaler(self, path: Optional[str] = None) -> None:
        """
        Load a previously saved scaler from disk.

        Parameters
        ----------
        path : str, optional
            File path to load scaler from. Defaults to config path.

        Raises
        ------
        FileNotFoundError
            If scaler file does not exist.
        """
        load_path = path or config.paths.scaler_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Scaler not found at {load_path}")

        self.scaler = joblib.load(load_path)
        logger.info(f"Scaler loaded from {load_path}")

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the fitted scaler.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Feature matrix to transform.

        Returns
        -------
        np.ndarray
            Scaled feature matrix.

        Raises
        ------
        ValueError
            If scaler has not been fitted or loaded.
        """
        if self.scaler is None:
            raise ValueError("Scaler not available. Fit or load scaler first.")

        return self.scaler.transform(X)
