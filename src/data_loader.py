"""
Data loading module for breast cancer dataset.

This module provides functionality to load and structure
the dataset into feature and target components.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer


class DataLoader:
    """
    DataLoader class for loading the breast cancer dataset.

    Methods
    -------
    load_data()
        Loads dataset and returns features and labels.
    """

    def load_data(self):
        """
        Load the breast cancer dataset.

        Returns
        -------
        tuple
            Feature matrix (X) and target labels (y).
        """
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)

        return X, y
