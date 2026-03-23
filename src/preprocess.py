"""
Preprocessing module for breast cancer dataset.

This module handles dataset splitting and feature scaling
to prepare data for machine learning models.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    Preprocessor class for handling data splitting and scaling.

    Methods
    -------
    split_and_scale(X, y)
        Splits dataset into train/test sets and applies standard scaling.
    """

    def split_and_scale(self, X, y):
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
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test
