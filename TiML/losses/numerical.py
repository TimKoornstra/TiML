# > Imports

# > 3rd Party Imports
import numpy as np


def r_squared(y_true, y_pred):
    """
    Calculate the R squared score.

    Parameters:
    -----------
    y_true : numpy.ndarray
        The true labels.
    y_pred : numpy.ndarray
        The predicted labels.

    Returns:
    --------
    score : float
        The R squared score.
    """

    return 1 - np.sum(np.square(y_pred - y_true)) / np.sum(np.square(y_true - np.mean(y_true)))


def mean_squared_error(y_true, y_pred):
    """
    Calculate the mean squared error.

    Parameters:
    -----------
    y_true : numpy.ndarray
        The true labels.
    y_pred : numpy.ndarray
        The predicted labels.

    Returns:
    --------
    score : float
        The mean squared error.
    """

    return np.mean(np.square(y_pred - y_true))


def mean_absolute_error(y_true, y_pred):
    """
    Calculate the mean absolute error.

    Parameters:
    -----------
    y_true : numpy.ndarray
        The true labels.
    y_pred : numpy.ndarray
        The predicted labels.

    Returns:
    --------
    score : float
        The mean squared error.
    """

    return np.mean(np.abs(y_pred - y_true))
