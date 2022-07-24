## > Imports

# > 3rd Party Imports
import numpy as np

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