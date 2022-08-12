## > Imports

# > Local Imports
from losses import r_squared

# > 3rd Party Imports
import numpy as np

class LinearRegression:
    """
    Linear Regression using Least Squares

    Attributes:
    -----------
    weights : numpy.ndarray of shape (n_features,)
        The weights of the model.
    bias : float
        The bias of the model.  
    """

    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X , y):
        """
        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        y : numpy.ndarray
            The labels of shape (m,).

        Returns:
        --------
        self : object
            Fitted estimator.
        """

        # First, add a bias term
        X = np.c_[X, np.ones(X.shape[0])]

        # Calculate the weights and bias using Least Squares
        # \beta = (X^T X)^{-1} X^T y
        beta = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)

        # Store the weights and bias
        self.weights = beta[:-1]
        self.bias = beta[-1]

        return self

    def predict(self, X):
        """
        Function to predict the output of the model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).

        Returns:
        --------
        y_pred : numpy.ndarray
            The predicted labels of shape (m,).
        """

        return X.dot(self.weights) + self.bias

    def score(self, X, y, loss=r_squared):
        """
        Function to calculate the score of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        y : numpy.ndarray
            The labels of shape (m,).
        loss : function
            The loss function to use.

        Returns:
        --------
        score : float
            The score of the model.
        """

        return loss(y, self.predict(X))
