## > Imports

# > Local Imports
from losses import mean_squared_error

# > 3rd Party Imports
import numpy as np

class LinearRegression:
    """
    Linear Regression using Gradient Descent

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

    def fit(self, X , y, learning_rate=0.001, epochs=1000):
        """
        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        y : numpy.ndarray
            The labels of shape (m,).
        learning_rate : float
            The learning rate for gradient descent.
        epochs : int
            The number of epochs to train the model.

        Returns:
        --------
        self : object
            Fitted estimator.
        """
        
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0.0

        # Perform gradient descent
        for _ in range(epochs):
            y_pred = self.predict(X)

            partial_w = (1 / X.shape[0]) * (2 * np.dot(X.T, y_pred - y))
            partial_d = (1 / X.shape[0]) * (2 * np.sum(y_pred - y))

            self.weights -= learning_rate * partial_w
            self.bias -= learning_rate * partial_d
        
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

    def score(self, X, y, loss=mean_squared_error):
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
