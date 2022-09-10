# > Imports

# > Local Imports
from losses.numerical import r_squared

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

    def fit(self, X, y):
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


class LogisticRegression:
    """
    Logistic Regression using gradient descent

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

    def fit(self, X, y, epochs=100, lr=0.001):
        """
        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        y : numpy.ndarray
            The labels of shape (m,).
        epochs : int
            The amount of epochs used to optimize the regressor.
        lr : numpy.float
            The learning rate used in the gradient descent.

        Returns:
        --------
        self : object
            Fitted estimator.
        """

        # Conversion to np.float128 to mitigate some exp OverflowErrors
        X = np.array(X, dtype=np.float128)

        # If the models has not been trained yet,
        # initialize the weights and bias.
        if not self.weights and not self.bias:
            self.weights = np.zeros(X.shape[1], dtype=np.float128)
            self.bias = np.float128(0.0)

        # Perform Gradient Descent by looping over epochs
        for _ in range(epochs):
            # Calculate the sigmoid activation
            sigmoid = 1 / (1 + np.exp(-(X.dot(self.weights) + self.bias)))

            # Calculate the difference w.r.t. the weights and bias vectors
            dW = np.dot(X.T, (sigmoid - y))
            db = 1/X.shape[0] * np.sum(sigmoid - y)

            # Update weights and bias using the learning rate
            self.weights -= lr * dW
            self.bias -= lr * db

        return self

    def predict(self, X, threshold=0.5):
        """
        Function to predict the output of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        threshold : numpy.float
            The cut-off to determine which values belong to which class.
            Anything smaller than the threshold is considered class 0, whereas
            anything greater than or equal to the threshold is class 1.

        Returns:
        --------
        y_pred : numpy.ndarray
            The predicted labels of shape (m,).
        """

        # Conversion to np.float128 to mitigate some exp OverflowErrors
        X = np.array(X, dtype=np.float128)

        # Calculate the sigmoid
        y_sigmoid = 1 / (1 + np.exp(-(X.dot(self.weights) + self.bias)))

        # Return 0 if value < threshold, else 1
        return np.where(y_sigmoid < threshold, 0, 1)
