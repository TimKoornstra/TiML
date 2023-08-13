# > Imports

# > 3rd Party Imports
import numpy as np


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

    def fit(self, X, y, epochs=1000, lr=0.001):
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

        # If the models has not been trained yet,
        # initialize the weights and bias.
        if not self.weights and not self.bias:
            self.weights = np.zeros(X.shape[1])
            self.bias = 0.0

        # Perform Gradient Descent by looping over epochs
        for _ in range(epochs):
            # Calculate the sigmoid activation
            sigmoid = np.array([self._sigmoid(x) for x in
                                (np.matmul(self.weights, X.T) + self.bias)])

            # Calculate the difference between prediction and actual once
            difference = sigmoid - y

            # Calculate the derivatives w.r.t. the weights and bias vectors
            dW = 1/X.shape[0] * np.dot(X.T, difference)
            db = 1/X.shape[0] * np.sum(difference)

            # Update weights and bias using the learning rate
            self.weights -= lr * dW
            self.bias -= lr * db

        return self

    def _sigmoid(self, x):
        """
        A helper function to calculate the sigmoid of a value. This function
        is necessary to mitigate the overflow errors that occur when the
        sigmoid is negative or positive infinity.
        Credit for exp-normalize trick goes to
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/.

        Parameters:
        -----------
        x : Number
            The value for which we want to calculate the sigmoid.

        Returns:
        --------
        The sigmoid value for x.
        """

        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    def predict(self, X, threshold=0.5):
        """
        Function to predict the output of the model.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        threshold : float
            The cut-off to determine which values belong to which class.
            Anything smaller than the threshold is considered class 0, whereas
            anything greater than or equal to the threshold is class 1.

        Returns:
        --------
        y_pred : numpy.ndarray
            The predicted labels of shape (m,).
        """

        # Calculate the sigmoid activation
        y_sigmoid = np.array([self._sigmoid(x) for x in
                              (np.matmul(self.weights, X.T) + self.bias)])

        # Return 0 if value < threshold, else 1
        return np.where(y_sigmoid < threshold, 0, 1)
