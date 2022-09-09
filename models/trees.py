# > Imports

# > Numpy
import numpy as np

# > Local Imports
from losses.categorical import gini_impurity


class DecisionTree():
    """
    A simple implementation of a Decision Tree

    Attributes:
    -----------
    left : DecisionTree
        A possible left child.
    right : DecisionTree
        A possible right child.
    split_feature : int
        The feature to split on in the case that this is a non-terminal node.
    split_value : float
        The value to split on in the case that this is a non-terminal node.
    majority_class : int
        The predicted class in the case that this is a leaf (terminal node).
    """

    def __init__(self, depth=1):
        # A tree can have left and right children
        self.left = None
        self.right = None

        # We want to know the split feature
        self.split_feature = None

        # And also the split value
        self.split_value = None

        # The majority class (if this is a leaf node)
        self.majority_class = None

        # Store the depth of this node
        self.depth = depth

    def fit(self, X, y, min_split=2, max_depth=None):
        """
        Fit the decision tree on the data.

        Parameters:
        -----------
        X : numpy.ndarray
            The input data of shape (m, n).
        y : numpy.ndarray
            The labels of shape (m, ).
        min_split : int
            The minimum number of samples required to split an internal node.
        max_depth : int
            The maximum depth of the Decision Tree.

        Returns:
        --------
        self : object
            Fitted Decision Tree.
        """
        # Determine if this is a leaf node

        if (X.shape[0] < min_split or self.depth == max_depth
                or np.max(y) == np.min(y)):
            self.majority_class = np.argmax(np.bincount(y))
            return self

        # We need to determine the best feature and feature value combination
        # to split on. To do that, we need to calculate the impurity
        # improvement for all combinations.
        best_impurity = gini_impurity(y)

        for i in range(X.shape[1]):
            # Determine all possible splits and their corresponding split
            # values for this feature.
            feature_values_sorted = np.unique(X[:, i])
            feature_possible_splits = np.unique(
                (feature_values_sorted[:-1] + feature_values_sorted[1:]) / 2)

            for split in feature_possible_splits:
                # Determine the left and right indices
                # Left contains all values <= split, right the values > split
                left_split_ind = np.where(X[:, i] <= split)[0]
                right_split_ind = np.where(X[:, i] > split)[0]

                # Calculate the impurity for this split
                # The new impurity is calculated using:
                # \pi(l) * impurity(l) + \pi(r) * impurity(r)
                # Where \pi stands for "proportion"
                proportion_l = len(left_split_ind) / len(y)
                proportion_r = len(right_split_ind) / len(y)

                new_impurity = proportion_l * gini_impurity(y[left_split_ind])\
                    + proportion_r * gini_impurity(y[right_split_ind])

                # If this beats our best impurity, save it
                if new_impurity < best_impurity:
                    best_impurity = new_impurity
                    self.split_feature = i
                    self.split_value = split

        # If we have not found any improvement, consider this a leaf
        if self.split_feature is None and self.split_value is None:
            self.majority_class = np.argmax(np.bincount(y))
            return self

        # Split our data based on the best split
        left_indices = np.where(X[:, self.split_feature] <= self.split_value)
        right_indices = np.where(X[:, self.split_feature] > self.split_value)

        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        # Create children
        self.left = DecisionTree(depth=self.depth+1)
        self.right = DecisionTree(depth=self.depth+1)

        # Grow the child nodes
        self.left.fit(left_X, left_y, min_split, max_depth)
        self.right.fit(right_X, right_y, min_split, max_depth)

        return self

    def predict(self, X, y):
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
        y_pred = []

        # Make a prediction for each of our datapoints
        for datapoint in X:
            current_node = self

            while True:
                # If the current node has a majority class (i.e. it is a leaf),
                # append it to the results.
                if current_node.majority_class is not None:
                    y_pred.append(current_node.majority_class)
                    break
                else:
                    # Otherwise keep going
                    if (datapoint[current_node.split_feature] <=
                            current_node.split_value):
                        current_node = current_node.left
                    else:
                        current_node = current_node.right

        return np.array(y_pred)
