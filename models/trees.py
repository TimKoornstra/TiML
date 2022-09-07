# > Imports

# > Numpy
import numpy as np

# > Local Imports
from losses.categorical import gini_impurity


class DecisionTree():
    def __init__(self):
        # A tree can have left and right children
        self.left = None
        self.right = None

        # We want to know the split feature
        self.split_feature = None

        # And also the split value
        self.split_value = None

        # The majority class (if this is a leaf node)
        self.majority_class = None

    def fit(self, X, y, min_split=0, min_leaf=0):

        # We need to determine the best feature and feature value combination to split on
        # To do that, we need to calculate the impurity improvement for all combinations

        if X.shape[0] < min_leaf or X.shape[0] < min_split or np.max(y) == np.min(y):
            self.majority_class = np.argmax(np.bincount(y))
            return self

        current_impurity = gini_impurity(y)
        lowest_impurity = float("-inf")

        for i in range(X.shape[1]):
            # Determine all possible splits and their corresponding split values for this feature
            feature_values_sorted = np.unique(X[:, i])
            feature_possible_splits = np.unique((feature_values_sorted[:-1] + feature_values_sorted[1:]) / 2)

            for split in feature_possible_splits:
                # Determine the left and right indices
                # Left contains all values <= split, right all the values > split
                left_split_ind = np.where(X[:, i] <= split)[0]
                right_split_ind = np.where(X[:, i] > split)[0]

                # Calculate the impurity for this split
                new_impurity = current_impurity - gini_impurity(y[left_split_ind]) + gini_impurity(y[right_split_ind])
                # If this beats our old impurity, save it
                if new_impurity > lowest_impurity:
                    current_impurity = new_impurity
                    self.split_feature = i
                    self.split_value = split

        # Split our data based on the best split
        left_indices = np.where(X[:, self.split_feature] <= self.split_value)
        right_indices = np.where(X[:, self.split_feature] > self.split_value)

        left_X, left_y = X[left_indices], y[left_indices]
        right_X, right_y = X[right_indices], y[right_indices]

        # Create children
        self.left = DecisionTree()
        self.right = DecisionTree()

        # Grow the child nodes
        self.left.fit(left_X, left_y)
        self.right.fit(right_X, right_y)

        return self

    def predict(self, X, y):
        y_pred = []

        # Make a prediction for each of our datapoints
        for datapoint in X:
            current_node = self

            while True:
                # If the current node has a majority class (i.e. it is a leaf), append it to the results
                if current_node.majority_class is not None:
                    y_pred.append(current_node.majority_class)
                    break
                else:
                    # Otherwise keep going
                    if datapoint[current_node.split_feature] <= current_node.split_value:
                        current_node = current_node.left
                    else:
                        current_node = current_node.right

        return np.array(y_pred)
