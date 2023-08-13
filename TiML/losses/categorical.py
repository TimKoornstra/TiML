# > Imports

# > Numpy
import numpy as np


def gini_impurity(y):
    probability_distribution = np.unique(y, return_counts=True)[1]/y.shape[0]
    return 1-np.sum(probability_distribution**2)


def entropy(y):
    probability_distribution = np.unique(y, return_counts=True)[1]/y.shape[0]
    return np.sum(-probability_distribution*np.log2(probability_distribution+1e-9))
