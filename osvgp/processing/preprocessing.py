#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset preprocessing functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from copy import deepcopy

import numpy as np

from osvgp.base import RegressionData, SplitRegressionData


# =============================================================================
#  FUNCTIONS
# =============================================================================

def train_test_split(data: RegressionData, 
                     test_size: float = 0.25,
                     shuffle: bool = True,
                     seed: int = 69) -> SplitRegressionData:
    """ Split a regression dataset into a training and test set. 
    
    Args:
        data (RegressionData) : full dataset to split
        test_size (float) : size of the test set (0-1)
        shuffle (bool) : whether to shuffle the dataset before splitting
        seed (int) : seed for the random number generator
    
    Raises:
        ValueError : if test_size < 0 or > 1

    Returns:
        (SplitRegressionData) : training and test sets
    """
    # Check test_size is valid
    if test_size < 0.0 or test_size > 1.0:
        raise ValueError("test_size must be a float between 0.0 and 1.0!")
    
    # Unpack data and determine splitting index
    X, y = data
    split_idx = int(X.shape[0] * (1 - test_size))

    # Shuffle data
    if shuffle:
        rng = np.random.default_rng(seed)
        shuffler = rng.permutation(X.shape[0])
        X, y = X[shuffler], y[shuffler]

    # Split training and test set
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    return ((X_train, y_train), (X_test, y_test))
    