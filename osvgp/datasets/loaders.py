#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset loading functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import csv
from importlib import resources

import numpy as np
from gpflow.base import RegressionData


# =============================================================================
#  GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

DATA_MODULE = "osvgp.datasets.data"


# =============================================================================
#  FUNCTIONS
# =============================================================================

def load_snelson() -> RegressionData:
    """ Load the Snelson dataset. 
    
    The Snelson dataset contains 200 1D (X, y) pairs.

    Returns:
        (np.ndarray) : X values
        (np.ndarray) : y values
    """
    # Initialise the X and y values
    X, y = [], []

    # Populate X and y values from the data file
    with resources.open_text(DATA_MODULE, "snelson.csv") as csv_file:
        data_file = csv.reader(csv_file)
        next(data_file, None)
        for line in data_file:
            X.append(line[0])
            y.append(line[1])

    # Cast X and y as numpy arrays
    X = np.array(X, dtype=np.float64).reshape(-1, 1)
    y = np.array(y, dtype=np.float64).reshape(-1, 1)

    return (X, y)
