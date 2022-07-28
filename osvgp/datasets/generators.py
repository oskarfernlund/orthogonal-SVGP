#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset generating functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
from gpflow.base import RegressionData


# =============================================================================
#  GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

rng = np.random.RandomState(69)


# =============================================================================
#  FUNCTIONS
# =============================================================================

def generate_multi_sine(num_data: int = 100,
                        noise_variance: float = 1.0) -> RegressionData:
    """ Generate an artificial dataset using multi-sine function.
    
    Args:
        num_data (int) : number of datapoints to generate
        noise_variance (float) : noise to add to the generator function
    """
    X = rng.rand(num_data, 1) * 2 - 1 
    y = multi_sine(X) + rng.randn(num_data, 1) * np.sqrt(noise_variance)

    return (X, y)


def multi_sine(x: np.ndarray) -> np.ndarray:
    """ Combination of multiple sinusoids. 
    
    Args:
        x (np.ndarray) : input values
    
    Returns:
        (np.ndarray) : corresponding output values
    """
    return np.sin(x*3* np.pi) + 0.3*np.cos(x*9*np.pi) + 0.5*np.sin(x*7*np.pi)