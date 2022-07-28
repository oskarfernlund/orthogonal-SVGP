#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple helper functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

from osvgp.base import ArrayLike


# =============================================================================
#  FUNCTIONS
# =============================================================================

def diagonal(x: ArrayLike) -> np.ndarray: 
    """ Get the diagonal of an array-type.
    
     Args:
        x (ArrayLike) : input array

    Raises:
        ValueError : if x is not square

    Returns:
        (np.ndarray) : diagonal of the input array
    
    """
    if is_square(x):
        return np.diag(np.array(x)).reshape(-1, 1)
    else:
        raise ValueError("Input array must be square!")


def flatten(x: ArrayLike) -> np.ndarray:
    """ Cast an array-type as a flat numpy array.
    
    Args:
        x (ArrayLike) : input array
    
    Returns:
        (np.ndarray or None) : x cast as a numpy array and flattened
    """
    if x is None:
        return None
    else:
        return np.array(x).ravel()


def is_square(x: ArrayLike) -> bool:
    """ Check if an array-type is square. 
    
    Args:
        x (ArrayLike) : input array

    Raises:
        ValueError : if x is not 2-dimensional

    Returns:
        (bool) : whether the input array is square
    """
    if len(x.shape) == 2:
        return x.shape[0] == x.shape[1]
    else:
        raise ValueError("Input array must be 2-dimensional!")
