#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple helper functions (from GPflow + some custom ones.)
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Union

import numpy as np
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper

from osvgp.base import InputData


# =============================================================================
#  FUNCTIONS
# =============================================================================

def diagonal(x: InputData) -> np.ndarray: 
    """ Get the diagonal of an array-type.
    
     Args:
        x (InputData) : input array

    Raises:
        ValueError : if x is not square

    Returns:
        (np.ndarray) : diagonal of the input array
    
    """
    if is_square(x):
        return np.diag(np.array(x)).reshape(-1, 1)
    else:
        raise ValueError("Input array must be square!")


def flatten(x: InputData) -> Union[np.ndarray, None]:
    """ Cast an array-type as a flat numpy array.
    
    Args:
        x (InputData) : input array
    
    Returns:
        (np.ndarray or None) : x cast as a numpy array and flattened
    """
    if x is None:
        return None
    else:
        return np.array(x).ravel()


def is_square(x: InputData) -> bool:
    """ Check if an array-type is square. 
    
    Args:
        x (InputData) : input array

    Raises:
        ValueError : if x is not 2-dimensional

    Returns:
        (bool) : whether the input array is square
    """
    if len(x.shape) == 2:
        return x.shape[0] == x.shape[1]
    else:
        raise ValueError("Input array must be 2-dimensional!")
