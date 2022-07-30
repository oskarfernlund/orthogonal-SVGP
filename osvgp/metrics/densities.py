#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Probability density-based performance metrics.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import tensorflow as tf

from osvgp.base import RegressionData
from osvgp.models import GPModel


# =============================================================================
#  FUNCTIONS
# =============================================================================

def compute_lpd(model: GPModel, data: RegressionData) -> float:
    """ Compute log predictive density on some data.

    Args:
        model (GPModel) : GP model to evaluate
        data (RegressionData) : data on which to compute lpd -- ideally some 
            test data that is yet unseen by the model
    
    Returns:
        (float) : log predictive density on the provided data
    """ 
    return float(tf.reduce_sum(model.predict_log_density(data)))
