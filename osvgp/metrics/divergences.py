#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Divergence-based performance metrics.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np

from osvgp.divergences import gaussian_kl
from osvgp.models import GPModel, GPR


# =============================================================================
#  FUNCTIONS
# =============================================================================

def compute_kl_from_exact_posterior(approx_model: GPModel, 
                                    num_points: int = 1000) -> float:
    """ Compute the KL divergence from an exact to approximate GP posterior.

    Given an approximate GP regression model (e.g. SGPR, OSGPR), an exact GP 
    regression model is created with the same training data and kernel 
    hyperparameters (so that the KL divergence is well-defined). Next, the 
    exact and approximate posteriors are computed at an evenly spaced set of 
    input values that span the range of the training data, then the KL 
    divergence KL[approx||exact] is computed.

    Note: this function only works with small (<= 5000 datapoints) 1D datasets! 
    The function could be amended to work with multidimensional datasets, but 
    the exact solution needs to be computationally tractable in order for it to 
    be useful, so I haave restricted it to 1D datasets for the time being as 
    this is will be my primary use case in my thesis.

    Args:
        approx_model (GPModel) : the approximate model to analyse
        num_points (int) : the number of posterior values to compute

    Returns:
        (float) : KL[approx||exact] -- KL divergence from the exact to 
                    approximate posterior
    """
    # Approx model training data and hyperparameters
    data = approx_model.data
    kernel = approx_model.kernel
    noise_variance = approx_model.likelihood.variance

    # Exact model with same training data and hyperparameters
    exact_model = GPR(data, kernel, noise_variance)

    # Input range for computing posteriors
    X, _ = approx_model.data
    Xmin, Xmax = float(min(X)), float(max(X))
    margin = 0.1 * (Xmax - Xmin)
    Xnew = np.linspace(Xmin - margin, Xmax + margin, num_points).reshape(-1, 1)

    # Approximate and exact posteriors
    mu_approx, sigma_approx = approx_model.predict_f(Xnew, full_cov=True)
    mu_exact, sigma_exact = exact_model.predict_f(Xnew, full_cov=True)

    # KL[approx||exact]
    return gaussian_kl(mu_approx, sigma_approx, mu_exact, sigma_exact)
