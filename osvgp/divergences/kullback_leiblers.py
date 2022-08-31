#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
KL divergence functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import tensorflow as tf

from osvgp.config import DEFAULT_FLOAT, DEFAULT_JITTER
from osvgp.util import is_square


# =============================================================================
#  FUNCTIONS
# =============================================================================

def gaussian_kl(q_mu: tf.Tensor,
                q_sigma: tf.Tensor,
                p_mu: tf.Tensor,
                p_sigma: tf.Tensor) -> float:
    """ Compute KL[q||p] where q and p are both multivariate Gaussians.

    Computes the KL divergence from p to q where
        
        - q = N(q_mu, q_sigma)
        - p = N(p_mu, p_sigma)

    Args:
        q_mu (tf.Tensor) : mean of q distribution
        q_sigma (tf.Tensor) : covariance of q distribution
        p_mu (tf.Tensor) : mean of p distribution
        p_sigma (tf.Tensor) : covariance of p distribution
    
    Raises:
        ValueError : if q_sigma or p_sigma are not full covariance matrices

    Returns:
        (float) : KL[q||p]
    """
    # Check dimensions are valid
    assert q_mu.shape == p_mu.shape
    assert q_sigma.shape == p_sigma.shape
    if not is_square(q_sigma) or not is_square(p_sigma):
        raise ValueError("Full covariance matrices required to compute KL!")

    m = p_mu - q_mu
    N = q_sigma.shape[0]
    IN = tf.eye(N, dtype=DEFAULT_FLOAT)
    Lq = tf.linalg.cholesky(q_sigma + DEFAULT_JITTER * IN)
    Lp = tf.linalg.cholesky(p_sigma + DEFAULT_JITTER * IN)

    # Trace term
    tmp1 = tf.linalg.triangular_solve(Lp, Lq, lower=True)
    tmp2 = tf.linalg.triangular_solve(tf.transpose(Lp), tmp1, lower=False)
    tmp3 = tf.linalg.matmul(tmp2, Lq, transpose_b=True)
    trace = tf.reduce_sum(tf.linalg.diag_part(tmp3))

    # Quadratic term
    tmp4 = tf.linalg.triangular_solve(Lp, m)
    quad = tf.linalg.matmul(tmp4, tmp4, transpose_a=True)

    # Log determinant term
    logdetq = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lq)))
    logdetp = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lp)))
    logdet = logdetp - logdetq

    # KL divergence
    KL = 0.5 * float(trace + quad - N + logdet)
    assert KL >= 0

    return KL
    