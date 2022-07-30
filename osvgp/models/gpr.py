#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exact Gaussian process regression model.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import tensorflow as tf

from osvgp.base import InputData, RegressionData, MeanAndVariance
from osvgp.config import DEFAULT_FLOAT
from osvgp.kernels import Kernel
from osvgp.likelihoods import Gaussian
from osvgp.models import GPModel, InternalDataTrainingLossMixin
from osvgp.util import data_input_to_tensor


# =============================================================================
#  CLASSES
# =============================================================================

class GPR(GPModel, InternalDataTrainingLossMixin):
    """ Exact Gaussian Process regression (Gaussian likelihood). """

    def __init__(self,
                 data: RegressionData,
                 kernel: Kernel,
                 noise_variance: float = 1.0) -> None:
        """ Constructor method for the exact GPR model class.

        Args:
            data (RegressionData) : training data (X, y values)
            kernel (Kernel) : kernel function for computing covariance
            noise_variance (float) : data noise variance
        """
        # Inherit GP model superclass
        likelihood = Gaussian(noise_variance)
        super().__init__(kernel, likelihood, None, 1)

        # Set data attributes
        X, y = data_input_to_tensor(data)
        self.data = (X, y)
        self.num_data = X.shape[0]

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """ Set maximum likelihood objective as log marginal likelihood. """
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        """ Compute the log marginal likelihood. """
        X, y = self.data
        kff = self.kernel(X)
        I = tf.eye(self.num_data, dtype=DEFAULT_FLOAT)
        L = tf.linalg.cholesky(kff + self.likelihood.variance * I)

        alpha = tf.linalg.triangular_solve(L, y, lower=True)
        num_dims = tf.cast(self.num_data, L.dtype)
        p = -0.5 * tf.reduce_sum(tf.square(alpha), 0)
        p -= 0.5 * num_dims * np.log(2 * np.pi)
        p -= tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))

        return tf.reduce_sum(p)

    def predict_f(self,
                  Xnew: InputData,
                  full_cov: bool = False,
                  full_output_cov: bool = False) -> MeanAndVariance:
        """ Predict the mean and variance of the latent function at some points. 
        
        Args:
            Xnew (InputData) : new points at which to compute predictions
            full_cov (bool) : whether to return covariance or variance
            full_output_cov (bool) : required argument for GPmodel superclass

        Returns:
            mean, var (MeanAndVariance) : Predictive mean and variance
        """
        X, y = self.data
        kff = self.kernel(X)
        kss = self.kernel(Xnew, full_cov=full_cov)
        kfs = self.kernel(X, Xnew)
        I = tf.eye(self.num_data, dtype=DEFAULT_FLOAT)
        L = tf.linalg.cholesky(kff + self.likelihood.variance * I)
        tmp1 = tf.linalg.triangular_solve(L, kfs, lower=True)
        tmp2 = tf.linalg.triangular_solve(L, y, lower=True)

        # Predictive mean
        mean = tf.linalg.matmul(tmp1, tmp2, transpose_a=True)

        # Predictive (co)variance
        if full_cov:
            var = kss - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        else:
            var = kss - tf.reduce_sum(tf.square(tmp1), -2)
            var = tf.expand_dims(var, -1)

        return (mean, var)
