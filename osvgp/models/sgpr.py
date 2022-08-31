#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sparse Gaussian process regression model.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import tensorflow as tf

from osvgp.base import InputData, RegressionData, MeanAndVariance
from osvgp.base import InducingPointsLike
from osvgp.config import DEFAULT_FLOAT, DEFAULT_JITTER
from osvgp.kernels import Kernel
from osvgp.likelihoods import Gaussian
from osvgp.models import GPModel, InternalDataTrainingLossMixin
from osvgp.util import data_input_to_tensor, inducingpoint_wrapper


# =============================================================================
#  CLASSES
# =============================================================================

class SGPR(GPModel, InternalDataTrainingLossMixin):
    """ Sparse Gaussian Process regression model. 
    
    Sparse variational Gaussian Process with a collapsed bound for regression. 
    The original paper is (Titsias, 2009) [1].

    [1] https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf 
    """
    def __init__(self, 
                 data: RegressionData,
                 kernel: Kernel,
                 inducing_variable: InducingPointsLike,
                 noise_variance: float = 1.0) -> None:
        """ Constructor method for the SGPR model class.

        Args:
            data (RegressionData) : training data (X, y values)
            kernel (Kernel) : kernel function for computing covariance
            inducing_inputs (InducingPointsLike) : inducing inputs
            noise_variance (float) : data noise variance 
        """
        # Inherit GP model superclass
        likelihood = Gaussian(noise_variance)
        super().__init__(kernel, likelihood, None, 1)

        # Set data attributes
        X, y = data_input_to_tensor(data)
        self.data = (X, y)
        self.num_data = X.shape[0]

        # Set inducing point attributes
        self.num_inducing = inducing_variable.shape[0]
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """ Set maximum likelihood objective as ELBO. """
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """ Compute collapsed lower bound on the marginal likelihood.
        
        Returns:
            (tf.Tensor) : collapsed evidence lower bound
        """
        X, y = self.data
        Z = self.inducing_variable.Z
        N = self.num_data
        M = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IM = tf.eye(M, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kff_diag = self.kernel(X, full_cov=False)
        kuf = self.kernel(Z, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IM
        L = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = IM + AAT
        LB = tf.linalg.cholesky(B)
        Ay = tf.linalg.matmul(A, y)
        c = tf.linalg.triangular_solve(LB, Ay, lower=True) / sigma

        # Constant term
        const = -0.5 * N * tf.math.log(2 * np.pi * sigma_sq)

        # Log determinant term
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # Quadratic term
        yTy = tf.reduce_sum(tf.square(y))
        cTc = tf.reduce_sum(tf.square(c))
        quad = -0.5 * (yTy / sigma_sq - cTc)

        # Trace term
        trace_kff = tf.reduce_sum(kff_diag)
        trace_AAT = tf.reduce_sum(tf.linalg.diag_part(AAT))
        trace = -0.5 * (trace_kff / sigma_sq - trace_AAT)

        return const + logdet + quad + trace

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
        Z = self.inducing_variable.Z
        M = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IM = tf.eye(M, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kss = self.kernel(Xnew, full_cov=full_cov)
        kuf = self.kernel(Z, X)
        kus = self.kernel(Z, Xnew)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IM
        L = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = IM + AAT
        LB = tf.linalg.cholesky(B)
        Ay = tf.linalg.matmul(A, y)
        c = tf.linalg.triangular_solve(LB, Ay, lower=True) / sigma

        # Predictive mean
        alpha = tf.linalg.triangular_solve(L, kus, lower=True)
        beta = tf.linalg.triangular_solve(LB, alpha, lower=True)
        mean = tf.linalg.matmul(beta, c, transpose_a=True)

        # Predictive (co)variance
        if full_cov:
            var = kss - tf.linalg.matmul(alpha, alpha, transpose_a=True)\
                + tf.linalg.matmul(beta, beta, transpose_a=True)          
        else:
            var = kss - tf.reduce_sum(tf.square(alpha), 0) \
                + tf.reduce_sum(tf.square(beta), 0)
            var = tf.expand_dims(var, -1)

        return (mean, var)

    def predict_g(self, 
                  Xnew: InputData, 
                  full_cov: bool = False, 
                  full_output_cov: bool = False) -> MeanAndVariance:
        """ Compute sub-process g(x). 
        
        Args:
            Xnew (InputData) : new points at which to compute predictions
            full_cov (bool) : whether to return covariance or variance
            full_output_cov (bool) : required argument for GPmodel superclass

        Returns:
            mean, var (MeanAndVariance) : mean and variance of g(x)
        """
        X, y = self.data
        Z = self.inducing_variable.Z
        M = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IM = tf.eye(M, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kus = self.kernel(Z, Xnew)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IM
        L = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = IM + AAT
        LB = tf.linalg.cholesky(B)
        Ay = tf.linalg.matmul(A, y)
        c = tf.linalg.triangular_solve(LB, Ay, lower=True) / sigma

        # Mean
        alpha = tf.linalg.triangular_solve(L, kus, lower=True)
        beta = tf.linalg.triangular_solve(LB, alpha, lower=True)
        mean = tf.linalg.matmul(beta, c, transpose_a=True)

        # (Co)variance
        if full_cov:
            var = tf.linalg.matmul(beta, beta, transpose_a=True)
        else:
            var = tf.reduce_sum(tf.square(beta), 0)
            var = tf.expand_dims(var, -1)

        return (mean, var)

    def predict_h(self, 
                  Xnew: InputData, 
                  full_cov: bool = False, 
                  full_output_cov: bool = False) -> MeanAndVariance:
        """ Compute sub-process g(x). 
        
        Args:
            Xnew (InputData) : new points at which to compute predictions
            full_cov (bool) : whether to return covariance or variance
            full_output_cov (bool) : required argument for GPmodel superclass

        Returns:
            mean, var (MeanAndVariance) : mean and variance of h(x)
        """
        X, _ = self.data
        Z = self.inducing_variable.Z
        M = self.num_inducing

        # Identity matrices
        IM = tf.eye(M, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kss = self.kernel(Xnew, full_cov=full_cov)
        kus = self.kernel(Z, Xnew)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IM
        L = tf.linalg.cholesky(kuu)

        # Mean
        mean = tf.zeros_like(Xnew)

        # (Co)variance
        alpha = tf.linalg.triangular_solve(L, kus, lower=True)
        if full_cov:
            var = kss - tf.linalg.matmul(alpha, alpha, transpose_a=True)
        else:
            var = kss - tf.reduce_sum(tf.square(alpha), 0)
            var = tf.expand_dims(var, -1)

        return (mean, var)

    @property
    def Z(self) -> tf.Tensor:
        """ Inducing inputs Z. """
        return self.inducing_variable.Z[:, :]

    @property
    def mu(self) -> tf.Tensor:
        """ Variational parameter mu. """
        X, y = self.data
        Z = self.inducing_variable.Z
        M = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IM = tf.eye(M, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IM
        L = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = IM + AAT
        LB = tf.linalg.cholesky(B)
        Ay = tf.linalg.matmul(A, y)
        c = tf.linalg.triangular_solve(LB, Ay, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(tf.transpose(LB), c, lower=False)

        return tf.linalg.matmul(L, tmp1)

    @property
    def Su(self) -> tf.Tensor:
        """ Variational parameter Su. """
        X, _ = self.data
        Z = self.inducing_variable.Z
        M = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IM = tf.eye(M, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IM
        L = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = IM + tf.linalg.matmul(A, A, transpose_b=True)
        LB = tf.linalg.cholesky(B)
        tmp1 = tf.linalg.triangular_solve(LB, tf.transpose(L), lower=True)
        
        return tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
    