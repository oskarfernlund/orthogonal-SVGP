#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experimental model classes for probing edge cases of orthogonal SGPR.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import tensorflow as tf

from osvgp.base import InputData, RegressionData, MeanAndVariance
from osvgp.base import InducingPointsLike, OrthogonalInducingPointsLike
from osvgp.config import DEFAULT_FLOAT, DEFAULT_JITTER
from osvgp.kernels import Kernel
from osvgp.likelihoods import Gaussian
from osvgp.models import GPModel, InternalDataTrainingLossMixin
from osvgp.util import data_input_to_tensor, inducingpoint_wrapper


# =============================================================================
#  CLASSES
# =============================================================================

class OSGPRBoundGap(GPModel, InternalDataTrainingLossMixin):
    """ Model comprised of both an SGPR model and OSGPR model. 
    
    The optimisation objective is to maximise the difference between the SGPR 
    and OSGPR lower bounds.
    """
    def __init__(self, 
                 data: RegressionData,
                 kernel: Kernel,
                 inducing_variable: InducingPointsLike,
                 num_orthogonal: int,
                 noise_variance: float = 1.0) -> None:
        """ Constructor method for the OSGPRBoundGap class.

        Args:
            data (RegressionData) : training data (X, y values)
            kernel (Kernel) : kernel function for computing covariance
            inducing_variable (InducingPointsLike) : inducing inputs
            num_regular (int) : number of regular inducing points
            num_orthogonal (int) : number of orthogonal inducing points
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
        assert num_orthogonal <= self.num_inducing
        self.num_orthogonal = num_orthogonal
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """ Set maximum likelihood objective as the gap between ELBO's. """
        return self.sgpr_elbo() - self.osgpr_elbo()

    def sgpr_elbo(self) -> tf.Tensor:
        """ Compute SGPR lower bound.
        
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

    def osgpr_elbo(self) -> tf.Tensor:
        """ Compute OSGPR lower bound.
        
        Returns:
            (tf.Tensor) : collapsed evidence lower bound
        """
        X, y = self.data
        Z = self.inducing_variable.Z[:self.num_inducing - self.num_orthogonal]
        O = self.inducing_variable.Z[self.num_inducing - self.num_orthogonal:]
        N = self.num_data
        Mu = self.num_inducing - self.num_orthogonal
        Mv = self.num_orthogonal
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IMu = tf.eye(Mu, dtype=DEFAULT_FLOAT)
        IMv = tf.eye(Mv, dtype=DEFAULT_FLOAT)

        # Zero matrices
        Zuv = tf.zeros((Mu, Mv), dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kff_diag = self.kernel(X, full_cov=False)
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        kvv = self.kernel(O) + DEFAULT_JITTER * IMv
        Lu = tf.linalg.cholesky(kuu)
        delta_v = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        delta_f = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        cvf = kvf - tf.linalg.matmul(delta_v, delta_f, transpose_a=True)
        cvv = kvv - tf.linalg.matmul(delta_v, delta_v, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)

        # Intermediate matrices
        Au = delta_f / sigma
        Av = tf.linalg.triangular_solve(Lv, cvf, lower=True) / sigma
        A = tf.concat([Au, Av], 0)
        Bu_prime = tf.linalg.matmul(Au, Au, transpose_b=True)
        Bv_prime = tf.linalg.matmul(Av, Av, transpose_b=True)
        Buv = tf.linalg.matmul(Au, Av, transpose_b=True)
        AAT = tf.concat([tf.concat([Bu_prime, Buv], 1),
                         tf.concat([tf.transpose(Buv), Bv_prime], 1)], 0) 
        Bu = IMu + Bu_prime
        Bv = IMv + Bv_prime
        LBu = tf.linalg.cholesky(Bu)
        gamma = tf.linalg.triangular_solve(LBu, Buv, lower=True)
        Q = Bv - tf.linalg.matmul(gamma, gamma, transpose_a=True)
        LQ = tf.linalg.cholesky(Q)
        LB = tf.concat([tf.concat([LBu, Zuv], 1),
                        tf.concat([tf.transpose(gamma), LQ], 1)], 0) 
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
        return tf.zeros_like(Xnew[:, 0])

    def sgpr_predict_f(self, 
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

    def osgpr_predict_f(self, 
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
        Z = self.inducing_variable.Z[:self.num_inducing - self.num_orthogonal]
        O = self.inducing_variable.Z[self.num_inducing - self.num_orthogonal:]
        N = self.num_data
        Mu = self.num_inducing - self.num_orthogonal
        Mv = self.num_orthogonal
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IN = tf.eye(N, dtype=DEFAULT_FLOAT)
        IM = tf.eye(Mu + Mv, dtype=DEFAULT_FLOAT)
        IMu = tf.eye(Mu, dtype=DEFAULT_FLOAT)
        IMv = tf.eye(Mv, dtype=DEFAULT_FLOAT)

        # Zero matrices
        Zuv = tf.zeros((Mu, Mv), dtype=DEFAULT_FLOAT)
        Zvu = tf.zeros((Mv, Mu), dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kss = self.kernel(Xnew, full_cov=full_cov)
        kuf = self.kernel(Z, X)
        kus = self.kernel(Z, Xnew)
        kvf = self.kernel(O, X)
        kvs = self.kernel(O, Xnew)
        kuv = self.kernel(Z, O)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        kvv = self.kernel(O) + DEFAULT_JITTER * IMv
        Lu = tf.linalg.cholesky(kuu)
        delta_v = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        delta_f = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        delta_s = tf.linalg.triangular_solve(Lu, kus, lower=True)
        cvf = kvf - tf.linalg.matmul(delta_v, delta_f, transpose_a=True)
        cvv = kvv - tf.linalg.matmul(delta_v, delta_v, transpose_a=True)
        cvs = kvs - tf.linalg.matmul(delta_v, delta_s, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, V, lower=True) / sigma
        Au = tf.linalg.triangular_solve(Lu, kuf, lower=True) / sigma
        Av = tf.linalg.triangular_solve(Lv, cvf, lower=True) / sigma
        B = IM + tf.linalg.matmul(A, A, transpose_b=True)
        Bu = IMu + tf.linalg.matmul(Au, Au, transpose_b=True)
        Bv = IMv + tf.linalg.matmul(Av, Av, transpose_b=True)
        LB = tf.linalg.cholesky(B)
        LBu = tf.linalg.cholesky(Bu)
        LBv = tf.linalg.cholesky(Bv)
        C = tf.linalg.triangular_solve(LB, A, lower=True)
        Cu = tf.linalg.triangular_solve(LBu, Au, lower=True)
        Cv = tf.linalg.triangular_solve(LBv, Av, lower=True)
        D = IN - tf.linalg.matmul(C, C, transpose_a=True)
        Du = IN - tf.linalg.matmul(Cu, Cu, transpose_a=True)
        Dv = IN - tf.linalg.matmul(Cv, Cv, transpose_a=True)
        Eu = IMu - tf.linalg.matmul(tf.linalg.matmul(Au, D), Au, transpose_b=True)
        Ev = IMv - tf.linalg.matmul(tf.linalg.matmul(Av, D), Av, transpose_b=True)
        Fuv = tf.linalg.matmul(tf.linalg.matmul(Eu, Au), Dv) / sigma 
        Fvu = tf.linalg.matmul(tf.linalg.matmul(Ev, Av), Du) / sigma 
        alpha_u = tf.linalg.triangular_solve(Lu, kus, lower=True)
        alpha_v = tf.linalg.triangular_solve(Lv, cvs, lower=True)
        beta_u = tf.linalg.triangular_solve(LBu, alpha_u, lower=True)
        beta_v = tf.linalg.triangular_solve(LBv, alpha_v, lower=True)

        # Predictive mean
        mean = tf.linalg.matmul(tf.linalg.matmul(alpha_u, Fuv, transpose_a=True), y) \
            + tf.linalg.matmul(tf.linalg.matmul(alpha_v, Fvu, transpose_a=True), y)

        # Predictive (co)variance
        if full_cov:
            var = kss \
                - tf.linalg.matmul(alpha_u, alpha_u, transpose_a=True) \
                - tf.linalg.matmul(alpha_v, alpha_v, transpose_a=True) \
                + tf.linalg.matmul(beta_u, beta_u, transpose_a=True) \
                + tf.linalg.matmul(beta_v, beta_v, transpose_a=True)
        else:
            var = kss \
                - tf.reduce_sum(tf.square(alpha_u), 0) \
                - tf.reduce_sum(tf.square(alpha_v), 0) \
                + tf.reduce_sum(tf.square(beta_u), 0) \
                + tf.reduce_sum(tf.square(beta_v), 0)
            var = tf.expand_dims(var, -1)

        return (mean, var)

    @property
    def SGPR_Z(self) -> tf.Tensor:
        """ Inducing inputs Z for SGPR. """
        return self.inducing_variable.Z[:, :]

    @property
    def OSGPR_Z(self) -> tf.Tensor:
        """ Inducing inputs Z for OSGPR. """
        return self.inducing_variable.Z[:self.num_inducing-self.num_orthogonal, :]

    @property
    def OSGPR_O(self) -> tf.Tensor:
        """ Orthogonal inducing inputs O for OSGPR. """
        return self.inducing_variable.Z[self.num_inducing-self.num_orthogonal:, :]
