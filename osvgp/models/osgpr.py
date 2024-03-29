#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orthogonal sparse Gaussian process regression model.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

import numpy as np
import tensorflow as tf

from osvgp.base import InputData, RegressionData, MeanAndVariance
from osvgp.base import OrthogonalInducingPointsLike
from osvgp.config import DEFAULT_FLOAT, DEFAULT_JITTER
from osvgp.kernels import Kernel
from osvgp.likelihoods import Gaussian
from osvgp.models import GPModel, InternalDataTrainingLossMixin
from osvgp.util import data_input_to_tensor, inducingpoint_wrapper


# =============================================================================
#  CLASSES
# =============================================================================

class OSGPR(GPModel, InternalDataTrainingLossMixin):
    """ Orthogonal sparse variational Gaussian process regression. 
    
    This is an extension of SGPR (Titsias, 2009) [1] with two orthogonal sets 
    of inducing points.
    
    There is an option to use the ODVGP parameterisation [2] or the SOLVE-GP 
    parameterisation [3]. The collapsed bound is the same for both, but ODVGP 
    restricts the variational covariance parameter Sv to be equal to the prior 
    conditional Cvv = Kvv - Kvu [Kuu]^-1 Kuv.

    [1] https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf
    [2] https://arxiv.org/pdf/1809.08820.pdf
    [3] https://arxiv.org/pdf/1910.10596.pdf
    """
    def __init__(self, 
                 data: RegressionData,
                 kernel: Kernel,
                 inducing_variable: OrthogonalInducingPointsLike,
                 noise_variance: float = 1.0, 
                 method: str = "SOLVE-GP") -> None:
        """ Constructor method for the Orthogonal SGPR model class.

        Default orthogonal parameterisation is SOLVE-GP, but ODVGP may be used 
        by setting the method argument to "ODVGP".

        Args:
            data (RegressionData) : training data (X, y values)
            kernel (Kernel) : kernel function for computing covariance
            inducing_inputs (OrthogonalInducingPointsLike) : inducing inputs
            noise_variance (float) : data noise variance 
            method (str) : orthogonal parameterisation ("SOLVE-GP" or "ODVGP")
        """
        # Inherit GP model superclass
        likelihood = Gaussian(noise_variance)
        super().__init__(kernel, likelihood, None, 1)

        # Set data attributes
        X, y = data_input_to_tensor(data)
        self.data = (X, y)
        self.num_data = X.shape[0]

        # Set inducing point attributes
        Z, O = inducing_variable
        self.num_inducing = (Z.shape[0], O.shape[0])
        self.inducing_variable_1 = inducingpoint_wrapper(Z)
        self.inducing_variable_2 = inducingpoint_wrapper(O)

        # Set orthogonal method
        if method not in ["SOLVE-GP", "ODVGP"]:
            raise ValueError("Unrecognised orthogonal method!")
        else:
            self.method = method

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """ Set maximum likelihood objective as ELBO. """
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """ Compute collapsed lower bound on the marginal likelihood.
        
        Returns:
            (tf.Tensor) : collapsed evidence lower bound
        """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        # IM = tf.eye(Mu + Mv, dtype=DEFAULT_FLOAT)
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
        """ Predict the mean and variance of the latent function at some points. 
        
        Args:
            Xnew (InputData) : new points at which to compute predictions
            full_cov (bool) : whether to return covariance or variance
            full_output_cov (bool) : required argument for GPmodel superclass

        Returns:
            mean, var (MeanAndVariance) : Predictive mean and variance
        """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
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
                + tf.linalg.matmul(beta_u, beta_u, transpose_a=True)
            if self.method == "SOLVE-GP":
                var -= tf.linalg.matmul(alpha_v, alpha_v, transpose_a=True)
                var += tf.linalg.matmul(beta_v, beta_v, transpose_a=True)
        else:
            var = kss \
                - tf.reduce_sum(tf.square(alpha_u), 0) \
                + tf.reduce_sum(tf.square(beta_u), 0)
            if self.method == "SOLVE-GP":
                var -= tf.reduce_sum(tf.square(alpha_v), 0)
                var += tf.reduce_sum(tf.square(beta_v), 0)
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
            mean, var (MeanAndVariance) : Mean and variance of g(x)
        """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
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
        kuf = self.kernel(Z, X)
        kus = self.kernel(Z, Xnew)
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
        Cv = tf.linalg.triangular_solve(LBv, Av, lower=True)
        D = IN - tf.linalg.matmul(C, C, transpose_a=True)
        Dv = IN - tf.linalg.matmul(Cv, Cv, transpose_a=True)
        Eu = IMu - tf.linalg.matmul(tf.linalg.matmul(Au, D), Au, transpose_b=True)
        Fuv = tf.linalg.matmul(tf.linalg.matmul(Eu, Au), Dv) / sigma 

        alpha_u = tf.linalg.triangular_solve(Lu, kus, lower=True)
        beta_u = tf.linalg.triangular_solve(LBu, alpha_u, lower=True)

        # Mean
        mean = tf.linalg.matmul(tf.linalg.matmul(alpha_u, Fuv, transpose_a=True), y)

        # (Co)variance
        if full_cov:
            var = tf.linalg.matmul(beta_u, beta_u, transpose_a=True)
        else:
            var = tf.reduce_sum(tf.square(beta_u), 0)
            var = tf.expand_dims(var, -1)

        return (mean, var)

    def predict_h(self, 
                  Xnew: InputData, 
                  full_cov: bool = False, 
                  full_output_cov: bool = False) -> MeanAndVariance:
        """ Compute sub-process h(x).
        
        Args:
            Xnew (InputData) : new points at which to compute predictions
            full_cov (bool) : whether to return covariance or variance
            full_output_cov (bool) : required argument for GPmodel superclass

        Returns:
            mean, var (MeanAndVariance) : mean and variance of h(x)
        """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
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
        D = IN - tf.linalg.matmul(C, C, transpose_a=True)
        Du = IN - tf.linalg.matmul(Cu, Cu, transpose_a=True)
        Ev = IMv - tf.linalg.matmul(tf.linalg.matmul(Av, D), Av, transpose_b=True)
        Fvu = tf.linalg.matmul(tf.linalg.matmul(Ev, Av), Du) / sigma 
        alpha_v = tf.linalg.triangular_solve(Lv, cvs, lower=True)
        beta_v = tf.linalg.triangular_solve(LBv, alpha_v, lower=True)

        # Mean
        mean = tf.linalg.matmul(tf.linalg.matmul(alpha_v, Fvu, transpose_a=True), y)

        # (Co)variance
        if full_cov:
            var = tf.zeros((Xnew.shape[0], Xnew.shape[0]), dtype=DEFAULT_FLOAT)
            if self.method == "SOLVE-GP":
                var += tf.linalg.matmul(beta_v, beta_v, transpose_a=True)
        else:
            var = tf.zeros_like(Xnew)
            if self.method == "SOLVE-GP":
                var += tf.reduce_sum(tf.square(beta_v), 0)

        return (mean, var)

    def predict_l(self, 
                  Xnew: InputData, 
                  full_cov: bool = False, 
                  full_output_cov: bool = False) -> MeanAndVariance:
        """ Compute sub-process l(x). 
        
        Args:
            Xnew (InputData) : new points at which to compute predictions
            full_cov (bool) : whether to return covariance or variance
            full_output_cov (bool) : required argument for GPmodel superclass

        Returns:
            mean, var (MeanAndVariance) : mean and variance of l(x)
        """
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        Mu, Mv = self.num_inducing

        # Identity matrices
        IMu = tf.eye(Mu, dtype=DEFAULT_FLOAT)
        IMv = tf.eye(Mv, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kss = self.kernel(Xnew, full_cov=full_cov)
        kus = self.kernel(Z, Xnew)
        kvs = self.kernel(O, Xnew)
        kuv = self.kernel(Z, O)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        kvv = self.kernel(O) + DEFAULT_JITTER * IMv
        Lu = tf.linalg.cholesky(kuu)
        delta_v = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        delta_s = tf.linalg.triangular_solve(Lu, kus, lower=True)
        cvv = kvv - tf.linalg.matmul(delta_v, delta_v, transpose_a=True)
        cvs = kvs - tf.linalg.matmul(delta_v, delta_s, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        alpha_u = tf.linalg.triangular_solve(Lu, kus, lower=True)
        alpha_v = tf.linalg.triangular_solve(Lv, cvs, lower=True)

        # Mean
        mean = tf.zeros_like(Xnew)

        # (Co)variance        
        if full_cov:
            var = kss - tf.linalg.matmul(alpha_u, alpha_u, transpose_a=True)
            if self.method == "SOLVE-GP":
                var -= tf.linalg.matmul(alpha_v, alpha_v, transpose_a=True)
        else:
            var = kss - tf.reduce_sum(tf.square(alpha_u), 0)
            if self.method == "SOLVE-GP":
                var -= tf.reduce_sum(tf.square(alpha_v), 0)
            var = tf.expand_dims(var, -1)

        return (mean, var)

    @property
    def Z(self) -> tf.Tensor:
        """ Inducing inputs Z. """
        return self.inducing_variable_1.Z[:, :]

    @property
    def O(self) -> tf.Tensor:
        """ Inducing inputs O. """
        return self.inducing_variable_2.Z[:, :]

    @property
    def mu(self) -> tf.Tensor:
        """ Variational parameter mu. """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
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
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        kvv = self.kernel(O) + DEFAULT_JITTER * IMv
        Lu = tf.linalg.cholesky(kuu)
        tmp1 = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        tmp2 = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        cvv = kvv - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        cvf = kvf - tf.linalg.matmul(tmp1, tmp2, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, V, lower=True) / sigma
        Au = tf.linalg.triangular_solve(Lu, kuf, lower=True) / sigma
        Av = tf.linalg.triangular_solve(Lv, cvf, lower=True) / sigma
        B = IM + tf.linalg.matmul(A, A, transpose_b=True)
        Bv = IMv + tf.linalg.matmul(Av, Av, transpose_b=True)
        LB = tf.linalg.cholesky(B)
        LBv = tf.linalg.cholesky(Bv)
        C = tf.linalg.triangular_solve(LB, A, lower=True)
        Cv = tf.linalg.triangular_solve(LBv, Av, lower=True)
        D = IN - tf.linalg.matmul(C, C, transpose_a=True)
        Dv = IN - tf.linalg.matmul(Cv, Cv, transpose_a=True)
        Eu = IMu - tf.linalg.matmul(tf.linalg.matmul(Au, D), Au, transpose_b=True)
        tmp3 = tf.linalg.matmul(tf.linalg.matmul(Lu, Eu), Au)
        
        return tf.linalg.matmul(tf.linalg.matmul(tmp3, Dv), y) / sigma

    @property
    def Su(self) -> tf.Tensor:
        """ Variational parameter Su. """
        X, _ = self.data
        Z = self.inducing_variable_1.Z
        Mu, _ = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IMu = tf.eye(Mu, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        Lu = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        Au = tf.linalg.triangular_solve(Lu, kuf, lower=True) / sigma
        Bu = IMu + tf.linalg.matmul(Au, Au, transpose_b=True)
        LBu = tf.linalg.cholesky(Bu)
        Gu = tf.linalg.triangular_solve(LBu, tf.transpose(Lu), lower=True)
        
        return tf.linalg.matmul(Gu, Gu, transpose_a=True)

    @property
    def mv(self) -> tf.Tensor:
        """ Variational parameter mv. """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
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
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        kvv = self.kernel(O) + DEFAULT_JITTER * IMv
        Lu = tf.linalg.cholesky(kuu)
        tmp1 = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        tmp2 = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        cvv = kvv - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        cvf = kvf - tf.linalg.matmul(tmp1, tmp2, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, V, lower=True) / sigma
        Au = tf.linalg.triangular_solve(Lu, kuf, lower=True) / sigma
        Av = tf.linalg.triangular_solve(Lv, cvf, lower=True) / sigma
        B = IM + tf.linalg.matmul(A, A, transpose_b=True)
        Bu = IMu + tf.linalg.matmul(Au, Au, transpose_b=True)
        LB = tf.linalg.cholesky(B)
        LBu = tf.linalg.cholesky(Bu)
        C = tf.linalg.triangular_solve(LB, A, lower=True)
        Cu = tf.linalg.triangular_solve(LBu, Au, lower=True)
        D = IN - tf.linalg.matmul(C, C, transpose_a=True)
        Du = IN - tf.linalg.matmul(Cu, Cu, transpose_a=True)
        Ev = IMv - tf.linalg.matmul(tf.linalg.matmul(Av, D), Av, transpose_b=True)
        tmp3 = tf.linalg.matmul(tf.linalg.matmul(Lv, Ev), Av)

        return tf.linalg.matmul(tf.linalg.matmul(tmp3, Du), y) / sigma

    @property
    def Sv(self) -> tf.Tensor:
        """ Variational parameter Sv. """
        X, _ = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        Mu, Mv = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IMu = tf.eye(Mu, dtype=DEFAULT_FLOAT)
        IMv = tf.eye(Mv, dtype=DEFAULT_FLOAT)

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + DEFAULT_JITTER * IMu
        kvv = self.kernel(O) + DEFAULT_JITTER * IMv
        Lu = tf.linalg.cholesky(kuu)
        tmp1 = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        tmp2 = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        cvv = kvv - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        cvf = kvf - tf.linalg.matmul(tmp1, tmp2, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)

        # Intermediate matrices
        Av = tf.linalg.triangular_solve(Lv, cvf, lower=True) / sigma
        Bv = IMv + tf.linalg.matmul(Av, Av, transpose_b=True)
        LBv = tf.linalg.cholesky(Bv)
        Gv = tf.linalg.triangular_solve(LBv, tf.transpose(Lv), lower=True)
        
        return tf.linalg.matmul(Gv, Gv, transpose_a=True)
