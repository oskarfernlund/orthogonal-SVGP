#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orthogonal sparse Gaussian process regression model.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Tuple

import numpy as np
import tensorflow as tf

from gpflow.base import InputData, MeanAndVariance, RegressionData
from gpflow.config import default_float, default_jitter
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models import GPModel, InternalDataTrainingLossMixin
from gpflow.models.util import data_input_to_tensor, inducingpoint_wrapper
from gpflow.models.util import InducingPointsLike


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
                 inducing_inputs: Tuple[InducingPointsLike, InducingPointsLike],
                 noise_variance: float = 1.0, 
                 method: str = "SOLVE-GP") -> None:
        """ Constructor method for the Orthogonal SGPR model class.

        Default orthogonal parameterisation is SOLVE-GP, but ODVGP may be used 
        by setting the method argument to "ODVGP".

        Args:
            data (RegressionData) : Training data (X, y values)
            kernel (Kernel) : Kernel function for computing covariance
            inducing_inputs (tuple of InducingPointsLike) : Inducing point sets
            noise_variance (float) : Data noise variance 
            method (str) : Orthogonal parameterisation ("SOLVE-GP" or "ODVGP")
        """
        # Inherit GP model superclass
        likelihood = Gaussian(noise_variance)
        super().__init__(kernel, likelihood, None, 1)

        # Set data attributes
        X, y = data_input_to_tensor(data)
        self.data = (X, y)
        self.num_data = X.shape[0]

        # Set inducing point attributes
        Z, O = inducing_inputs
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
            (tf.Tensor) : Collapsed evidence lower bound
        """
        X, y = self.data
        Z, O = self.inducing_variable_1.Z, self.inducing_variable_2.Z
        N = self.num_data
        Mu, Mv = self.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Identity matrices
        IM = tf.eye(Mu + Mv, dtype=default_float())
        IMu = tf.eye(Mu, dtype=default_float())
        IMv = tf.eye(Mv, dtype=default_float())

        # Zero matrices
        Zuv = tf.zeros((Mu, Mv), dtype=default_float())
        Zvu = tf.zeros((Mv, Mu), dtype=default_float())

        # Covariances & Cholesky decompositions
        kff_diag = self.kernel(X, full_cov=False)
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + default_jitter() * IMu
        kvv = self.kernel(O) + default_jitter() * IMv
        Lu = tf.linalg.cholesky(kuu)
        tmp1 = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        tmp2 = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        cvf = kvf - tf.linalg.matmul(tmp1, tmp2, transpose_a=True)
        cvv = kvv - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, V, lower=True) / sigma
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
            Xnew (InputData) : New points at which to compute predictions
            full_cov (bool) : Whether to return covariance or variance
            full_output_cov (bool) : Required argument for GPmodel superclass

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
        IN = tf.eye(N, dtype=default_float())
        IM = tf.eye(Mu + Mv, dtype=default_float())
        IMu = tf.eye(Mu, dtype=default_float())
        IMv = tf.eye(Mv, dtype=default_float())

        # Zero matrices
        Zuv = tf.zeros((Mu, Mv), dtype=default_float())
        Zvu = tf.zeros((Mv, Mu), dtype=default_float())

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kus = self.kernel(Z, Xnew)
        kvf = self.kernel(O, X)
        kvs = self.kernel(O, Xnew)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + default_jitter() * IMu
        kvv = self.kernel(O) + default_jitter() * IMv
        Lu = tf.linalg.cholesky(kuu)
        tmp1 = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        tmp2 = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        tmp3 = tf.linalg.triangular_solve(Lu, kus, lower=True)
        cvv = kvv - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        cvf = kvf - tf.linalg.matmul(tmp1, tmp2, transpose_a=True)
        cvs = kvs - tf.linalg.matmul(tmp1, tmp3, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)
        Vs = tf.concat([kus, cvs], 0)

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
        Gu = tf.linalg.triangular_solve(tf.transpose(Lu), Eu, lower=False)
        Gv = tf.linalg.triangular_solve(tf.transpose(Lv), Ev, lower=False)
        
        if self.method == "SOLVE-GP":
            Hs = tf.linalg.triangular_solve(L, Vs, lower=True)
            Js = tf.linalg.triangular_solve(LB, Hs, lower=True)
        elif self.method == "ODVGP":
            Hs = tf.linalg.triangular_solve(Lu, kus, lower=True)
            Js = tf.linalg.triangular_solve(LBu, Hs, lower=True)

        # Predictive mean
        tmp4 = tf.linalg.matmul(tf.linalg.matmul(kus, Gu, transpose_a=True), Au)
        tmp5 = tf.linalg.matmul(tf.linalg.matmul(cvs, Gv, transpose_a=True), Av)
        tmp6 = tf.linalg.matmul(tf.linalg.matmul(tmp4, Dv), y) / sigma
        tmp7 = tf.linalg.matmul(tf.linalg.matmul(tmp5, Du), y) / sigma
        mean = tmp6 + tmp7

        # Predictive variance
        if full_cov:
            var = (self.kernel(Xnew)
                + tf.linalg.matmul(Js, Js, transpose_a=True)
                - tf.linalg.matmul(Hs, Hs, transpose_a=True))
            var = tf.tile(var[None, ...], [1, 1, 1])
        else:
            var = (self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(Js), 0)
                - tf.reduce_sum(tf.square(Hs), 0))
            var = tf.tile(var[:, None], [1, 1])

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
        IN = tf.eye(N, dtype=default_float())
        IM = tf.eye(Mu + Mv, dtype=default_float())
        IMu = tf.eye(Mu, dtype=default_float())
        IMv = tf.eye(Mv, dtype=default_float())

        # Zero matrices
        Zuv = tf.zeros((Mu, Mv), dtype=default_float())
        Zvu = tf.zeros((Mv, Mu), dtype=default_float())

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + default_jitter() * IMu
        kvv = self.kernel(O) + default_jitter() * IMv
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
        IMu = tf.eye(Mu, dtype=default_float())

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kuu = self.kernel(Z) + default_jitter() * IMu
        Lu = tf.linalg.cholesky(kuu)

        # Intermediate matrices
        Au = tf.linalg.triangular_solve(Lu, kuf, lower=True) / sigma
        Bu = IMu + tf.linalg.matmul(Au, Au, transpose_b=True)
        LBu = tf.linalg.cholesky(Bu)
        Fu = tf.linalg.triangular_solve(LBu, tf.transpose(Lu), lower=True)
        
        return tf.linalg.matmul(Fu, Fu, transpose_a=True)

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
        IN = tf.eye(N, dtype=default_float())
        IM = tf.eye(Mu + Mv, dtype=default_float())
        IMu = tf.eye(Mu, dtype=default_float())
        IMv = tf.eye(Mv, dtype=default_float())

        # Zero matrices
        Zuv = tf.zeros((Mu, Mv), dtype=default_float())
        Zvu = tf.zeros((Mv, Mu), dtype=default_float())

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + default_jitter() * IMu
        kvv = self.kernel(O) + default_jitter() * IMv
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
        IMu = tf.eye(Mu, dtype=default_float())
        IMv = tf.eye(Mv, dtype=default_float())

        # Covariances & Cholesky decompositions
        kuf = self.kernel(Z, X)
        kvf = self.kernel(O, X)
        kuv = self.kernel(Z, O)
        kvf = self.kernel(O, X)
        kuu = self.kernel(Z) + default_jitter() * IMu
        kvv = self.kernel(O) + default_jitter() * IMv
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
        Fv = tf.linalg.triangular_solve(LBv, tf.transpose(Lv), lower=True)
        
        return tf.linalg.matmul(Fv, Fv, transpose_a=True)
