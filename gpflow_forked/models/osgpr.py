# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .. import likelihoods, posteriors
from ..base import InputData, MeanAndVariance, RegressionData
from ..config import default_float, default_jitter
from ..covariances.dispatch import Kuf, Kuu, Kuv
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
# from ..mean_functions import MeanFunction
from ..utilities import add_noise_cov, to_default_float
from .model import GPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import InducingPointsLike, data_input_to_tensor, inducingpoint_wrapper


class OSGPR(GPModel, InternalDataTrainingLossMixin):
    """ Orthogonal sparse variational Gaussian process regression. 
    
    Option to use SOLVE-GP or ODVGP methods. The collapsed bound is the same 
    for both, but ODVGP restricts the variance of the bing bong.
    
    """
    Common1 = namedtuple("Common1", ["L", "A", "AAT", "LB", "c"])
    Common2 = namedtuple("Common2", ["d", "Lu", "Lv", "Au", "Av", "du", "dv", "eu", "ev"])

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable_1: InducingPointsLike,
        inducing_variable_2: InducingPointsLike,
        noise_variance: float = 1.0,
        method: str = "SOLVE-GP"
    ) -> None:
        """ Bing Bong """
        # Inherit GP model superclass
        likelihood = likelihoods.Gaussian(noise_variance)
        super().__init__(kernel, likelihood, None, 1)

        # Set other attributes
        X, y = data_input_to_tensor(data)
        self.data = X, y
        self.num_data = X.shape[0]
        self.inducing_variable_1: InducingPoints = \
            inducingpoint_wrapper(inducing_variable_1)
        self.inducing_variable_2: InducingPoints = \
            inducingpoint_wrapper(inducing_variable_2)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        """ Set maximum likelihood objective as ELBO. """
        return self.elbo()

    def _common_calculations_1(self) -> "OSGPR.Common1":
        """ Compute common matrices used in ELBO computation. """
        X, y = self.data
        Z = self.inducing_variable_1
        O = self.inducing_variable_2
        num_inducing_1 = Z.num_inducing
        num_inducing_2 = O.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Covariances & Cholesky decompositions
        kuf = Kuf(Z, self.kernel, X)
        kuv = Kuv(Z, self.kernel, O)
        kvf = Kuf(O, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        kvv = Kuu(O, self.kernel, jitter=default_jitter())
        Lu = tf.linalg.cholesky(kuu)
        Tuf = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        Tuv = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        cvf = kvf - tf.linalg.matmul(Tuv, Tuf, transpose_a=True)
        cvv = kvv - tf.linalg.matmul(Tuv, Tuv, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        Zuv = tf.zeros((num_inducing_1, num_inducing_2), dtype=Lu.dtype)
        Zvu = tf.zeros((num_inducing_2, num_inducing_1), dtype=Lu.dtype)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, V, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)
        Ay = tf.linalg.matmul(A, y)
        c = tf.linalg.triangular_solve(LB, Ay, lower=True) / sigma

        return self.Common1(L, A, AAT, LB, c)

    def _common_calculations_2(self) -> "OSGPR.Common2":
        """ Compute common matrices used in predictive equations. """
        X, _ = self.data
        Z = self.inducing_variable_1
        O = self.inducing_variable_2
        num_inducing_1 = Z.num_inducing
        num_inducing_2 = O.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        # Covariances & Cholesky decompositions
        kuf = Kuf(Z, self.kernel, X)
        kuv = Kuv(Z, self.kernel, O)
        kvf = Kuf(O, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        kvv = Kuu(O, self.kernel, jitter=default_jitter())
        Lu = tf.linalg.cholesky(kuu)
        Tuf = tf.linalg.triangular_solve(Lu, kuf, lower=True)
        Tuv = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        cvf = kvf - tf.linalg.matmul(Tuv, Tuf, transpose_a=True)
        cvv = kvv - tf.linalg.matmul(Tuv, Tuv, transpose_a=True)
        Lv = tf.linalg.cholesky(cvv)
        Zuv = tf.zeros((num_inducing_1, num_inducing_2), dtype=Lu.dtype)
        Zvu = tf.zeros((num_inducing_2, num_inducing_1), dtype=Lu.dtype)
        L = tf.concat([tf.concat([Lu, Zuv], 1), tf.concat([Zvu, Lv], 1)], 0) 
        V = tf.concat([kuf, cvf], 0)

        # Intermediate matrices
        A = tf.linalg.triangular_solve(L, V, lower=True) / sigma
        Au = tf.linalg.triangular_solve(Lu, kuf, lower=True) / sigma
        Av = tf.linalg.triangular_solve(Lv, cvf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        AuAuT = tf.linalg.matmul(Au, Au, transpose_b=True)
        AvAvT = tf.linalg.matmul(Av, Av, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        Bu = add_noise_cov(AuAuT, tf.cast(1.0, AuAuT.dtype))
        Bv = add_noise_cov(AvAvT, tf.cast(1.0, AvAvT.dtype))
        LB = tf.linalg.cholesky(B)
        LBu = tf.linalg.cholesky(Bu)
        LBv = tf.linalg.cholesky(Bv)
        d = tf.linalg.triangular_solve(LB, A, lower=True)
        du = tf.linalg.triangular_solve(LBu, Au, lower=True)
        dv = tf.linalg.triangular_solve(LBv, Av, lower=True)
        eu = tf.linalg.triangular_solve(LBu, tf.transpose(Lu), lower=True)
        ev = tf.linalg.triangular_solve(LBv, tf.transpose(Lv), lower=True)

        return self.Common2(d, Lu, Lv, Au, Av, du, dv, eu, ev)

    def elbo(self) -> tf.Tensor:
        """ Compute a lower bound on the marginal likelihood. """
        X, y = self.data
        num_data = self.num_data
        kff_diag = self.kernel(X, full_cov=False)
        sigma_sq = self.likelihood.variance
        common1 = self._common_calculations_1()
        AAT, LB, c = common1.AAT, common1.LB, common1.c

        # Constant term
        const = -0.5 * num_data * tf.math.log(2 * np.pi * sigma_sq)

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

    def predict_f(self, Xnew: InputData, full_cov: bool = False,
        full_output_cov: bool = False) -> MeanAndVariance:
        """ Compute the mean and variance of the latent function at some new points. """
        X, y = self.data
        Z = self.inducing_variable_1
        O = self.inducing_variable_2
        num_data = self.num_data
        num_inducing_1 = Z.num_inducing
        num_inducing_2 = O.num_inducing
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)

        common1 = self._common_calculations_1()
        L, LB = common1.L, common1.LB

        common2 = self._common_calculations_2()
        Lu, Au, du = common2.Lu, common2.Au, common2.du
        Lv, Av, dv = common2.Lv, common2.Av, common2.dv
        d = common2.d

        kus = Kuf(Z, self.kernel, Xnew)
        kuv = Kuv(Z, self.kernel, O)
        kvs = Kuf(O, self.kernel, Xnew)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        Lu = tf.linalg.cholesky(kuu)
        Tus = tf.linalg.triangular_solve(Lu, kus, lower=True)
        Tuv = tf.linalg.triangular_solve(Lu, kuv, lower=True)
        cvs = kvs - tf.linalg.matmul(Tuv, Tus, transpose_a=True)
        Tvs = tf.linalg.triangular_solve(Lv, cvs, lower=True)
        Vs = tf.concat([kus, cvs], 0)

        # dTd = tf.reduce_sum(tf.square(d))
        # duTdu = tf.reduce_sum(tf.square(du))
        # dvTdv = tf.reduce_sum(tf.square(dv))

        IN = tf.eye(num_data, dtype=default_float())
        IM1 = tf.eye(num_inducing_1, dtype=default_float())
        IM2 = tf.eye(num_inducing_2, dtype=default_float())

        # tmp1 = tf.matmul(Au, IN - dTd)
        # tmp2 = tf.matmul(tmp1, Au, transpose_b=True)
        # tmp3 = tf.matmul(IM1 - tmp2, Au)
        # tmp4 = tf.matmul(tmp3, IN - dvTdv)
        # tmp5 = tf.matmul(Tus, tmp4, transpose_a=True)
        # tmp6 = tf.matmul(tmp5, y) / sigma

        # tmp7 = tf.matmul(Av, IN - dTd)
        # tmp8 = tf.matmul(tmp7, Av, transpose_b=True)
        # tmp9 = tf.matmul(IM2 - tmp8, Av)
        # tmp10 = tf.matmul(tmp9, IN - duTdu)
        # tmp11 = tf.matmul(Tvs, tmp10, transpose_a=True)
        # tmp12 = tf.matmul(tmp11, y) / sigma 

        # mean = tmp6 + tmp12

        # mean = tf.zeros_like(Xnew)


        kuf = Kuf(Z, self.kernel, X)
        kuv = Kuv(Z, self.kernel, O)
        kvf = Kuf(O, self.kernel, X)
        kuu = Kuu(Z, self.kernel, jitter=default_jitter())
        kvv = Kuu(O, self.kernel, jitter=default_jitter())
        kuu_inv = tf.linalg.inv(kuu)
        Qff = tf.matmul(tf.matmul(kuf, kuu_inv, transpose_a=True), kuf)
        Qvv = tf.matmul(tf.matmul(kuv, kuu_inv, transpose_a=True), kuv)
        cvv = kvv - Qvv
        cvf = kvf - tf.matmul(tf.matmul(kuv, kuu_inv, transpose_a=True), kuf)
        cvv_inv = tf.linalg.inv(cvv)

        Rvv = tf.matmul(tf.matmul(cvf, cvv_inv, transpose_a=True), cvf)
        A_inv = tf.linalg.inv(Rvv + sigma_sq * IN)
        B_inv = tf.linalg.inv(Qff + sigma_sq * IN)
        C = kuu + tf.matmul(tf.matmul(kuf, A_inv), kuf, transpose_b=True)
        C_inv = tf.linalg.inv(C)
        D = cvv + tf.matmul(tf.matmul(cvf, B_inv), cvf, transpose_b=True)
        D_inv = tf.linalg.inv(D)

        mu = tf.matmul(tf.matmul(tf.matmul(C_inv, kuf), A_inv), y)
        mv = tf.matmul(tf.matmul(tf.matmul(D_inv, cvf), B_inv), y)

        mean = tf.matmul(kus, mu, transpose_a=True) + tf.matmul(cvs, mv, transpose_a=True)


        tmp13 = tf.linalg.triangular_solve(L, Vs, lower=True)
        tmp14 = tf.linalg.triangular_solve(LB, tmp13, lower=True)

        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp14, tmp14, transpose_a=True)
                - tf.linalg.matmul(tmp13, tmp13, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [1, 1, 1])
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp14), 0)
                - tf.reduce_sum(tf.square(tmp13), 0)
            )
            var = tf.tile(var[:, None], [1, 1])

        return mean, var

    # Use @property decorator for mu, mv, Su, Sv

    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Compute the mean and variance of q(u) = N(mu, Su). """ 
        X, y = self.data
        num_data = to_default_float(tf.shape(X)[0])
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)
        num_inducing_1 = self.inducing_variable_1.num_inducing

        common = self._common_calculations_2()
        d, Au, Lu, dv, eu = common.d, common.Au, common.Lu, common.dv, common.eu

        IN = tf.eye(num_data, dtype=default_float())
        IM1 = tf.eye(num_inducing_1, dtype=default_float())

        dTd = tf.matmul(d, d, transpose_a=True)
        dvTdv = tf.matmul(dv, dv, transpose_a=True)

        tmp1 = IM1 - tf.matmul(tf.matmul(Au, IN - dTd), Au, transpose_b=True)
        tmp2 = tf.matmul(tf.matmul(Au, IN - dvTdv), y)
        mu = tf.matmul(Lu, tf.matmul(tmp1, tmp2)) / sigma

        Su = tf.matmul(eu, eu, transpose_a=True)

        return mu, Su

    def compute_qv(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """ Compute the mean and variance of q(v) = N(mv, Sv). """ 
        X, y = self.data
        num_data = to_default_float(tf.shape(X)[0])
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)
        num_inducing_2 = self.inducing_variable_2.num_inducing

        common = self._common_calculations_2()
        d, Av, Lv, du, ev = common.d, common.Av, common.Lv, common.du, common.ev

        IN = tf.eye(num_data, dtype=default_float())
        IM2 = tf.eye(num_inducing_2, dtype=default_float())

        dTd = tf.matmul(d, d, transpose_a=True)
        duTdu = tf.matmul(du, du, transpose_a=True)

        tmp1 = IM2 - tf.matmul(tf.matmul(Av, IN - dTd), Av, transpose_b=True)
        tmp2 = tf.matmul(tf.matmul(Av, IN - duTdu), y)
        mv = tf.matmul(Lv, tf.matmul(tmp1, tmp2)) / sigma

        Sv = tf.matmul(ev, ev, transpose_a=True)

        return mv, Sv
