#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Some type aliases (from GPflow + some custom ones).
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Tuple, Union

import tensorflow as tf
from gpflow.base import InputData, OutputData
from gpflow.base import RegressionData, MeanAndVariance
from gpflow.models.util import InducingPointsLike 


# =============================================================================
#  TYPE ALIASES
# =============================================================================

OrthogonalInducingPointsLike = Tuple[InducingPointsLike, InducingPointsLike]
AnyInducingPointsLike = Union[InducingPointsLike, OrthogonalInducingPointsLike]
ProbabilisticPredictions = Tuple[InputData, tf.Tensor, tf.Tensor]
SplitRegressionData = Tuple[RegressionData, RegressionData]
