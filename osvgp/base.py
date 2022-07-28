#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom type aliases.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Tuple, Union

import numpy as np
import tensorflow as tf
from gpflow.base import InputData
from gpflow.models.util import InducingPointsLike 


# =============================================================================
#  TYPE ALIASES
# =============================================================================

ArrayLike = Union[np.ndarray, tf.Tensor]
OrthogonalInducingPointsLike = Tuple[InducingPointsLike, InducingPointsLike]
AnyInducingPointsLike = Union[InducingPointsLike, OrthogonalInducingPointsLike]
ProbabilisticPredictions = Tuple[InputData, tf.Tensor, tf.Tensor]
