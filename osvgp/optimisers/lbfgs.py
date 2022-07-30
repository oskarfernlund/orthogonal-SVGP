#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
L-BFGS optimiser.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Any, Sequence

import tensorflow as tf

from osvgp.optimisers import LossClosure, OptimizeResult, Scipy


# =============================================================================
#  CLASSES
# =============================================================================

class LBFGS(Scipy):
    """ Limited memory BFGS optimiser (inherited from GPflow). """

    def minimise(self,
                 closure: LossClosure,
                 variables: Sequence[tf.Variable], 
                 **scipy_kwargs: Any) -> OptimizeResult:
        """ Minimise objective using L-BFGS algorithm.
        
        Args:
            closure (LossClosure) : loss function to minimise
            variables (Sequence of tf.Variable) : variables to optimise over
            scipy_kwargs (Any) : keyword arguments for scipy.optimize.minimize
        
        Returns:
            (OptimizeResult) : result of the optimisation
        """
        return super().minimize(closure, variables, **scipy_kwargs)
