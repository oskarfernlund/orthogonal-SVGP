from scipy.optimize import OptimizeResult
from gpflow.optimizers.scipy import LossClosure
from gpflow.optimizers import Scipy
from .lbfgs import LBFGS

__all__ = ["OptimizeResult", 
           "LossClosure",
           "Scipy",
           "LBFGS"]