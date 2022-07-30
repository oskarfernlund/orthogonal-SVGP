from gpflow.models import GPModel
from gpflow.models import InternalDataTrainingLossMixin
from .gpr import GPR
from .sgpr import SGPR
from .osgpr import OSGPR

__all__ = ["GPModel",
           "InternalDataTrainingLossMixin",
           "GPR",
           "SGPR",
           "OSGPR"]
