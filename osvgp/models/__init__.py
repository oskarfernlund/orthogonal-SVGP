from gpflow.models import GPModel
from gpflow.models import InternalDataTrainingLossMixin
from .gpr import GPR
from .sgpr import SGPR
from .osgpr import OSGPR
from .experimental import OSGPRBoundGap

__all__ = ["GPModel",
           "InternalDataTrainingLossMixin",
           "GPR",
           "SGPR",
           "OSGPR",
           "OSGPRBoundGap"]
