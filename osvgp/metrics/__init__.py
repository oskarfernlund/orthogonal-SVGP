from .densities import compute_lpd
from .divergences import compute_kl_from_exact_posterior

__all__ = ["compute_lpd",
           "compute_kl_from_exact_posterior"]
           