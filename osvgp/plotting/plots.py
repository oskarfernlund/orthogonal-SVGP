#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from gpflow.base import RegressionData

from osvgp.base import ProbabilisticPredictions, AnyInducingPointsLike
from osvgp.util import diagonal, flatten, is_square


# =============================================================================
#  GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

plt.style.use("seaborn")


# =============================================================================
#  FUNCTIONS
# =============================================================================

def plot_dataset_1D(data: RegressionData,
                    title: Optional[str] = None,
                    legend: bool = False,
                    figsize: tuple = (8, 5)) -> None:
    """ Generate a plot of a 1D regression dataset. 
    
    Args:
        data (RegressionData) : training data to plot with predictions
        title (str or None) : figure title
        legend (bool) : whether or not to include a legend
        figsize (tuple of int) : figure size
    """
    # Unpack & flatten inputs
    X, y = data
    X, y = flatten(X), flatten(y)

    # Generate figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.scatter(X, y, facecolor="k", edgecolor="w", s=15, label="data")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()


def plot_predictions_1D(data: RegressionData, 
                        preds: ProbabilisticPredictions,
                        iv: AnyInducingPointsLike,
                        c: str = "C0", 
                        title: Optional[str] = None,
                        legend: bool = False,
                        figsize: tuple = (8, 5)) -> None:
    """ Generate a plot of SGPR or OSGPR predictions.

    Plots the predicted mean +/- 3 standard deviations, the training data and 
    the learned inducing input locations.
    
    Args:
        data (RegressionData) : training data to plot with predictions
        preds (ProbabilisticPredictions) : predictions (x*, mu, cov)
        iv (AnyInducingPointsLike) : inducing variables (inputs)
        c (str) : prediction colour
        title (str or None) : figure title
        legend (bool) : whether or not to include a legend
        figsize (tuple of int) : figure size
    """
    # Unpack inputs
    X, y = data
    Xs, mu, cov = preds
    sd = np.sqrt(diagonal(cov)) if is_square(cov) else np.sqrt(cov)
    Z, O = iv if type(iv) == tuple else (iv, None)

    # Flatten arrays/tensors
    X, y = flatten(X), flatten(y)
    Xs, mu, sd = flatten(Xs), flatten(mu), flatten(sd)
    Z, O = flatten(Z), flatten(O)

    # Compute inducing point plotting lattitude
    y_min, y_max = min([min(mu - 3*sd), min(y)]), max([max(mu - 3*sd), max(y)])
    Z_lat = np.full_like(Z, y_min - 0.1 * (y_max - y_min))
    O_lat = np.full_like(O, y_min - 0.1 * (y_max - y_min))

    # Generate figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    ax.scatter(X, y, facecolor="k", edgecolor="w", s=15, label="data")
    ax.plot(Xs, mu, color=c, lw=2, label="mean")
    ax.fill_between(Xs, mu + 3*sd, mu - 3*sd, color=c, alpha=0.25, label="SD")
    ax.scatter(Z, Z_lat, s=30, marker="+", 
               facecolor="k", edgecolor="w", label="Z")
    if O is not None:
        ax.scatter(O, O_lat, s=30, marker="^", 
                   facecolor="k", edgecolor="w", label="O")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()
    