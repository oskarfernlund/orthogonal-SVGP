#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plotting functions.
"""


# =============================================================================
#  IMPORTS AND DEPENDENCIES
# =============================================================================

from typing import Optional, Union, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from mpl_toolkits.mplot3d import axes3d

from osvgp.base import RegressionData
from osvgp.base import ProbabilisticPredictions, AnyInducingPointsLike
from osvgp.util import diagonal, flatten, is_square


# =============================================================================
#  GLOBAL VARIABLES AND CONFIGURATION
# =============================================================================

plt.style.use("seaborn-deep")


# =============================================================================
#  FUNCTIONS
# =============================================================================

def log_tick_formatter(val, pos=None):
    """ Log tick formatter for 3D plots. """
    return r"$10^{%i}$" % val


def plot_dataset(data: RegressionData,
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

    # Generate plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    ax.scatter(X, y, facecolor="k", edgecolor="w", s=15, label="data")

    # Optional labels
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()


def plot_predictions(data: RegressionData, 
                     preds: ProbabilisticPredictions,
                     iv: AnyInducingPointsLike,
                     c: str = "C0", 
                     title: Optional[str] = None,
                     legend: bool = False,
                     figsize: tuple = (8, 5)) -> None:
    """ Generate a plot of SGPR or OSGPR predictions.

    Plots the predicted mean +/- 1 and 2 standard deviations, the training data
    and the learned inducing input locations.
    
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
    y_min, y_max = min([min(mu - 2*sd), min(y)]), max([max(mu - 2*sd), max(y)])
    Z_lat = np.full_like(Z, y_min - 0.1 * (y_max - y_min))
    O_lat = np.full_like(O, y_min - 0.1 * (y_max - y_min))

    # Generate plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    ax.scatter(X, y, facecolor="k", edgecolor="w", s=15, label="data")
    ax.plot(Xs, mu, color=c, lw=2, label="mean")
    ax.fill_between(Xs, mu + 1*sd, mu - 1*sd, facecolor=c, 
                    edgecolor="none", alpha=0.25, label="1 SD")
    ax.fill_between(Xs, mu + 2*sd, mu - 2*sd, color=c, alpha=0.25, label="2 SD")
    ax.scatter(Z, Z_lat, s=30, marker="+", 
               facecolor="k", edgecolor="w", label="Z")
    if O is not None:
        ax.scatter(O, O_lat, s=30, marker="^", 
                   facecolor="k", edgecolor="w", label="O")
    
    # Optional labels
    if title:
        ax.set_title(title)
    if legend:
        ax.legend()


def overlay_predictions(data: RegressionData, 
                        preds1: ProbabilisticPredictions,
                        preds2: ProbabilisticPredictions,
                        iv1: AnyInducingPointsLike,
                        iv2: AnyInducingPointsLike,
                        c1: str = "grey",
                        c2: str = "C2",  
                        title: Optional[str] = None,
                        legend: bool = False,
                        figsize: tuple = (8, 5)) -> None:
    """ Generate an overlay of SGPR or OSGPR predictions.

    Plots the predicted mean +/- 2 standard deviations, the training data and 
    the learned inducing input locations.
    
    Args:
        data (RegressionData) : training data to plot with predictions
        preds1 (ProbabilisticPredictions) : 1st set of predictions (x*, mu, cov)
        preds1 (ProbabilisticPredictions) : 2nd set of predictions (x*, mu, cov)
        iv1 (AnyInducingPointsLike) : 1st set of inducing variables (inputs)
        iv2 (AnyInducingPointsLike) : 2nd set of inducing variables (inputs)
        c1 (str) : prediction colour 1
        c2 (str) : prediction colour 2
        title (str or None) : figure title
        legend (bool) : whether or not to include a legend
        figsize (tuple of int) : figure size
    """
    # Unpack inputs
    X, y = data
    Xs1, mu1, cov1 = preds1
    Xs2, mu2, cov2 = preds2
    sd1 = np.sqrt(diagonal(cov1)) if is_square(cov1) else np.sqrt(cov1)
    sd2 = np.sqrt(diagonal(cov2)) if is_square(cov2) else np.sqrt(cov2)
    Z1, O1 = iv1 if type(iv1) == tuple else (iv1, None)
    Z2, O2 = iv2 if type(iv2) == tuple else (iv2, None)

    # Flatten arrays/tensors
    X, y = flatten(X), flatten(y)
    Xs1, mu1, sd1 = flatten(Xs1), flatten(mu1), flatten(sd1)
    Xs2, mu2, sd2 = flatten(Xs2), flatten(mu2), flatten(sd2)
    Z1, O1 = flatten(Z1), flatten(O1)
    Z2, O2 = flatten(Z2), flatten(O2)

    # Compute inducing point plotting lattitude
    y_min = min([min(mu1 - 2*sd1), min(mu2 - 2*sd2), min(y)]) 
    y_max = max([max(mu1 + 2*sd1), max(mu2 + 2*sd2), max(y)])
    Z1_lat = np.full_like(Z1, y_max + 0.1 * (y_max - y_min))
    O1_lat = np.full_like(O1, y_max + 0.1 * (y_max - y_min))
    Z2_lat = np.full_like(Z2, y_min - 0.1 * (y_max - y_min))
    O2_lat = np.full_like(O2, y_min - 0.1 * (y_max - y_min))

    # Generate plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    ax.scatter(X, y, facecolor="k", edgecolor="w", s=15, label="data")
    ax.plot(Xs1, mu1, color=c1, lw=2)
    ax.fill_between(Xs1, mu1 + 2*sd1, mu1 - 2*sd1, color=c1, alpha=0.25)
    ax.plot(Xs2, mu2, color=c2, lw=2, ls="--")
    ax.plot(Xs2, mu2 + 2*sd2, color=c2, lw=1)
    ax.plot(Xs2, mu2 - 2*sd2, color=c2, lw=1)
    ax.scatter(Z1, Z1_lat, s=30, marker="+", facecolor="k", edgecolor="w")
    if O1 is not None:
        ax.scatter(O1, O1_lat, s=30, marker="^", facecolor="k", edgecolor="w")
    ax.scatter(Z2, Z2_lat, s=30, marker="+", facecolor="k", edgecolor="w")
    if O2 is not None:
        ax.scatter(O2, O2_lat, s=30, marker="^", facecolor="k", edgecolor="w")
    
    # Optional labels
    if title:
        ax.set_title(title)
    

def plot_metrics_2d(xdata: Union[range, list, np.ndarray],
                    metrics: dict,
                    colours: Optional[List[str]] = None,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    yscale: str = "linear",
                    legend: bool = False,
                    figsize: tuple = (8, 5)) -> None:
    """ Generate a plot of evaluation metrics vs. some control variable.

    Args:
        xdata (range, list or np.ndarray) : control variable to plot against
        metrics (dict) : evaluation metrics keyed by model
        colours (list of str or None) : curve colours
        title (str or None) : figure title
        xlabel (str or None) : x-axis label
        ylabel (str or None) : y-axis label
        yscale (str) : y-axis scale -- "linear" or "log"
        legend (bool) : whether or not to include a legend
        figsize (tuple of int) : figure size
    """
    # Generate list of default colours if none provided
    if not colours:
        colours = ["C" + str(i) for i in range(len(metrics.keys()))]

    # Generate plot
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=150)
    for i, (model, vals) in enumerate(metrics.items()):
        ax.plot(xdata, vals, c=colours[i], lw=2, label=model)
    ax.set_yscale(yscale)

    # Optional labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend:
        ax.legend()


def plot_metrics_3d(xdata: np.ndarray,
                    ydata: np.ndarray,
                    metrics: np.ndarray,
                    cmap: str = "viridis",
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    zlabel: Optional[str] = None,
                    zscale: str = "linear",
                    angle: Tuple[int, int] = (20, 30),
                    legend: bool = False,
                    figsize: tuple = (8, 5)) -> None:
    """ Generate a surface plot of evaluation metrics vs. 2 control variables.
    
    Args:
        xdata (np.ndarray) : control variable to plot on x axis
        ydata (np.ndarray) : control variable to plot on y axis
        metrics (np.ndarray) : evaluation metrics to plot on z axis
        cmap (str) : surface colourmap
        title (str or None) : figure title
        xlabel (str or None) : x-axis label
        ylabel (str or None) : y-axis label
        zlabel (str or None) : z-axis label
        zscale (str) : z-axis scale -- "linear" or "log"
        angle (tuple of int) : plot viewing angle
        legend (bool) : whether or not to include a legend
        figsize (tuple of int) : figure size
    """
    # Generate plot
    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    # Plot metrics in linear or log scale
    if zscale == "linear":
        ax.plot_surface(xdata, ydata, metrics, alpha=0.9, cmap=cmap)
    elif zscale == "log":
        ax.plot_surface(xdata, ydata, np.log10(metrics), alpha=0.9, cmap=cmap)
        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    
    # Integer axis tick formats
    # locator = mticker.MultipleLocator(2)
    # formatter = mticker.StrMethodFormatter("{x:.0f}")
    # plt.gca().xaxis.set_major_locator(locator)
    # plt.gca().xaxis.set_major_formatter(formatter)
    # plt.gca().yaxis.set_major_locator(locator)
    # plt.gca().yaxis.set_major_formatter(formatter)
    
    ax.view_init(*angle)

    # Optional labels
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if zlabel:
        ax.set_zlabel(zlabel)
    if legend:
        ax.legend()
