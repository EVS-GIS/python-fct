# coding: utf-8

"""
Elevation Swath Vizualization
"""

# import os
from typing import Tuple
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import xarray as xr

from .MapFigureSizer import MapFigureSizer

def setup_plot() -> Tuple[Figure, Axes]:
    """
    setup profile plot
    """

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    ax.grid(which='both', axis='both', alpha=0.5)

    return fig, ax

def finalize_plot(
        fig: Figure,
        ax: Axes,
        xlabel: str = None,
        ylabel: str = None,
        title: str = None,
        filename: str = None):
    """
    set profile plot details and show plot or save to file.
    """

    if xlabel:
        ax.set_xlabel(xlabel)

    if ylabel:
        # ax.set_ylabel("Relative Elevation (m)")
        ax.set_ylabel(ylabel)

    # fig_size_inches = 6.25
    # aspect_ratio = 3
    fig_size_inches = 12.5
    aspect_ratio = 4
    cbar_L = "None"

    [fig_size_inches, map_axes, cbar_axes] = MapFigureSizer(
        fig_size_inches,
        aspect_ratio,
        cbar_loc=cbar_L,
        title=title is not None)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def plot_profile_mean(ax: Axes, dataset: xr.Dataset) -> Axes:
    """
    Plot mean elevation profile
    """

    x = dataset.distance.values
    density = dataset.density.values
    profile = dataset['mean'].values
    count = np.sum(density, axis=1)

    mean_profile = np.zeros_like(count, dtype='float32')
    mean_profile[count > 0] = (
        np.sum(profile[count > 0] * density[count > 0], axis=1) /
        count[count > 0]
    )

    parts = np.split(
        np.column_stack([x, mean_profile]),
        np.where(count == 0)[0])

    for k, part in enumerate(parts):

        if k == 0:

            xk = part[:, 0]
            profilek = part[:, 1:]

        else:

            xk = part[1:, 0]
            profilek = part[1:, 1:]

        if xk.size > 0:

            ax.plot(
                xk, profilek,
                "#48638a",
                linewidth=1.5)

    ax.set_xlabel("Distance from reference axis (m)")

    return ax

def plot_profile_quantiles(ax: Axes, dataset: xr.Dataset) -> Axes:
    """
    Plot elevation quantiles profile
    """

    x = dataset.distance.values
    profile = dataset.profile.values
    density = dataset.density.values
    count = np.sum(density, axis=1)

    parts = np.split(
        np.column_stack([x, profile]),
        np.where(count == 0)[0])

    for k, part in enumerate(parts):

        if k == 0:

            xk = part[:, 0]
            profilek = part[:, 1:]

        else:

            xk = part[1:, 0]
            profilek = part[1:, 1:]

        if xk.size > 0:

            ax.fill_between(
                xk, profilek[:, 0], profilek[:, 4],
                facecolor='#b9d8e6',
                alpha=0.2,
                interpolate=True)

            ax.plot(
                xk, profilek[:, 0], "gray", xk, profilek[:, 4],
                "gray",
                linewidth=0.5,
                linestyle='--')

            ax.fill_between(
                xk, profilek[:, 1], profilek[:, 3],
                facecolor='#48638a',
                alpha=0.5,
                interpolate=True)

            ax.plot(
                xk, profilek[:, 2],
                "#48638a",
                linewidth=1.2)

    ax.set_xlabel("Distance from reference axis (m)")

    return ax
