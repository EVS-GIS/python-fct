# coding: utf-8

"""
Plot Commands

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import click
import xarray as xr

from ..config import config
from .MapFigureSizer import MapFigureSizer
from .PlotCorridor import (
    SetupPlot,
    SetupMeasureAxis
)

# pylint: disable=import-outside-toplevel

def FinalizePlot(fig, ax, title='', filename=None):

    fig_size_inches = 12.5
    aspect_ratio = 4
    cbar_L = "None"
    fig_size_inches, map_axes, cbar_axes = MapFigureSizer(
        fig_size_inches,
        aspect_ratio,
        cbar_loc=cbar_L,
        title=True)

    plt.title(title)
    fig.set_size_inches(fig_size_inches[0], fig_size_inches[1])
    ax.set_position(map_axes)

    if filename is None:
        fig.show()
        plt.show(block=True)
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, dpi=300)
        plt.clf()

@click.group()
def cli():
    """
    Preconfigured plots for visualizing FCT data
    """

@cli.command('swath')
@click.argument('axis', type=int)
@click.argument('swath', type=int)
@click.option(
    '--kind',
    type=click.Choice(['absolute', 'hand', 'havf'], case_sensitive=True),
    default='absolute',
    help="""select plot variant,
    absolute elevation,
    height above nearest drainage
    or height above valley floor""")
@click.option(
    '--clip',
    default=None,
    type=float,
    help='clip data at given height above nearest drainage')
@click.option(
    '--filename', '-f',
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, exists=False),
    help='save output to file')
def plot_elevation_swath(axis, swath, kind, clip, filename):
    """
    Elevation swath profile
    """

    from .PlotElevationSwath import PlotSwath

    config.default()

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    PlotSwath(axis, swath, kind=kind, clip=clip, output=filename)

    if filename is None:
        plt.show(block=True)

@cli.command('valleyprofile')
@click.argument('axis', type=int)
@click.option(
    '--filename', '-f',
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, exists=False),
    help='save output to file')
def plot_valley_elevation_profile(axis, filename):
    """
    Idealized valley elevation profile
    """

    # from ..corridor.ValleyElevationProfile import ValleySwathElevation

    config.default()

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    datafile = config.filename('ax_refaxis_valley_profile', axis=axis)
    data = xr.open_dataset(datafile)

    data = data.sortby('measure', ascending=False)
    x = data['measure']
    y = data['z']

    fig, ax = SetupPlot()

    # _, values = ValleySwathElevation(axis)
    # ax.plot(values[:, 0], values[:, 1], 'darkgray', linewidth=0.8)
    
    ax.plot(x, y)
    ax.set_ylabel('Elevation (m NGF)')
    SetupMeasureAxis(ax, x)
    FinalizePlot(fig, ax, title='Valley Elevation Profile', filename=filename)

@cli.command('talwegheight')
@click.argument('axis', type=int)
@click.option(
    '--filename', '-f',
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, exists=False),
    help='save output to file')
def plot_talweg_height(axis, filename):
    """
    Talweg height relative to valley floor
    """

    config.default()

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    datafile = config.filename('metrics_talweg_height', axis=axis)
    data = xr.open_dataset(datafile)
    data = data.sortby('measure', ascending=False)

    x = data['measure']
    y = data['hmed']
    interpolated = data['interp']
    print(interpolated.dtype)

    fig, ax = SetupPlot()
    ax.plot(x, y, 'darkorange', label='interpolated')
    y[interpolated] = np.nan
    ax.plot(x, y, label='measured')
    ax.set_ylabel('Height relative to valley floor (m)')
    ax.legend()
    SetupMeasureAxis(ax, x)
    FinalizePlot(fig, ax, title='Talweg Relative Height', filename=filename)
