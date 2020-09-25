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

from ..cli import fct_entry_point
from ..config import config
from .MapFigureSizer import MapFigureSizer
from .PlotCorridor import (
    SetupPlot,
    SetupMeasureAxis
)

# pylint: disable=import-outside-toplevel,invalid-name

filename_opt = click.option(
    '--filename', '-f',
    default=None,
    type=click.Path(file_okay=True, dir_okay=False, writable=True, exists=False),
    help='save output to file'
)

def FinalizePlot(fig, ax, title='', filename=None):

    fig_size_inches = 12.5
    aspect_ratio = 4
    cbar_L = "None"
    fig_size_inches, map_axes, _ = MapFigureSizer(
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

@fct_entry_point
def cli(env):
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
@filename_opt
def plot_elevation_swath(axis, swath, kind, clip, filename):
    """
    Elevation swath profile
    """

    from .PlotElevationSwath import PlotSwath

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    PlotSwath(axis, swath, kind=kind, clip=clip, output=filename)

    if filename is None:
        plt.show(block=True)

@cli.command('valleyprofile')
@click.argument('axis', type=int)
@filename_opt
def plot_valley_elevation_profile(axis, filename):
    """
    Idealized valley elevation profile
    """

    # from ..corridor.ValleyElevationProfile import ValleySwathElevation

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
@filename_opt
def plot_talweg_height(axis, filename):
    """
    Talweg height relative to valley floor
    """

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

@cli.command('landcover-profile')
@click.argument('axis', type=int)
@filename_opt
def plot_landcover_profile(axis, filename):
    """
    Landcover class width long profile
    """

    from .PlotCorridor import PlotLandCoverProfile

    data_file = config.filename('metrics_lcw_variant', variant='TOTAL_BDT', axis=axis)
    data = xr.open_dataset(data_file).sortby('measure')

    fig, ax = SetupPlot()
    PlotLandCoverProfile(
        ax,
        data['measure'],
        data['buffer_width'].sel(type='total'),
        basis=2,
        window=5
    )

    SetupMeasureAxis(ax, data['measure'])
    ax.set_ylabel('Width (m)')
    FinalizePlot(
        fig,
        ax,
        title='Total Landcover Width',
        filename=filename)

@cli.command('landcover-profile-lr')
@click.argument('axis', type=int)
@click.option('--max-class', default=8, help='Plot until max_class continuity class')
@filename_opt
def plot_left_right_landcover_profile(axis, max_class, filename):
    """
    Left/rigth continuous landcover buffer width long profile
    """

    from .PlotCorridor import (
        PlotLeftRightLandcoverProfile,
        PlotLeftRightCorridorLimit
    )

    data_file = config.filename('metrics_lcw_variant', variant='TOTAL_BDT', axis=axis)
    width_file = config.filename('metrics_valleybottom_width', axis=axis)

    width = xr.open_dataset(width_file)
    data = xr.open_dataset(data_file)

    merged = data.merge(width).sortby('measure')
    
    data_vb_width = merged['valley_bottom_width']
    data_vb_area_lr = merged['valley_bottom_area_lr']
    vbw_left = data_vb_width * data_vb_area_lr.sel(side='left') / np.sum(data_vb_area_lr, axis=1)
    vbw_right = data_vb_width * data_vb_area_lr.sel(side='right') / np.sum(data_vb_area_lr, axis=1)

    fig, ax = SetupPlot()

    PlotLeftRightCorridorLimit(
        ax,
        merged,
        merged['measure'],
        vbw_left,
        vbw_right,
        window=5)

    PlotLeftRightLandcoverProfile(
        ax,
        merged,
        merged['measure'],
        merged['buffer_width'].sel(type='left'),
        merged['buffer_width'].sel(type='right'),
        max_class=max_class,
        clip=False,
        window=5)

    SetupMeasureAxis(ax, merged['measure'])
    ax.set_ylabel('Width (m)')
    ax.legend(ncol=2)
    FinalizePlot(
        fig,
        ax,
        title='Left and Right Bank Landcover Width',
        filename=filename)

@cli.command('continuity-profile-lr')
@click.argument('axis', type=int)
@click.option('--max-class', default=6, help='Plot until max_class continuity class')
@filename_opt
def plot_left_right_continuity_profile(axis, max_class, filename):
    """
    Left/rigth continuous landcover buffer width long profile
    """

    from .PlotCorridor import (
        PlotLeftRightLandcoverProfile,
        PlotLeftRightCorridorLimit
    )

    data_file = config.filename('metrics_lcw_variant', variant='CONT_BDT', axis=axis)
    width_file = config.filename('metrics_valleybottom_width', axis=axis)

    width = xr.open_dataset(width_file)
    data = xr.open_dataset(data_file)

    merged = data.merge(width).sortby('measure')

    data_vb_width = merged['valley_bottom_width']
    data_vb_area_lr = merged['valley_bottom_area_lr']
    vbw_left = data_vb_width * data_vb_area_lr.sel(side='left') / np.sum(data_vb_area_lr, axis=1)
    vbw_right = data_vb_width * data_vb_area_lr.sel(side='right') / np.sum(data_vb_area_lr, axis=1)

    fig, ax = SetupPlot()

    PlotLeftRightCorridorLimit(
        ax,
        merged,
        merged['measure'],
        vbw_left,
        vbw_right,
        window=5)

    PlotLeftRightLandcoverProfile(
        ax,
        merged,
        merged['measure'],
        merged['buffer_width'].sel(type='left'),
        merged['buffer_width'].sel(type='right'),
        max_class=max_class,
        clip=True,
        window=5)

    SetupMeasureAxis(ax, merged['measure'])
    ax.set_ylabel('Width (m)')
    ax.legend(ncol=2)
    FinalizePlot(
        fig,
        ax,
        title='Left and Right Bank Continuity Buffer Width',
        filename=filename)
