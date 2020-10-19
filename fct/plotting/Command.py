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

from functools import wraps
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

import click
import xarray as xr

from ..cli import (
    fct_entry_point,
    arg_axis
)
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
    help='Save output to file'
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

def fct_plot(group, name=None, title=None):
    """
    FCT plot command wrapper, with standard options:

    - title
    - xlim
    - ylim
    - filename:
        exports to PNG or PDF file,
        and sets matplotlib to use cairo if needed
    """

    def decorate(fun):

        @group.command(name)
        @click.option('--title', type=str, default=title, help='Set figure title')
        @click.option('--xlim', type=(float, float), default=(None, None), help='Set x axis limits')
        @click.option('--ylim', type=(float, float), default=(None, None), help='Set y axis limits')
        @filename_opt
        @wraps(fun)
        def decorated(title, xlim, ylim, filename, *args, **kwargs):

            if filename is None:
                plt.ion()
            elif filename.endswith('.pdf'):
                mpl.use('cairo')

            fig, ax = SetupPlot()

            fun(*args, ax=ax, **kwargs)

            if xlim != (None, None):
                ax.set_xlim(xlim)

            if ylim != (None, None):
                ax.set_ylim(ylim)

            if filename == 'auto':
                filename = None
                _filename = config.filename(fun.__name__, **kwargs)
                click.echo(_filename)

            FinalizePlot(fig, ax, title=title, filename=filename)

        return decorated

    return decorate

@cli.command('hypsometry-global')
@filename_opt
def plot_hypsometry_global(filename):
    """
    Plot elevation distributions (hypsometer)

    @api   fct-plot:hypsometry-global
    @input metrics_hypsometry: metrics_hypsometry_global
    """

    from .PlotHypsometry import PlotHypsometry

    datafile = config.filename('metrics_hypsometry_global')
    hypsometry = xr.open_dataset(datafile)

    # if filename is None:
    #     plt.ion()
    # elif filename.endswith('.pdf'):
    #     mpl.use('cairo')

    fig = PlotHypsometry(hypsometry)

    if filename == 'auto':
        filename = None
        _filename = config.filename('plot_hypsometry_global')
        click.echo(_filename)

    if filename is None:
        fig.show()
        plt.show(block=True)
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, dpi=300)
        plt.clf()

@cli.command('hypsometry')
@arg_axis
@filename_opt
def plot_hypsometry(axis, filename):
    """
    Plot elevation distributions (hypsometer)

    @api   fct-plot:hypsometry
    @input metrics_hypsometry: metrics_hypsometry
    """

    from .PlotHypsometry import PlotHypsometry

    datafile = config.filename('metrics_hypsometry', axis=axis)
    hypsometer = xr.open_dataset(datafile)

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    fig = PlotHypsometry(hypsometer)

    if filename == 'auto':
        filename = None
        _filename = config.filename('plot_hypsometry', axis=axis)
        click.echo(_filename)

    if filename is None:
        fig.show()
        plt.show(block=True)
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, dpi=300)
        plt.clf()

@fct_plot(cli, 'drainage-area', title='Drainage area')
@arg_axis
def plot_drainage_area(ax, axis):
    """
    Plot drainage area profile

    @api   fct-plot:drainage-area
    @input metrics_drainage_area:metrics_drainage_area
    """

    datafile = config.filename('metrics_drainage_area', axis=axis)
    data = xr.open_dataset(datafile).sortby('measure')

    x = data['measure'].values
    y = data['drainage_area']

    ax.plot(x, y)
    SetupMeasureAxis(ax, x)

    # formatter = EngFormatter(unit='km^2')
    # ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel('Drainage area ($km^2$)')


@fct_plot(cli, 'swath-area-height', title='Area-height swath profile')
@click.argument('swath', type=int)
@arg_axis
def plot_swath_area_height(ax, swath, axis):
    """
    Plot cumulative area-height curve for one swath

    @api   fct-plot:swath-area-height
    @input swath_bounds: ax_valley_swaths_bounds
    @input metrics_valleybottom_width: metrics_valleybottom_width
    """

    bounds = xr.open_dataset(config.filename('ax_valley_swaths_bounds', axis=axis))
    data = xr.open_dataset(config.filename('metrics_valleybottom_width', axis=axis))

    measure = bounds['measure'].sel(swath=swath).values
    swath = data.sel(measure=measure)

    heights = swath['height'].values
    areas = swath['valley_bottom_area_h'].values

    diff_area = np.concatenate([
        [areas[0]],
        areas[1:] - areas[:-1]
    ])

    maxh = np.max(heights[diff_area > 0])

    # convert m^2 -> hectares with pixel area = 25 m^2
    ax.plot(areas * 25e-4, heights)
    ax.set_ylim([None, maxh])

    formatter = EngFormatter(unit='ha')
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel('Cumulative area')

    formatter = EngFormatter(unit='m')
    ax.yaxis.set_major_formatter(formatter)
    ax.set_ylabel('Height above nearest drainage')


@cli.command('swath-elevation')
@click.argument('swath', type=int)
@arg_axis
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
def plot_swath_elevation(swath, axis, kind, clip, filename):
    """
    Elevation swath profile

    @api   fct-plot:elevation-swath
    @input swath_elevation_npz: ax_swath_elevation_npz
    """

    from .PlotElevationSwath import PlotSwath

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    if filename is None:

        PlotSwath(axis, swath, kind=kind, clip=clip)
        plt.show(block=True)

    else:

        PlotSwath(axis, swath, kind=kind, clip=clip, output=filename)

    # 
    #     # fig.show()
    #     plt.show(block=True)
    # elif filename.endswith('.pdf'):
    #     plt.savefig(filename, format='pdf', dpi=600)
    #     plt.clf()
    # else:
    #     plt.savefig(filename, dpi=300)
    #     plt.clf()

@cli.command('swath-landcover')
@click.argument('swath', type=int)
@arg_axis
@click.option(
    '--kind',
    type=click.Choice(['std', 'continuity', 'interpreted'], case_sensitive=True),
    default='std',
    help="""select plot variant""")
@filename_opt
def plot_swath_landcover(swath, axis, kind, filename):
    """
    Elevation swath profile

    @api   fct-plot:elevation-swath
    @input swath_elevation_npz: ax_swath_elevation_npz
    """

    from .PlotLandCoverSwath import PlotLandCoverSwath

    if filename is None:
        plt.ion()
    elif filename.endswith('.pdf'):
        mpl.use('cairo')

    if filename is None:

        PlotLandCoverSwath(axis, swath, kind=kind)
        plt.show(block=True)

    else:

        PlotLandCoverSwath(axis, swath, kind=kind, output=filename)

    # if filename is None:
    #     # fig.show()
    #     plt.show(block=True)
    # elif filename.endswith('.pdf'):
    #     plt.savefig(filename, format='pdf', dpi=600)
    #     plt.clf()
    # else:
    #     plt.savefig(filename, dpi=300)
    #     plt.clf()

# def plot_swath_landcover():

@fct_plot(cli, 'profile-elevation', title='Elevation profile')
@arg_axis
@click.option('--floodplain/--no-floodplain', default=False, help='Plot flooplain profile')
@click.option('--talweg/--no-talweg', default=False, help='Plot talweg profile')
@click.pass_context
def plot_profile_elevation(ctx, ax, axis, floodplain, talweg):
    """
    Smoothed elevation profile

    @api   fct-plot:profile-elevation
    @input elevation_profile_floodplain: ax_elevation_profile_floodplain
    @input elevation_profile_talweg: ax_elevation_profile_talweg
    """

    if not (floodplain or talweg):
        click.echo('use at least one of --floodplain or --talweg options')
        ctx.exit(1)

    if floodplain:

        datafile = config.filename('ax_elevation_profile_floodplain', axis=axis)
        data = xr.open_dataset(datafile)

        data = data.sortby('measure', ascending=False)
        x = data['measure']
        y = data['z']

        ax.plot(x, y, label='floodplain')
        ax.set_ylabel('Elevation (m NGF)')
        SetupMeasureAxis(ax, x)

    if talweg:

        talweg_datafile = config.filename('ax_elevation_profile_talweg', axis=axis)
        talweg_data = xr.open_dataset(talweg_datafile)

        talweg_data = talweg_data.sortby('measure', ascending=False)
        x = talweg_data['measure']
        y = talweg_data['z']
        ax.plot(x, y, label='talweg')
        SetupMeasureAxis(ax, x)

        ax.legend()

@fct_plot(cli, 'profile-slope', title='Slope profile')
@arg_axis
@click.option('--floodplain/--no-floodplain', default=False, help='Plot flooplain profile')
@click.option('--talweg/--no-talweg', default=False, help='Plot talweg profile')
@click.option('--smooth/--no-smooth', default=False, help='Plot smoothed profile')
@click.pass_context
def plot_profile_slope(ctx, ax, axis, floodplain, talweg, smooth):
    """
    Idealized slope profile

    @api   fct-plot:profile-slope
    @input elevation_profile_floodplain: ax_elevation_profile_floodplain
    @input elevation_profile_talweg: ax_elevation_profile_talweg
    @input metrics_talweg: metrics_talweg
    """

    if not (floodplain or talweg):
        click.echo('use at least one of --floodplain or --talweg options')
        ctx.exit(1)

    if talweg:

        if smooth:

            datafile = config.filename('ax_elevation_profile_talweg', axis=axis)
            data = xr.open_dataset(datafile)

            data = data.sortby('measure', ascending=False)
            x = data['measure']
            y = data['slope']
            ax.plot(x, y, label='talweg')
            SetupMeasureAxis(ax, x)

        else:

            datafile = config.filename('metrics_talweg', axis=axis)
            data = xr.open_dataset(datafile)

            data = data.sortby('measure', ascending=False)
            x = data['measure']
            y = data['talweg_slope']

            # _, values = ValleySwathElevation(axis)
            # ax.plot(values[:, 0], values[:, 1], 'darkgray', linewidth=0.8)

            ax.plot(x, y, label='talweg')
            SetupMeasureAxis(ax, x)

    if floodplain:

        datafile = config.filename('ax_elevation_profile_floodplain', axis=axis)
        data = xr.open_dataset(datafile)

        data = data.sortby('measure', ascending=False)
        x = data['measure']
        y = data['slope']
        ax.plot(x, y, label='floodplain')
        SetupMeasureAxis(ax, x)

    if floodplain and talweg:
        ax.legend()

    ax.set_ylabel('Slope (%)')


@fct_plot(cli, 'talweg-height', title='Talweg relative height')
@arg_axis
def plot_talweg_height(ax, axis):
    """
    Talweg height relative to valley floor

    @api fct-plot:talweg-height
    @input metrics_talweg: metrics_talweg
    """

    datafile = config.filename('metrics_talweg', axis=axis)
    data = xr.open_dataset(datafile)
    data = data.sortby('measure', ascending=False)

    x = data['measure']
    y = data['talweg_height_median']
    interpolated = data['talweg_height_is_interpolated']

    ax.plot(x, y, 'darkorange', label='interpolated')
    y[interpolated] = np.nan
    ax.plot(x, y, label='measured')
    ax.set_ylabel('Height relative to valley floor (m)')
    ax.legend()

    SetupMeasureAxis(ax, x)

@fct_plot(cli, 'planform', title='Talweg shift relative to reference axis')
@arg_axis
@click.option(
    '--measure', '-m',
    type=click.Choice(['refaxis', 'talweg'], case_sensitive=True),
    default='refaxis')
def plot_planform_shift(ax, axis, measure):
    """
    Planform shift

    @api   fct-plot:planform
    @input metrics_planform: metrics_planform
    """

    data_file = config.filename('metrics_planform', axis=axis)

    if measure == 'refaxis':

        data = xr.open_dataset(data_file).sortby('measure')

        # print(
        #     np.min(data['measure']).values,
        #     np.max(data['measure']).values
        # )

        ax.plot(data['measure'], data['talweg_shift'])
        ax.set_ylabel('Distance to reference axis (m)')
        SetupMeasureAxis(ax, data['measure'])

    else:

        data = xr.open_dataset(data_file).sortby('talweg_measure')

        # print(
        #     np.min(data['talweg_measure']).values,
        #     np.max(data['talweg_measure']).values
        # )

        ax.plot(data['talweg_measure'], data['talweg_shift'])
        ax.set_xlim([
            np.min(data['talweg_measure']),
            np.max(data['talweg_measure'])
        ])
        ax.set_xlabel('Stream distance from source')
        ax.set_ylabel('Distance to reference axis')

        formatter = EngFormatter(unit='m')
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)

@fct_plot(cli, 'amplitude', title='Planform shift amplitude')
@arg_axis
@click.option(
    '--measure', '-m',
    type=click.Choice(['refaxis', 'talweg'], case_sensitive=True),
    default='refaxis')
@click.option('--window', '-win', default=20, help='smoothin window')
def plot_planform_amplitude(ax, axis, measure, window):
    """
    Planform shift

    @api   fct-plot:amplitude
    @input metrics_planform: metrics_planform
    """

    from ..metrics.PlanformShift import PlanformAmplitude

    data_file = config.filename('metrics_planform', axis=axis)

    if measure == 'refaxis':

        data = xr.open_dataset(data_file).sortby('measure')
        amplitude = PlanformAmplitude(data['talweg_shift'], window)

        # print(
        #     np.min(data['measure']).values,
        #     np.max(data['measure']).values
        # )

        ax.plot(amplitude['measure'].values, amplitude.values)
        ax.set_ylabel('Amplitude (m)')
        SetupMeasureAxis(ax, data['measure'])

    else:

        data = xr.open_dataset(data_file).sortby('measure')
        amplitude = PlanformAmplitude(data['talweg_shift'], window)

        # print(
        #     np.min(data['talweg_measure']).values,
        #     np.max(data['talweg_measure']).values
        # )

        ax.plot(amplitude['measure'].values, amplitude.values)
        ax.set_xlim([
            np.min(data['talweg_measure']),
            np.max(data['talweg_measure'])
        ])
        ax.set_xlabel('Stream distance from source')
        ax.set_ylabel('Amplitude')

        formatter = EngFormatter(unit='m')
        ax.yaxis.set_major_formatter(formatter)
        ax.xaxis.set_major_formatter(formatter)

@fct_plot(cli, 'landcover-profile', title='Total landcover width')
@arg_axis
def plot_landcover_profile(ax, axis):
    """
    Landcover class width long profile

    @api   fct-plot:landcover-profile
    @input metrics_landcover_width: metrics_landcover_width
    """

    from .PlotCorridor import PlotLandCoverProfile

    data_file = config.filename('metrics_landcover_width', variant='TOTAL_BDT', axis=axis)
    data = xr.open_dataset(data_file).sortby('measure')

    PlotLandCoverProfile(
        ax,
        data['measure'],
        np.sum(data['buffer_width'], axis=2),
        basis=2,
        window=5
    )

    SetupMeasureAxis(ax, data['measure'])
    ax.set_ylabel('Width (m)')

@fct_plot(cli, 'continuity-profile', title='Continuity buffer width')
@arg_axis
def plot_continuity_profile(ax, axis):
    """
    Continuity buffer width long profile

    @api   fct-plot:continuity-profile
    @input metrics_continuity_width: metrics_continuity_width
    """

    from .PlotCorridor import PlotLandCoverProfile

    data_file = config.filename('metrics_landcover_width', variant='CONT_BDT', axis=axis)
    data = xr.open_dataset(data_file).sortby('measure')

    PlotLandCoverProfile(
        ax,
        data['measure'],
        np.sum(data['buffer_width'], axis=2),
        basis=2,
        window=5
    )

    SetupMeasureAxis(ax, data['measure'])
    ax.set_ylabel('Width (m)')

@fct_plot(cli, 'landcover-profile-lr', title='Left and right bank landcover width')
@arg_axis
@click.option('--max-class', default=8, help='Plot until max_class continuity class')
def plot_left_right_landcover_profile(ax, axis, max_class):
    """
    Left/rigth total landcover width long profile

    @api   fct-plot:landcover-profile-lr
    @input metrics_valleybottom_width: metrics_valleybottom_width
    @input metrics_landcover_width: metrics_landcover_width
    """

    from .PlotCorridor import (
        PlotLeftRightLandcoverProfile,
        PlotLeftRightCorridorLimit
    )

    data_file = config.filename('metrics_landcover_width', variant='TOTAL_BDT', axis=axis)
    width_file = config.filename('metrics_valleybottom_width', axis=axis)

    width = xr.open_dataset(width_file)
    data = xr.open_dataset(data_file)

    merged = data.merge(width).sortby('measure')

    data_vb_width = merged['valley_bottom_width']
    data_vb_area_lr = merged['valley_bottom_area_lr']
    vbw_left = data_vb_width * data_vb_area_lr.sel(side='left') / np.sum(data_vb_area_lr, axis=1)
    vbw_right = data_vb_width * data_vb_area_lr.sel(side='right') / np.sum(data_vb_area_lr, axis=1)

    PlotLeftRightLandcoverProfile(
        ax,
        merged,
        merged['measure'],
        merged['buffer_width'].sel(side='left'),
        merged['buffer_width'].sel(side='right'),
        basis=0,
        max_class=max_class,
        clip=False,
        window=5)

    PlotLeftRightCorridorLimit(
        ax,
        merged,
        merged['measure'],
        vbw_left,
        vbw_right,
        basis=0,
        window=5)

    SetupMeasureAxis(ax, merged['measure'])
    ax.set_ylabel('Width (m)')
    ax.legend(ncol=2, loc='lower left')

@fct_plot(cli, 'continuity-profile-lr', title='Left and right bank continuity buffer width')
@arg_axis
@click.option('--max-class', default=6, help='Plot until max_class continuity class')
def plot_left_right_continuity_profile(ax, axis, max_class):
    """
    Left/rigth continuity buffer width long profile

    @api   fct-plot:continuity-profile-lr
    @input metrics_valleybottom_width: metrics_valleybottom_width
    @input metrics_continuity_width: metrics_continuity_width
    """

    from .PlotCorridor import (
        PlotLeftRightLandcoverProfile2,
        PlotLeftRightCorridorLimit
    )

    data_file = config.filename('metrics_width_continuity', variant='REMAPPED', axis=axis)
    width_file = config.filename('metrics_valleybottom_width', axis=axis)

    width = xr.open_dataset(width_file)
    data = xr.open_dataset(data_file)

    merged = data.merge(width).sortby('measure')

    data_vb_width = merged['valley_bottom_width']
    data_vb_area_lr = merged['valley_bottom_area_lr']
    vbw_left = data_vb_width * data_vb_area_lr.sel(side='left') / np.sum(data_vb_area_lr, axis=1)
    vbw_right = data_vb_width * data_vb_area_lr.sel(side='right') / np.sum(data_vb_area_lr, axis=1)

    PlotLeftRightLandcoverProfile2(
        ax,
        merged,
        merged['measure'],
        merged['buffer_width'].sel(side='left'),
        merged['buffer_width'].sel(side='right'),
        max_class=0,
        clip=True,
        window=5)

    PlotLeftRightCorridorLimit(
        ax,
        merged,
        merged['measure'],
        vbw_left,
        vbw_right,
        basis=0,
        window=5)

    SetupMeasureAxis(ax, merged['measure'])
    ax.set_ylabel('Width (m)')
    ax.legend(ncol=2, loc='lower left')
