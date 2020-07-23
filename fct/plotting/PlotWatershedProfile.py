# coding: utf-8

"""
Watershed Metrics Plot

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
from matplotlib.ticker import EngFormatter
from scipy.interpolate import interp1d
import click

default_aes = {
    'pop': ('#faafb4', u'Population × $10^3 / km$'), # Population
    'income': ('#faafb4', u'Household income (euros)'), # Income
    'water': ('#a5bfdd', u'Water $km^2 / km$'), # Water
    'gravels': ('#cccccc', u'Gravels $km^2 / km$'), # Gravels
    'natural': ('#bee62e', u'Natural Open $km^2 / km$'), # Natural
    'forest': ('#6f9e00', u'Forest $km^2 / km$'), # Forest
    'grassland': ('#ffe45a', u'Grassland $km^2 / km$'), # Grassland
    'crops': ('#ffff99', u'Crops $km^2 / km$'), # Crops
    'diffuse': ('#fa7c85', u'Diffuse urban area $km^2 / km$'), # Diffuse Urban
    'urban': ('#fa1524', u'Dense urban area $km^2 / km$'), # Urban
    'infrastructures': ('#fa1665', u'Infrastructures $km^2 / km$')  # Disconnected
}

names = [
    'water',
    'gravels',
    'natural',
    'forest',
    'grassland',
    'crops',
    'diffuse',
    'urban',
    'infrastructures'
]

def PlotWatershedProfile(ax, x, y, variable, dx=2000.0):

    xmin = np.min(x)
    xmax = np.max(x)
    xp = np.arange(xmin, xmax, dx)
    ycum = np.cumsum(y)

    fun = interp1d(x, ycum, kind='slinear', bounds_error=False)
    yp = -(fun(xp+0.5*dx) - fun(xp-0.5*dx)) / dx
    # yp[0] = 0

    yp[np.isnan(yp)] = 0.0
    yp = yp * 1000 # m (dx) -> km

    if variable in default_aes:
        color, label = default_aes[variable]
    else:
        click.secho('Unknown variable %s' % variable)
        color = '#faafb4'
        label = '%s / km' % variable

    ax.bar(xp, height=yp, width=dx, align='edge', color=color, edgecolor='k')
    ax.set_ylabel(label)

def PlotStackedProfile(ax, x, *ys, variables=None, dx=2000.0):
    
    xmin = np.min(x)
    xmax = np.max(x)
    xp = np.arange(xmin, xmax, dx)

    yp = np.zeros((xp.shape[0], len(ys)))
    bottom = np.zeros(xp.shape[0])

    for k, y in enumerate(ys):

        ycum = np.cumsum(y)
        fun = interp1d(x, ycum, kind='slinear', bounds_error=False)
        yp[:, k] = -(fun(xp+0.5*dx) - fun(xp-0.5*dx)) / dx
        # yp[0, k] = -fun(dx) / dx

    yp[np.isnan(yp)] = 0.0
    yp = yp * 1000 # m (dx) -> km

    for k, y in enumerate(ys):

        if variables:

            variable = variables[k]

            if variable in default_aes:
                color, ylabel = default_aes[variable]
            else:
                click.secho('Unknown variable %s' % variable)
                color = '#faafb4'
                ylabel = None

        else:

            color = default_aes[k][0]
            ylabel = None

        scaled = yp[:, k]
        ax.bar(
            xp,
            height=scaled,
            bottom=bottom,
            width=dx,
            align='center',
            color=color,
            edgecolor='k',
            label=ylabel)

        bottom = bottom + scaled

    label = 'density / km'
    ax.set_ylabel(label)
    ax.legend()

def PlotPopulation(axis, width):

    #pylint: disable=import-outside-toplevel
    import xarray as xr
    from .PlotCorridor import SetupPlot, SetupMeasureAxis, FinalizePlot
    from ..config import config

    config.default()
    metric_file = config.filename('metrics_watershed', axis=axis, subset='BUF1K')
    data = xr.open_dataset(metric_file)

    data = data.sortby(data['measure'], ascending=False)
    pop = data['pop'].sortby('width')
    variable = np.cumsum(pop, axis=1).sel(width=width)

    fig, ax = SetupPlot()
    PlotWatershedProfile(ax, variable['measure'], variable, 'pop')
    SetupMeasureAxis(ax, variable['measure'])
    FinalizePlot(fig, ax, 'Population')

def PlotIncome(axis, width):

    #pylint: disable=import-outside-toplevel
    import xarray as xr
    from .PlotCorridor import SetupPlot, SetupMeasureAxis, FinalizePlot
    from ..config import config

    config.default()
    metric_file = config.filename('metrics_watershed', axis=axis, subset='BUF1K')
    data = xr.open_dataset(metric_file)

    data = data.sortby(data['measure'], ascending=False)
    pop = data['income'].sortby('width')
    variable = np.cumsum(pop, axis=1).sel(width=width)

    fig, ax = SetupPlot()
    PlotWatershedProfile(ax, variable['measure'], variable, 'income')
    SetupMeasureAxis(ax, variable['measure'])
    FinalizePlot(fig, ax, 'Population Revenue')

def PlotLandCover(axis, landcover, width):

    #pylint: disable=import-outside-toplevel
    import xarray as xr
    from .PlotCorridor import SetupPlot, SetupMeasureAxis, FinalizePlot
    from ..config import config

    config.default()
    metric_file = config.filename('metrics_watershed', axis=axis, subset='BUF1K')
    data = xr.open_dataset(metric_file)

    klass = data['landcover'][landcover]
    data = data.sortby(data['measure'], ascending=False)
    lck = data['lck'].sel(landcover=klass).sortby('width')
    variable = np.cumsum(lck, axis=1).sel(width=width)

    fig, ax = SetupPlot()
    PlotWatershedProfile(ax, variable['measure'], variable, names[landcover])
    SetupMeasureAxis(ax, variable['measure'])
    FinalizePlot(fig, ax, klass)

def PlotAgriculture(axis, width):

    #pylint: disable=import-outside-toplevel
    import xarray as xr
    from .PlotCorridor import SetupPlot, SetupMeasureAxis, FinalizePlot
    from ..config import config

    config.default()
    metric_file = config.filename('metrics_watershed', axis=axis, subset='BUF1K')
    data = xr.open_dataset(metric_file)

    data = data.sortby(data['measure'], ascending=False)
    ys = list()

    for k in (5, 4):

        klass = data['landcover'][k]
        lck = data['lck'].sel(landcover=klass).sortby('width')
        y = np.cumsum(lck, axis=1).sel(width=width)
        ys.append(y)

    fig, ax = SetupPlot()
    PlotStackedProfile(ax, y['measure'], *ys, variables=('crops', 'grassland'))
    SetupMeasureAxis(ax, y['measure'])
    FinalizePlot(fig, ax, 'Agriculture, buffer = %.0f m' % width)

def PlotBuiltEnvironment(axis, width):

    #pylint: disable=import-outside-toplevel
    import xarray as xr
    from .PlotCorridor import SetupPlot, SetupMeasureAxis, FinalizePlot
    from ..config import config

    config.default()
    metric_file = config.filename('metrics_watershed', axis=axis, subset='BUF1K')
    data = xr.open_dataset(metric_file)

    data = data.sortby(data['measure'], ascending=False)
    ys = list()

    for k in (8, 7, 6):

        klass = data['landcover'][k]
        lck = data['lck'].sel(landcover=klass).sortby('width')
        y = np.cumsum(lck, axis=1).sel(width=width)
        ys.append(y)

    fig, ax = SetupPlot()
    PlotStackedProfile(ax, y['measure'], *ys, variables=('infrastructures', 'urban', 'diffuse'))
    SetupMeasureAxis(ax, y['measure'])
    FinalizePlot(fig, ax, 'Built Environment, buffer = %.0f m' % width)