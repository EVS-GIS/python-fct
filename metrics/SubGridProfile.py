#!/usr/bin/env python
# coding: utf-8

"""
Raster buffer around stream

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import itertools
from operator import itemgetter

from multiprocessing import Pool
import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

import terrain_analysis as ta
import speedup

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def ExtractSubGridProfile(axis=1044, dataset='SUBGRID'):

    subgrid_profile = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'PROFILE',
        'SUBGRID_PROFILE.shp'
    )

    measure_raster = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'AXIS_MEASURE.vrt'
    )

    if dataset == 'SUBGRID':

        population_raster = os.path.join(
            workdir,
            'SUBGRID',
            'POP_2015_ACC.tif'
        )

        landcover_raster = os.path.join(
            workdir,
            'SUBGRID',
            'LANDCOVER_2018_ACC.tif'
        )

    else:

        population_raster = os.path.join(
            workdir,
            'AXES',
            'AX%03d' % axis,
            dataset,
            'POP_2015_ACC.tif'
        )

        landcover_raster = os.path.join(
            workdir,
            'AXES',
            'AX%03d' % axis,
            dataset,
            'LANDCOVER_2018_ACC.tif'
        )

    # output = os.path.join(
    #     workdir,
    #     'METRICS'
    #     'AX%03d_SUBGRID_PROFILE.npz' % axis
    # )

    xy = list()
    # fij = list()

    with fiona.open(subgrid_profile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:
                xy.append(feature['geometry']['coordinates'][0])
                # fij.append((
                #     int(feature['id']),
                #     feature['properties']['i'],
                #     feature['properties']['i'])
                # )

    xy = np.array(xy, dtype='float32')
    # fij = np.array(fij)

    with rio.open(measure_raster) as measure_ds:
        measure = np.array(list(measure_ds.sample(xy, 1)))
        measure[measure == measure_ds.nodata] = np.nan

    with rio.open(population_raster) as pop_ds:
        pop = np.array(list(pop_ds.sample(xy, 1)))
        pop[pop == pop_ds.nodata] = np.nan

    with rio.open(landcover_raster) as landcover_ds:
        landcover = np.array(list(landcover_ds.sample(xy)))
        landcover[landcover == landcover_ds.nodata] = np.nan

    data = np.column_stack([measure, xy, pop, landcover])

    data = xr.Dataset(
        {
            'x': ('measure', data[:, 1]),
            'y': ('measure', data[:, 2]),
            'pop': ('measure', data[:, 3]),
            'lcc': (('measure', 'landcover'), data[:, 4:])
        },
        coords={
            'axis': axis,
            'measure': data[:, 0],
            'landcover': [
                'Water Channel',
                'Gravel Bars',
                'Natural Open',
                'Forest',
                'Grassland',
                'Crops',
                'Diffuse Urban',
                'Dense Urban',
                'Infrastructures'
            ],
        },
        attrs={
            'FCT': 'Fluvial Corridor Toolbox SubGrid Profile 1.0.5',
            'crs': 'EPSG:2154'
        }
    )

    return data.sortby(data['measure'])

def WriteSubGridProfile(data, axis, dataset='SUBGRID'):

    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', '%s.nc' % dataset)

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'x': dict(zlib=True, complevel=9, least_significant_digit=1),
            'y': dict(zlib=True, complevel=9, least_significant_digit=1),
            'pop': dict(zlib=True, complevel=9, least_significant_digit=3),
            'lcc': dict(zlib=True, complevel=9, least_significant_digit=2)
        })

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter
from Plotting import MapFigureSizer

def PlotMetric(data, fieldx, fieldy, filename=None, **kwargs):

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)

    ax.set_ylabel(fieldy)
    ax.set_xlabel(fieldx)
    ax.set_xlim([np.max(data[fieldx]), np.min(data[fieldx])])
    
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)

    ax.tick_params(axis='both', width=1, pad = 2)
    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)
    
    ax.grid(which='both', axis='both', alpha=0.5)

    x = data[fieldx]
    y = data[fieldy]

    if kwargs:
        y = y.sel(**kwargs)

    ax.plot(x, y, "#48638a", linewidth = 1)

    if filename is None:
        fig.show()
    elif filename.endswith('.pdf'):
        plt.savefig(filename, format='pdf', dpi=600)
        plt.clf()
    else:
        plt.savefig(filename, format='png', dpi=300)
        plt.clf()

def PlotPopulationProfile(data, dx=2000.0, title='', filename=None):

    x = data['measure']
    pop = data['pop']
    # pop = data['pop'].rolling(measure=5, min_periods=1, center=True).mean()

    # fig = plt.figure(1, facecolor='white', constrained_layout=True)
    # gs = plt.GridSpec(figure=fig, nrows=1, ncols=1)
    # ax = fig.add_subplot(gs[0, 0])

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(140,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    ax = fig.add_subplot(gs[25:100,10:95])

    xmin = np.min(x)
    xmax = np.max(x)
    # xx = (xmax - x[:, 0]) / scale_x
    xp = np.arange(xmin, xmax, dx)
    yp = np.zeros(xp.shape[0])
    bottom = np.zeros(xp.shape[0])

    y = pop
    fun = interp1d(x, y, kind='slinear', bounds_error=False)
    yp = -(fun(xp+dx) - fun(xp)) / dx
    yp[0] = -fun(dx) / dx

    yp[np.isnan(yp)] = 0.0
    yp = yp * 1000

    # ax = fig.add_subplot(gs[0, 0])

    ax.bar(xp, height=yp, width=dx, align='edge', color='#faafb4', edgecolor='k')
    ax.set_ylabel(u"Population × $10^3 / km$")

    ax.set_xlim(xmax, xmin)
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)

    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    # ax.set_title(titles[k])
    ax.tick_params(axis='both', width=1, pad = 2)

    for tick in ax.xaxis.get_major_ticks():
        tick.set_pad(2)

    ax.set_xlabel("Location along reference axis (from network outlet)")

    fig_size_inches = 9.25
    aspect_ratio = 3.0
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

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

def PlotStackedProfile(data, dx=2000.0, classes=None, title='', filename=None):

    colors = [
        '#a5bfdd', # Water
        '#cccccc', # Gravels
        '#bee62e', # Natural
        '#6f9e00', # Forest
        '#ffe45a', # Grassland
        '#ffff99', # Crops
        '#fa7c85', # Diffuse Urban
        '#fa1524', # Urban
        '#fa1665'  # Disconnected
    ]

    x = data['measure']
    # lcc = data['lcc']
    lcc = data['lcc'].rolling(measure=5, min_periods=1, center=True).mean()

    fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
    gs = plt.GridSpec(75,100,bottom=0.15,left=0.1,right=1.0,top=1.0)

    xmin = np.min(x)
    xmax = np.max(x)
    # xx = (xmax - x[:, 0]) / scale_x
    xp = np.arange(xmin, xmax, dx)
    yp = np.zeros((xp.shape[0], lcc.shape[1]))
    bottom = np.zeros(xp.shape[0])

    for k in range(lcc.shape[1]):

        y = lcc[:, k]
        fun = interp1d(x, y, kind='slinear', bounds_error=False)
        yp[:, k] = -(fun(xp+0.5*dx) - fun(xp-0.5*dx)) / dx
        yp[0, k] = -fun(dx) / dx

    yp[np.isnan(yp)] = 0
    yp = yp * 1000.0

    ax = fig.add_subplot(gs[10:45,10:95])
    ax.set_xlim(xmax, xmin)
    formatter = EngFormatter(unit='m')
    ax.xaxis.set_major_formatter(formatter)

    for k in range(lcc.shape[1]):

        if classes is not None and k not in classes:
            continue

        scaled = yp[:, k]
        ax.bar(xp, height=scaled, bottom=bottom, width=dx, align='center', color=colors[k], edgecolor='k')
        bottom = bottom + scaled

    ax.set_ylabel("$km^2 / km$")
    ax.set_xlabel("Location along reference axis (from network outlet)")

    fig_size_inches = 9.25
    aspect_ratio = 3.0
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

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

def PlotProfile(data, dx=2000.0, classes=None, title='', filename=None):

    colors = [
        '#a5bfdd', # Water
        '#cccccc', # Gravels
        '#bee62e', # Natural
        '#6f9e00', # Forest
        '#ffe45a', # Grassland
        '#ffff99', # Crops
        '#fa7c85', # Diffuse Urban
        '#fa1524', # Urban
        '#fa1665'  # Disconnected
    ]

    x = data['measure']
    lcc = data['lcc']
    # lcc = data['lcc'].rolling(measure=5, min_periods=1, center=True).mean()
    print(x.shape, lcc.shape)

    fig = plt.figure(1, facecolor='white', constrained_layout=True)
    # gs = plt.GridSpec(600,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
    gs = plt.GridSpec(figure=fig, nrows=len(classes), ncols=1)

    xmin = np.min(x)
    xmax = np.max(x)
    xp = np.arange(xmin, xmax, dx)
    yp = np.zeros((xp.shape[0], lcc.shape[1]))
    bottom = np.zeros(xp.shape[0])

    for k in range(lcc.shape[1]):

        y = lcc[:, k]
        fun = interp1d(x, y, kind='slinear', bounds_error=False)
        yp[:, k] = -(fun(xp+0.5*dx) - fun(xp-0.5*dx)) / dx
        yp[0, k] = -fun(dx) / dx

    yp[np.isnan(yp)] = 0.0
    yp = yp * 1000

    for k in range(lcc.shape[1]):

        if classes is not None and k not in classes:
            continue

        ax = fig.add_subplot(gs[k, 0])

        ax.bar(xp, height=yp[:, k], width=dx, align='center', color=colors[k], edgecolor='k')
        ax.set_ylabel("$km^2 / km$")
    
        ax.set_xlim(xmax, xmin)
        formatter = EngFormatter(unit='m')
        ax.xaxis.set_major_formatter(formatter)

        ax.spines['top'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        # ax.set_title(titles[k])
        ax.tick_params(axis='both', width=1, pad = 2)

        for tick in ax.xaxis.get_major_ticks():
            tick.set_pad(2)

    ax.set_xlabel("Location along reference axis (from network outlet)")

    fig_size_inches = 9.25
    aspect_ratio = 3.0
    cbar_L = "None"
    [fig_size_inches,map_axes,cbar_axes] = MapFigureSizer(fig_size_inches, aspect_ratio, cbar_loc=cbar_L, title=True)

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

def test():

    mpl.use('cairo')

    axis = 1044
    dx = 5000.0
    data = ExtractSubGridProfile(axis)

    PlotPopulationProfile(
        data,
        dx=dx,
        title='Population',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SUBGRID_POP.pdf'))

    PlotStackedProfile(
        data,
        dx=dx,
        title='Land Cover',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SUBGRID_LANDCOVER.pdf'))

    # PlotStackedProfile(
    #     data,
    #     dx=dx,
    #     title='Built Areas & Agriculture',
    #     classes={4, 5, 6, 7, 8},
    #     filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SUBGRID_STACKED_NON_NATURAL.pdf'))

    # PlotStackedProfile(
    #     data,
    #     dx=dx,
    #     title='Forest & Natural Areas',
    #     classes={0, 1, 2, 3},
    #     filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SUBGRID_NATURAL.pdf'))

    PlotStackedProfile(
        data,
        dx=dx,
        title='Agriculture',
        classes={4, 5},
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SUBGRID_AGRICULTURE.pdf'))

    PlotStackedProfile(
        data,
        dx=dx,
        title='Built Areas',
        classes={6, 7, 8},
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'SUBGRID_BUILT.pdf'))

def test_buffer(dataset='BUFFER200', subtitle='Tampon 200 m'):

    mpl.use('cairo')

    axis = 1044
    dx = 5000.0
    data = ExtractSubGridProfile(axis, dataset)
    WriteSubGridProfile(data, axis, dataset)

    pdfdir = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF')

    PlotPopulationProfile(
        data,
        dx=dx,
        title='Population (%s)' % subtitle,
        filename=os.path.join(pdfdir, '%s_POP.pdf' % dataset))

    PlotStackedProfile(
        data,
        dx=dx,
        title='Land Cover (%s)' % subtitle,
        filename=os.path.join(pdfdir, '%s_LANDCOVER.pdf' % dataset))

    PlotStackedProfile(
        data,
        dx=dx,
        title='Agriculture (%s)' % subtitle,
        classes={4, 5},
        filename=os.path.join(pdfdir, '%s_AGRICULTURE.pdf' % dataset))

    PlotStackedProfile(
        data,
        dx=dx,
        title='Built Areas (%s)' % subtitle,
        classes={6, 7, 8},
        filename=os.path.join(pdfdir, '%s_BUILT.pdf' % dataset))
