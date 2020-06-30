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

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

import terrain_analysis as ta
import speedup

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def ExtractSubGridProfile1(axis=1044):

    subgrid_profile = os.path.join(
        workdir,
        'SUBGRID',
        'AX%03d_SUBGRID_PROFILE.shp' % axis
    )

    measure_raster = os.path.join(
        workdir,
        'AX%03d_AXIS_MEASURE.vrt' % axis
    )

    population_raster = os.path.join(
        workdir,
        'METRICS',
        'POP_INSEE_ACC.vrt'
    )

    landcover_raster = os.path.join(
        workdir,
        'OCS',
        'CESBIO_ACC.vrt'
    )

    # output = os.path.join(
    #     workdir,
    #     'METRICS'
    #     'AX%03d_SUBGRID_PROFILE.npz' % axis
    # )

    with fiona.open(subgrid_profile) as fs:
        with click.progressbar(fs) as iterator:
            xy = np.array([
                feature['geometry']['coordinates'][0]
                for feature in iterator
            ])

    with rio.open(measure_raster) as measure_ds:
        measure = np.array(list(measure_ds.sample(xy, 1)))
        measure[measure == measure_ds.nodata] = np.nan

    with rio.open(population_raster) as pop_ds:
        pop = np.array(list(pop_ds.sample(xy, 1)))
        pop[pop == pop_ds.nodata] = np.nan

    with rio.open(landcover_raster) as landcover_ds:
        landcover = np.array(list(landcover_ds.sample(xy)))
        landcover[landcover == landcover_ds.nodata] = np.nan

    data = np.column_stack([xy, measure, pop, landcover])
    print(data.shape, data.dtype)

    dtype = np.dtype([
        ('x', 'float64'),
        ('y', 'float64'),
        ('measure', 'float32'),
        ('population', 'float32'),
        ('water', 'float32'),
        ('gravel', 'float32'),
        ('natural', 'float32'),
        ('forest', 'float32'),
        ('grassland', 'float32'),
        ('crops', 'float32'),
        ('diffuse', 'float32'),
        ('dense', 'float32'),
        ('infrast', 'float32')
    ])

    return np.sort(np.array([tuple(data[k, :]) for k in range(data.shape[0])], dtype=dtype), order='measure')

def ExtractSubGridProfile(axis=1044):

    subgrid_profile = os.path.join(
        workdir,
        'SUBGRID',
        'AX%03d_SUBGRID_PROFILE.shp' % axis
    )

    measure_raster = os.path.join(
        workdir,
        'AX%03d_AXIS_MEASURE.vrt' % axis
    )

    population_raster = os.path.join(
        workdir,
        'SUBGRID',
        'POP_INSEE_2015_ACC.tif'
    )

    landcover_raster = os.path.join(
        workdir,
        'SUBGRID',
        'LANDCOVER_CESBIO_2018_ACC.tif'
    )

    # output = os.path.join(
    #     workdir,
    #     'METRICS'
    #     'AX%03d_SUBGRID_PROFILE.npz' % axis
    # )

    xy = list()
    fij = list()

    with fiona.open(subgrid_profile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:
                xy.append(feature['geometry']['coordinates'][0])
                fij.append((
                    int(feature['id']),
                    feature['properties']['i'],
                    feature['properties']['i'])
                )

    xy = np.array(xy)
    fij = np.array(fij)

    with rio.open(measure_raster) as measure_ds:
        measure = np.array(list(measure_ds.sample(xy, 1)))
        measure[measure == measure_ds.nodata] = np.nan

    with rio.open(population_raster) as pop_ds:
        pop = np.array(list(pop_ds.sample(xy, 1)))
        pop[pop == pop_ds.nodata] = np.nan

    with rio.open(landcover_raster) as landcover_ds:
        landcover = np.array(list(landcover_ds.sample(xy)))
        landcover[landcover == landcover_ds.nodata] = np.nan

    data = np.column_stack([xy, measure, pop, landcover])
    print(data.shape, data.dtype)

    dtype = np.dtype([
        ('x', 'float64'),
        ('y', 'float64'),
        ('measure', 'float32'),
        ('population', 'float32'),
        ('water', 'float32'),
        ('gravel', 'float32'),
        ('natural', 'float32'),
        ('forest', 'float32'),
        ('grassland', 'float32'),
        ('crops', 'float32'),
        ('diffuse', 'float32'),
        ('dense', 'float32'),
        ('infrast', 'float32'),
        ('dummy', 'float32')
    ])

    return np.sort(np.array([tuple(data[k, :]) for k in range(data.shape[0])], dtype=dtype), order='measure')

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import EngFormatter

def PlotMetric(data, fieldx, fieldy):

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

    ax.plot(data[fieldx], data[fieldy], "#48638a", linewidth = 1)

    fig.show()
