#!/usr/bin/env python
# coding: utf-8

"""
Accumulate values from metric raster
according to D8 flow raster.

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
from collections import Counter

from multiprocessing import Pool
import numpy as np
import xarray as xr

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

import terrain_analysis as ta
import speedup

from SubGridProfile import PlotMetric

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def starcall(args):
    """
    Invoke first arg function with all other arguments.
    """

    fun = args[0]
    return fun(*args[1:-1], **args[-1])

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def TileOutlets(row, col, bounds, conv=1.0, band=1, **kwargs):
    
    flow_raster = kwargs['flow']
    rasterfile = kwargs['raster'](row, col)
    nodata = -99999.0

    with rio.open(rasterfile) as ds1:

        window1 = as_window(bounds, ds1.transform)
        data = ds1.read(band, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(flow_raster) as ds2:

            window2 = as_window(bounds, ds2.transform)
            flow = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)
            height, width = flow.shape

            acc = np.float32(data)
            acc[data == ds1.nodata] = 0
            speedup.flow_accumulation(flow, acc)

            outlets, targets = speedup.outlets(flow)

            def translate(i, j):

                if i < 0:
                    trow = row - 1
                    ti = i + height
                elif i >= height:
                    trow = row + 1
                    ti = i - height
                else:
                    trow = row
                    ti = i

                if j < 0:
                    tcol = col - 1
                    tj = j + width
                elif j >= width:
                    tcol = col + 1
                    tj = j - width
                else:
                    tcol = col
                    tj = j

                return trow, tcol, ti, tj

            # connect outlet->inlet

            return {
                (row, col, i, j): translate(ti, tj) + (conv*acc[i, j],)
                for (i, j), (ti, tj)
                in zip(outlets, targets)
            }

def TileConnectInlets(row, col, bounds, inlets, **kwargs):

    flow_raster = kwargs['flow']

    graph = dict()

    with rio.open(flow_raster) as ds:

        window = as_window(bounds, ds.transform)
        flow = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        for _, _, i, j in inlets:

            # connect inlet->tile outlet
            
            io, jo = ta.outlet(flow, i, j)
            
            if io == i and jo == j:
                continue

            graph[row, col, i, j] = (row, col, io, jo, 0.0)

    return graph

def TileAccumulate(row, col, bounds, inlets, conv=1.0, band=1, **kwargs):
    
    flow_raster = kwargs['flow']
    rasterfile = kwargs['raster'](row, col)
    output = kwargs['output'](row, col)
    nodata = -99999.0

    with rio.open(rasterfile) as ds1:

        window1 = as_window(bounds, ds1.transform)
        data = ds1.read(band, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(flow_raster) as ds2:

            window2 = as_window(bounds, ds2.transform)
            flow = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)
            transform = ds2.transform * ds2.transform.translation(window2.col_off, window2.row_off)

        out = np.float32(np.copy(data) * conv)
        out[data == ds1.nodata] = 0.0

        for ti, tj, value in inlets:
            out[ti, tj] += value

        speedup.flow_accumulation(flow, out)

        out[data == ds1.nodata] = nodata

        profile = ds1.profile.copy()
        profile.update(
            dtype='float32',
            nodata=nodata,
            compress='deflate',
            transform=transform
        )

        if os.path.exists(output) and band > 1:
            with rio.open(output, 'r+') as dst:
                dst.write(out, band)
        else:
            with rio.open(output, 'w', **profile) as dst:
                dst.write(out, band)

def Accumulate(processes, **kwargs):
    
    tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')
    tiles = dict()

    with fiona.open(tile_shapefile) as fs:
        
        arguments = list()
        
        for feature in fs:

            minx, miny, maxx, maxy = fs.bounds

            properties = feature['properties']
            x0 = properties['left']
            y0 = properties['top']
            x1 = properties['right']
            y1 = properties['bottom']
            bounds = (x0, y1, x1, y0)
            
            row = int((maxy - y0) // 10000)
            col = int((x0 - minx) // 10000)
            tiles[row, col] = bounds
            
            arguments.append((TileOutlets, row, col, bounds, kwargs))

    tile = itemgetter(0, 1)
    coords = itemgetter(2, 3)
    graph = dict()

    # 1. Find tile outlets

    click.secho('Find tile outlets', fg='cyan')

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _graph in iterator:
                graph.update(_graph)   

    outlets = set(graph.keys())

    # 2. Resolve inlet/outlet graph

    click.secho('Resolve inlet/outlet graph', fg='cyan')

    arguments = list()
    inlets = sorted([(row, col, i, j) for row, col, i, j, _ in graph.values()], key=tile)
    groups = itertools.groupby(inlets, key=tile)

    for (row, col), items in groups:
        if (row, col) in tiles:

            bounds = tiles[row, col]
            arguments.append((TileConnectInlets, row, col, bounds, list(items), kwargs))

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _graph in iterator:
                graph.update(_graph)

    click.secho('Accumulate graph', fg='cyan')

    areas = speedup.graph_acc2(graph)

    # 3. Accumulate tiles with inlets contributions

    click.secho('Accumulate tiles', fg='cyan')

    arguments = list()
    # bnodes = {(tr, tc, ti, tj) for tr, tc, ti, tj, _ in graph.values()}
    inlets = sorted(graph.keys(), key=tile)
    groups = itertools.groupby(inlets, key=tile)

    def contribution(pixel):

        t = tile(pixel)
        ij = coords(pixel)
        ti, tj = ij
        
        if (t, ij) in areas:
            return (ti, tj, areas[t, ij])

        return (ti, tj, 0.0)

    for (row, col), items in groups:
        if (row, col) in tiles:

            contributions = [contribution(item) for item in items if item not in outlets]
            bounds = tiles[row, col]
            arguments.append((TileAccumulate, row, col, bounds, contributions, kwargs))

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def population_tile(row, col):

    return os.path.join(
        workdir,
        'GLOBAL',
        'POPULATION',
        'POP_INSEE_%02d_%02d.tif' % (row, col))

def population_output(row, col):

    return os.path.join(
        workdir,
        'GLOBAL',
        'ACC',
        'POP_INSEE_ACC_%02d_%02d.tif' % (row, col))

def AccumulatePopulation(processes=1):

    flow_raster = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'

    Accumulate(
        processes,
        flow=flow_raster,
        raster=population_tile,
        conv=1e-3,
        output=population_output)

def TileLandCoverSplit(row, col, bands=1, **kwargs):
    """
    Split land cover classes
    into separate contingency bands
    """

    rasterfile = kwargs['raster'](row, col)
    output = kwargs['output'](row, col)

    with rio.open(rasterfile) as ds:

        data = ds.read(1)

        profile = ds.profile.copy()
        profile.update(
            count=bands,
            dtype='uint8',
            nodata=255
        )

        with rio.open(output, 'w', **profile) as dst:
            for k in range(bands):
                
                band = np.uint8(data == k)
                band[data == ds.nodata] = 255
                dst.write(band, k+1)

def LandCoverSplit():
    """
    Split land cover classes
    into separate contingency bands
    """

    tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')

    rasterfile = lambda row, col: os.path.join(
        workdir,
        'GLOBAL',
        'LANDCOVER',
        'CESBIO_%02d_%02d.tif' % (row, col))

    output = lambda row, col: os.path.join(
        workdir,
        'GLOBAL',
        'ACC',
        'CESBIO_MB_%02d_%02d.tif' % (row, col))

    with fiona.open(tile_shapefile) as fs:

        minx, miny, maxx, maxy = fs.bounds

        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                x0 = properties['left']
                y0 = properties['top']
                x1 = properties['right']
                y1 = properties['bottom']
                bounds = (x0, y1, x1, y0)

                row = int((maxy - y0) // 10000)
                col = int((x0 - minx) // 10000)

                TileLandCoverSplit(
                    row, col,
                    bands=9,
                    raster=rasterfile,
                    output=output)

def landcover_tile(row, col):

    return os.path.join(
        workdir,
        'GLOBAL',
        'ACC',
        'CESBIO_MB_%02d_%02d.tif' % (row, col))

def landcover_output(row, col):

    return os.path.join(
        workdir,
        'GLOBAL',
        'ACC',
        'CESBIO_ACC_%02d_%02d.tif' % (row, col))

def AccumulateLandCover(processes=1, bands=9):

    flow_raster = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'

    classes = [
        'Water Channel',
        'Gravel Bars',
        'Natural Open',
        'Forest',
        'Grassland',
        'Crops',
        'Diffuse Urban',
        'Dense Urban',
        'Infrastructures'
    ]
    
    for k in range(bands):

        click.echo('Processing cover class %d (%s)' % (k, classes[k]))

        Accumulate(
            processes,
            flow=flow_raster,
            raster=landcover_tile,
            conv=25e-6,
            output=landcover_output,
            band=k+1)

def ExtractCumulativeProfile(axis=1044):

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

    population_raster = os.path.join(
        workdir,
        'GLOBAL',
        'POP_2015_ACC.vrt'
    )

    landcover_raster = os.path.join(
        workdir,
        'GLOBAL',
        'LANDCOVER_2018_ACC.vrt'
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

    return xr.Dataset(
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
        }
    )

    # print(data.shape, data.dtype)

    # dtype = np.dtype([
    #     ('x', 'float64'),
    #     ('y', 'float64'),
    #     ('measure', 'float32'),
    #     ('population', 'float32'),
    #     ('water', 'float32'),
    #     ('gravel', 'float32'),
    #     ('natural', 'float32'),
    #     ('forest', 'float32'),
    #     ('grassland', 'float32'),
    #     ('crops', 'float32'),
    #     ('diffuse', 'float32'),
    #     ('dense', 'float32'),
    #     ('infrast', 'float32')
    # ])

    # return np.sort(np.array([tuple(data[k, :]) for k in range(data.shape[0])], dtype=dtype), order='measure')

def WriteCumulativeProfile(axis, data):

    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'WATERSHED_PROFILE.nc')

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'x': dict(zlib=True, complevel=9, least_significant_digit=1),
            'y': dict(zlib=True, complevel=9, least_significant_digit=1),
            'pop': dict(zlib=True, complevel=9, least_significant_digit=3),
            'lcc': dict(zlib=True, complevel=9, least_significant_digit=2)
        })
