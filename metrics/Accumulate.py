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

from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

import terrain_analysis as ta
import speedup

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

def TileOutlets(row, col, bounds, band=1, **kwargs):
    
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

            return [
                translate(ti, tj) + (data[i, j],)
                for (i, j), (ti, tj)
                in zip(outlets, targets)
            ]

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

        for _, _, ti, tj, value in inlets:
            out[ti, tj] += value * conv

        speedup.flow_accumulation(flow, out)

        out[data == ds1.nodata] = nodata

        profile = ds1.profile.copy()
        profile.update(
            dtype='float32',
            nodata=nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, band)

def Accumulate(processes, **kwargs):
    
    tile_shapefile = '/media/crousson/Backup/PRODUCTION/OCSOL/GRILLE_10K_AIN.shp'
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
    outlets = list()

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _outlets in iterator:
                outlets.extend(sorted(_outlets, key=tile))

    arguments = list()
    outlets = sorted(outlets, key=tile)
    groups = itertools.groupby(outlets, key=tile)

    for (row, col), inlets in groups:
        if (row, col) in tiles:

            bounds = tiles[row, col]
            arguments.append((TileAccumulate, row, col, bounds, inlets, kwargs))

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def population_tile(row, col):

    return os.path.join(
        workdir,
        'METRICS',
        'POP_INSEE_%02d_%02d.tif' % (row, col))

def population_output(row, col):

    return os.path.join(
        workdir,
        'METRICS',
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

    tile_shapefile = '/media/crousson/Backup/PRODUCTION/OCSOL/GRILLE_10K_AIN.shp'

    rasterfile = lambda row, col: os.path.join(
        workdir,
        'OCS',
        'CESBIO_%02d_%02d.tif' % (row, col))

    output = lambda row, col: os.path.join(
        workdir,
        'OCS',
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
        'OCS',
        'CESBIO_MB_%02d_%02d.tif' % (row, col))

def landcover_output(row, col):

    return os.path.join(
        workdir,
        'OCS',
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
