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

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def TileStreamBuffer(row, col, bounds, distance, **kwargs):

    rasterfile = kwargs['raster']
    output = kwargs['output']
    fill = kwargs['fill']

    with rio.open(rasterfile) as ds:

        window = as_window(bounds, ds.transform)
        landcover = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        height, width = landcover.shape
        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        out = np.float32((landcover == 0) | (landcover == 1))

        speedup.raster_buffer(
            out,
            0.0,
            distance,
            fill)

        out = np.uint8(out)
        # out[landcover == ds.nodata] = 255

        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            dtype='uint8',
            nodata=255,
            compress='deflate',
            height=height,
            width=width,
            transform=transform
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def StreamBuffer(axis, distance, fill=2.0):

    tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')

    resolution = 5.0

    rasterfile = os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'CONTINUITY.vrt')

    output = lambda row, col: os.path.join(
        workdir,
        'AXES',
        'AX%03d' % axis,
        'TILES',
        'BUFFER%.0f_%02d_%02d.tif' % (distance, row, col))

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

                TileStreamBuffer(
                    row, col, bounds,
                    raster=rasterfile,
                    output=output(row, col),
                    distance=distance / resolution,
                    fill=fill)

def test():

    for width in (30, 100, 200):

        click.echo('Create buffer of width = %.1f m' % width)
        StreamBuffer(width)
