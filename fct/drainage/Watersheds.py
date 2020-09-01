# coding: utf-8

"""
Watershed Analysis

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
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
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..config import config
from ..cli import starcall
from .. import terrain_analysis as ta
from .. import speedup
from ..tileio import PadRaster

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

def border(height, width):
    """
    Generate a sequence of pixel (row, col)
    over the border of an array of shape (height, width)
    """

    offset = 0
    for i in (0, height-1):
        for j in range(offset, width+offset-1):
            yield i, j
        offset = 1

    offset = 1
    for j in (0, width-1):
        for i in range(offset, height+offset-1):
            yield i, j
        offset = 0

def WatershedTile(row, col, seeds, axis=1, tmp=''):
    """
    seeds: (x, y, value, distance)
    """

    flow, profile = PadRaster(row, col, 'flow', padding=1)
    transform = profile['transform']
    destination = config.tileset().tilename('ax_watershed_raster', row=row, col=col, axis=axis)
    height, width = flow.shape

    if os.path.exists(destination):
        # with rio.open(destination) as ds:
        #     ds.read(1, out=out[1:-1, 1:-1])
        data, _ = PadRaster(row, col, 'ax_watershed_raster', padding=1, axis=axis)
    else:
        data = np.zeros_like(flow, dtype='float32')

    out = np.copy(data)
    coord = itemgetter(0, 1)
    value = itemgetter(2)
    pixels = ta.worldtopixel(np.array([coord(seed) for seed in seeds], dtype='float32'), transform, gdal=False)
    values = np.array([value(seed) for seed in seeds], dtype='float32')
    out[pixels[:, 0], pixels[:, 1]] = values
    # ta.watershed(flow, out, 0)
    speedup.watershed(flow, out, 0)

    spillover = list()
    for i, j in border(height, width):
        if data[i, j] == 0 and out[i, j] != 0:
            spillover.append((i, j))

    def attribute(p):
        """
        Returns value, dest. row, dest. col for pixel (i, j)
        """

        i, j = p

        if i == 0:
            prow = row - 1
        elif i == height-1:
            prow = row + 1
        else:
            prow = row

        if j == 0:
            pcol = col - 1
        elif j == width-1:
            pcol = col + 1
        else:
            pcol = col

        return out[p], prow, pcol

    if spillover:
        points = ta.pixeltoworld(np.array(spillover, dtype='int32'), transform, gdal=False)
        spillover = [tuple(points[k]) + attribute(p) for k, p in enumerate(spillover)]

    out = out[1:-1, 1:-1]
    height, width = out.shape
    transform = transform * transform.translation(1, 1)
    profile.update(dtype='float32', height=height, width=width, transform=transform, nodata=0)

    with rio.open(destination + tmp, 'w', **profile) as dst:
        dst.write(out, 1)

    return spillover, destination + tmp

def VectorizeTile(axis, row, col):
    """
    DOCME
    """

    rasterfile = config.tileset().tilename(
        'ax_watershed_raster',
        axis=axis,
        row=row,
        col=col)

    if os.path.exists(rasterfile):

        with rio.open(rasterfile) as ds:
            watershed = ds.read(1)
            transform = ds.transform

        polygons = features.shapes(
            watershed,
            connectivity=4,
            transform=transform)

        return [polygon for polygon, value in polygons if value == axis], row, col

    else:

        return [], row, col

def VectorizeWatershed(axis, processes=1):
    """
    DOCME
    """

    output = config.filename('ax_watershed', axis=axis) # filename ok

    epsg = 2154
    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('AXIS', 'int:4'),
            ('ROW', 'int:3'),
            ('COL', 'int:3')
        ]
    }
    crs = fiona.crs.from_epsg(epsg)

    options = dict(
            driver='ESRI Shapefile',
            crs=crs,
            schema=schema
        )

    if processes == 1:

        with fiona.open(output, 'w', **options) as dst:
            with click.progressbar(tileindex()) as bar:
                for row, col in bar:
                    polygons, _, _ = VectorizeTile(axis, row, col)
                    for polygon in polygons:
                        geom = asShape(polygon).buffer(0.0)
                        feature = {
                            'geometry': geom.__geo_interface__,
                            'properties': {
                                'AXIS': axis,
                                'ROW': row,
                                'COL': col
                            }
                        }
                        dst.write(feature)

    else:

        kwargs = dict()
        arguments = [(VectorizeTile, axis, row, col, kwargs) for row, col in tileindex()]

        with fiona.open(output, 'w', **options) as dst:
            with Pool(processes=processes) as pool:

                pooled = pool.imap_unordered(starcall, arguments)
                with click.progressbar(pooled, length=len(arguments)) as bar:

                    for polygons, row, col in bar:
                        for polygon in polygons:
                            geom = asShape(polygon).buffer(0.0)
                            feature = {
                                'geometry': geom.__geo_interface__,
                                'properties': {
                                    'AXIS': axis,
                                    'ROW': row,
                                    'COL': col
                                }
                            }
                            dst.write(feature)

def print_spillover_tiles(spillover):

    tile = itemgetter(3, 4)
    tiles = set([tile(s) for s in spillover])
    print(tiles)

def WatershedStep(spillover, axis=1, processes=1):

    xy = itemgetter(0, 1)
    value = itemgetter(2)
    tile = itemgetter(3, 4)

    spillover = sorted(spillover, key=tile)
    tiles = itertools.groupby(spillover, key=tile)

    g_spillover = list()
    tmpfiles = list()

    if processes == 1:

        for (row, col), seeds in tiles:
            seeds = [xy(seed) + (value(seed),) for seed in seeds]
            t_spillover, tmpfile = WatershedTile(row, col, seeds, axis=axis, tmp='.tmp')
            g_spillover.extend(t_spillover)
            tmpfiles.append(tmpfile)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    else:

        kwargs = {'tmp': '.tmp', 'axis': axis}
        arguments = list()

        for (row, col), seeds in tiles:
            seeds = [xy(seed) + (value(seed),) for seed in seeds]
            arguments.append((WatershedTile, row, col, seeds, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)
            for t_spillover, tmpfile in pooled:
                g_spillover.extend(t_spillover)
                tmpfiles.append(tmpfile)

            # with click.progressbar(pooled, length=len(arguments)) as bar:
            #     for t_spillover in bar:
            #         g_spillover.extend(t_spillover)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    return g_spillover

def Watershed(axis, processes=1):

    # River Ain
    # x = 869165.0
    # y = 6523885.0

    # River Rhone
    x = 849265.0
    y = 6250190.0
    # row, col = config.tileset().index(x, y)

    # seeds = [(x, y, 1, row, col)]

    def generate_seeds(feature):

        row = feature['properties']['ROW']
        col = feature['properties']['COL']

        for point in feature['geometry']['coordinates']:
            x, y = point[:2]
            yield (x, y, axis, row, col)


    drainage_shapefile = config.tileset().filename(
        'ax_drainage_network',
        axis=axis)

    with fiona.open(drainage_shapefile) as fs:
        
        seeds = [
            seed
            for feature in fs
            for seed in generate_seeds(feature)
        ]

    count = 0
    tile = itemgetter(3, 4)

    click.secho('Watershed ID = %d' % axis, fg='cyan')
    click.secho('Run %d processes' % processes, fg='yellow')

    while seeds:

        seeds = WatershedStep(seeds, axis, processes)

        count += 1
        tiles = {tile(s) for s in seeds}
        click.echo('Step %02d -- %d spillovers, %d tiles' % (count, len(seeds), len(tiles)))

    click.secho('Ok', fg='green')
