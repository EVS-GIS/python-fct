# coding: utf-8

"""
Valley Bottom

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
import fiona
import fiona.crs

from ..config import config
from ..cli import starcall
from .. import transform as fct
from .. import speedup
from ..tileio import PadRaster

from .ValleyBottom import (
    CropAndScaleRasterTile,
    border
)

def ExtendValleyBottomTile(axis, row, col, spillovers, max_distance):
    """
    ExtendValleyBottomToTopographicLimits Workhorse
    """

    tileset = config.tileset('landcover')
    # elevation_raster = tileset.tilename('tiled', row=tile.row, col=tile.col)
    # valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=tile.row, col=tile.col)
    output_height = tileset.tilename('ax_valley_bottom', axis=axis, row=row, col=col)
    output_distance = tileset.tilename('ax_valley_distance', axis=axis, row=row, col=col)

    elevations, profile1 = PadRaster(
        row,
        col,
        dataset='dem',
        tileset='landcover',
        padding=1)

    height, width = elevations.shape
    nodata1 = profile1['nodata']

    if not os.path.exists(output_height):

        MkValleyBottomNoDataTile(axis, row, col)

    with rio.open(output_height) as ds2:

        valley_bottom = ds2.read(1)
        transform = ds2.transform
        nodata2 = ds2.nodata
        profile2 = ds2.profile.copy()
        # print('from direct IO', transform, type(transform))

    with rio.open(output_distance) as ds3:

        distance = ds3.read(1)
        profile3 = ds3.profile.copy()
        nodata3 = ds3.nodata
        distance[distance == nodata3] = 0.0

    def intile(i, j):
        return all([i >= 0, i < height, j >= 0, j < width])

    for x, y, hand, dist in spillovers:

        i, j = fct.index(x, y, transform)
        if intile(i, j):

            valley_bottom[i, j] = hand
            distance[i, j] = dist

    reference = elevations - valley_bottom
    reference[(elevations == nodata1) | (valley_bottom == nodata2)] = nodata1
    control = np.copy(reference)

    speedup.valley_bottom_shortest(
        reference,
        elevations,
        nodata1,
        distance,
        max_dz=0.0,
        max_distance=max_distance)

    valley_bottom = elevations - reference
    valley_bottom[(elevations == nodata1) | (reference == nodata1)] = nodata2
    distance[(elevations == nodata1) | (reference == nodata1)] = nodata3

    output_height += '.tmp'
    output_distance += '.tmp'

    profile2.update(compress='deflate')
    profile3.update(compress='deflate')

    with rio.open(output_height, 'w', **profile2) as dst:
        dst.write(valley_bottom, 1)

    with rio.open(output_distance, 'w', **profile3) as dst:
        dst.write(distance, 1)

    spillovers = list()

    def attribute(pixel):
        """
        Returns value, dest. row, dest. col for pixel (i, j)
        """

        i, j = pixel
        x, y = fct.xy(i, j, transform)

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

        return x, y, reference[pixel], distance[pixel], prow, pcol

    for pixel in border(height, width):
        if control[pixel] == nodata1 and reference[pixel] != nodata1:
            spillovers.append(attribute(pixel))

    return spillovers, (output_height, output_distance)

def ExtendValleyBottomIteration(axis, spillovers, max_distance, processes=1):

    attributes = itemgetter(0, 1, 2, 3)
    tile = itemgetter(4, 5)

    spillovers = sorted(spillovers, key=tile)
    tiles = itertools.groupby(spillovers, key=tile)

    g_spillover = list()

    if processes == 1:

        for (row, col), seeds in tiles:

            seeds = [attributes(seed) for seed in seeds if not np.isnan(seed[0])]
            t_spillover, outputs = ExtendValleyBottomTile(axis, row, col, seeds, max_distance)
            g_spillover.extend(t_spillover)

            for tmpfile in outputs:
                os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    else:

        kwargs = dict()
        arguments = list()
        tmpfiles = list()

        for (row, col), seeds in tiles:
            seeds = [attributes(seed) for seed in seeds]
            arguments.append((ExtendValleyBottomTile, axis, row, col, seeds, max_distance, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)
            for t_spillover, outputs in pooled:
                g_spillover.extend(t_spillover)
                tmpfiles.extend(outputs)

            # with click.progressbar(pooled, length=len(arguments)) as bar:
            #     for t_spillover in bar:
            #         g_spillover.extend(t_spillover)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    return g_spillover

def CopyValleyBottomTile(axis, row, col):

    tileset = config.tileset('landcover')
    output_height = tileset.tilename('ax_valley_bottom', axis=axis, row=row, col=col)
    output_distance = tileset.tilename('ax_valley_distance', axis=axis, row=row, col=col)

    valley_bottom, profile = PadRaster(
        row,
        col,
        axis=axis,
        dataset='ax_relative_elevation',
        tileset='landcover',
        padding=1)

    distance = np.zeros_like(valley_bottom)

    profile.update(
        compress='deflate'
    )

    with rio.open(output_height, 'w', **profile) as dst:
        dst.write(valley_bottom, 1)

    with rio.open(output_distance, 'w', **profile) as dst:
        dst.write(distance, 1)

def MkValleyBottomNoDataTile(axis, row, col):

    tileset = config.tileset('landcover')
    template = tileset.tilename('tiled', row=row, col=col)
    output = tileset.tilename('ax_relative_elevation', axis=axis, row=row, col=col)

    with rio.open(template) as ds:

        data = np.full((ds.height, ds.width), ds.nodata, dtype='float32')
        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(data, 1)

    CopyValleyBottomTile(axis, row, col)

def CopyValleyBottom(axis, processes=1, **kwargs):

    tilefile = config.tileset().filename('ax_tiles', axis=axis)
    arguments = list()

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    for _, row, col in tiles:
        arguments.append((CopyValleyBottomTile, axis, row, col, kwargs))

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def CropAndScaleValleyBottom(axis, tiles, processes=1):

    def rasters(row, col):

        yield config.tileset('landcover').tilename('ax_valley_bottom', axis=axis, row=row, col=col), 1.0
        yield config.tileset('landcover').tilename('ax_valley_distance', axis=axis, row=row, col=col), 5.0

    kwargs = dict(padding=1)
    arguments = [
        (CropAndScaleRasterTile, rasterfile, scale, kwargs)
        for row, col in tiles
        for rasterfile, scale in rasters(row, col)
    ]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        with click.progressbar(pooled, length=len(arguments)) as processing:
            for _ in processing:
                pass

def ClipValleyBottom(axis, tiles, maxz):
    """
    Truncate data where height > maxz
    """

    with click.progressbar(tiles) as iterator:
        for row, col in iterator:

            rasterfile = config.tileset('landcover').tilename('ax_valley_bottom', axis=axis, row=row, col=col)
            
            with rio.open(rasterfile) as ds:

                data = ds.read(1)
                profile = ds.profile.copy()
                profile.update(compress='deflate')

            data[data > maxz] = ds.nodata

            with rio.open(rasterfile, 'w', **profile) as dst:
                dst.write(data, 1)


def ExtendValleyBottomToTopographicLimits(axis, max_distance, processes=1):
    """
    Extend valley bottom to topographic limits,
    using shortest path space exploration.
    """

    # 1. copy ax_relative_elevation to ax_valley_bottom with 1 pixel padding
    #    create distance tiles
    # 2. iterate over tiles until no more spillovers
    # 3. crop tiles

    click.secho('Copy valley bottom', fg='cyan')
    
    CopyValleyBottom(axis, processes)

    click.secho('Extend valley bottom', fg='cyan')
    
    iteration = 1
    g_tiles = set()
    tile = itemgetter(4, 5)

    tilefile = config.tileset().filename('ax_tiles', axis=axis)

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    spillovers = [
        (np.nan, np.nan, 0.0, 0.0, row, col)
        for _, row, col in tiles
    ]

    while spillovers:

        tiles = set([tile(s) for s in spillovers])
        g_tiles.update(tiles)
        iteration += 1
        spillovers = ExtendValleyBottomIteration(axis, spillovers, max_distance, processes)

    click.echo('Done after %d iterations' % iteration)

    click.secho('Crop and scale output tiles', fg='cyan')
    
    CropAndScaleValleyBottom(axis, g_tiles, processes)
    ClipValleyBottom(axis, g_tiles, 15.0)

    click.secho('Save axis tiles list', fg='cyan')

    output = config.tileset().filename('ax_tiles', axis=axis)

    with open(output, 'w') as fp:
        for row, col in sorted(g_tiles):
            fp.write('%d,%d,%d\n' % (axis, row, col))
