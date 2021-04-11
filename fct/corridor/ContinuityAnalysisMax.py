# coding: utf-8

"""
LandCover Lateral Continuity Analysis

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
from operator import itemgetter
import itertools
import logging
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window

from .. import speedup
from ..cli import starcall
from ..config import (
    config,
    LiteralParameter,
    DatasetParameter
)
from .. import transform as fct
from ..tileio import (
    PadRaster,
    border
)
from .ValleyBottomFeatures import MASK_EXTERIOR

class Parameters:
    """
    Continuity analysis parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    landcover = DatasetParameter('landcover raster map', type='input')
    distance = DatasetParameter('distance to talweg', type='input')

    output = DatasetParameter('continuity map', type='output')
    output_distance = DatasetParameter('distance to reference pixel', type='output')
    state = DatasetParameter('processing state raster', type='output')

    class_max = LiteralParameter('maximum landcover class (stop criterion)')
    distance_min = LiteralParameter('minimum distance')
    distance_max = LiteralParameter('maximum distance')
    padding = LiteralParameter('tile padding in pixels')
    infrastructures = LiteralParameter('consider transport infrastructures (landcover class = 8)')
    jitter = LiteralParameter('apply jitter on performing shortest path raster exploration')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.landcover = 'landcover_valley_bottom'
            self.distance = 'nearest_distance'
            self.output = 'continuity'
            self.output_distance = 'continuity_distance'
            self.state = 'continuity_state'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.landcover = dict(key='ax_landcover_valley_bottom', axis=axis)
            self.distance = dict(key='ax_nearest_distance', axis=axis)
            self.output = dict(key='ax_continuity', axis=axis)
            self.output_distance = dict(key='ax_continuity_distance', axis=axis)
            self.state = dict(key='ax_continuity_state', axis=axis)

        self.class_max = 0
        self.distance_min = 20
        self.distance_max = 0
        self.padding = 1
        self.infrastructures = True
        self.jitter = 0.4
        self.tmp_suffix = '.tmp'

def ContinuityTile(row, col, seeds, params, **kwargs):
    """
    Tile Implementation
    """

    tileset = config.tileset()

    output = str(params.output.tilename(row=row, col=col, **kwargs))
    output_state = str(params.state.tilename(row=row, col=col, **kwargs))
    output_distance = str(params.output_distance.tilename(row=row, col=col, **kwargs))

    padding = params.padding
    tile = tileset.tileindex[row, col]

    landcover, profile = PadRaster(
        row, col,
        params.landcover,
        padding=padding)

    transform = profile['transform']
    nodata = profile['nodata']
    height, width = landcover.shape

    if not params.infrastructures:
        # Remove infrastructures
        infrastructure_mask = (landcover == 8)
        landcover[infrastructure_mask] = 2

    def BoundlessRaster(row, col, dataset):
        """
        Read tile with some padding,
        handling the case if the tile does not exist.
        """

        filename = dataset.tilename(row=row, col=col, **kwargs)

        if os.path.exists(filename):

            raster, profile = PadRaster(row, col, dataset, padding=padding, **kwargs)
            nodata = profile['nodata']

        else:

            filename = dataset.filename(**kwargs)
            assert os.path.exists(filename)

            with rio.open(filename) as ds:

                i0, j0 = ds.index(tile.x0, tile.y0)
                window = Window(j0 - padding, i0 - padding, width, height)
                raster = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
                nodata = ds.nodata

        return raster, nodata

    nearest_distance, nearest_distance_nodata = BoundlessRaster(row, col, params.distance)

    if os.path.exists(output):

        out, _ = PadRaster(row, col, params.output, padding=padding, **kwargs)
        distance, _ = PadRaster(row, col, params.output_distance, padding=padding)
        state, _ = PadRaster(row, col, params.state, padding=padding)

    else:

        out = np.full_like(landcover, nodata)
        distance = np.zeros_like(nearest_distance, dtype='float32')
        state = np.uint8(np.abs(nearest_distance) < 1)

    state[landcover == nodata] = 255

    if seeds:

        coordxy = itemgetter(0, 1)
        pixels = fct.worldtopixel(
            np.array([coordxy(seed) for seed in seeds], dtype='float32'),
            transform)
        intile = np.array([0 <= i < height and 0 <= j < width for i, j in pixels])

        seed_value = np.array([seed[2] for seed in seeds], dtype='uint8')
        seed_distance = np.array([seed[3] for seed in seeds], dtype='float32')

        pixels = pixels[intile]
        seed_value = seed_value[intile]
        seed_distance = seed_distance[intile]

        recorded_value = out[pixels[:, 0], pixels[:, 1]]
        recorded_distance = distance[pixels[:, 0], pixels[:, 1]]

        shortest = (
            (recorded_distance == nearest_distance_nodata) |
            (recorded_distance == 0) |
            (seed_value < recorded_value) |
            (
                (seed_value == recorded_value) &
                (seed_distance < recorded_distance)
            )
        )

        pixels = pixels[shortest]
        out[pixels[:, 0], pixels[:, 1]] = seed_value[shortest]
        distance[pixels[:, 0], pixels[:, 1]] = seed_distance[shortest]
        state[pixels[:, 0], pixels[:, 1]] = 1

    # Continuity analysis on shortest path

    del nearest_distance
    # cost = params.costf(landcover)
    control = np.copy(out)

    # unlock previously resolved cells
    state[state == 2] = 6

    speedup.layered_continuity_analysis(
        landcover,
        # nearest_height,
        out,
        distance,
        state,
        max_class=params.class_max,
        min_distance=params.distance_min,
        max_distance=params.distance_max,
        # max_height=params.height_max,
        jitter=params.jitter)

    def extract_spillovers():

        spillovers = [
            (i, j) for i, j in border(height, width)
            if control[i, j] != out[i, j]
        ]

        if spillovers:

            xy = fct.pixeltoworld(np.array(spillovers, dtype='int32'), transform)

            def attributes(k, ij):
                """
                Returns (row, col, x, y, height, distance)
                """

                i, j = ij

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

                return (prow, pcol) + tuple(xy[k]) + (out[i, j], distance[i, j])

            spillovers = [
                attributes(k, ij)
                for k, ij in enumerate(spillovers)
            ]

        return spillovers

    spillovers = extract_spillovers()

    # restore unresolved cells' state
    state[state == 1] = 2

    if not params.infrastructures:
        # Restore infrastructures
        out[infrastructure_mask & (out != nodata)] = 8

    out[(landcover == 0) & (out == 1)] = 0

    # Crop out nodata and padded border

    out[landcover == nodata] = nodata
    out = out[padding:-padding, padding:-padding]
    state = state[padding:-padding, padding:-padding]
    distance = distance[padding:-padding, padding:-padding]

    # Output

    height, width = out.shape
    transform = transform * transform.translation(padding, padding)

    tmp_suffix = params.tmp_suffix
    output += tmp_suffix
    output_state += tmp_suffix
    output_distance += tmp_suffix

    profile.update(
        driver='GTiff',
        height=height,
        width=width,
        transform=transform,
        compress='deflate')

    with rio.open(output, 'w', **profile) as dst:
        dst.write(out, 1)

    with rio.open(output_state, 'w', **profile) as dst:
        dst.write(state, 1)

    profile.update(dtype='float32', nodata=nearest_distance_nodata)

    with rio.open(output_distance, 'w', **profile) as dst:
        dst.write(distance, 1)

    return spillovers, (output, output_state, output_distance)

def ContinuityIteration(params, spillovers, ntiles, processes=1, **kwargs):
    """
    Iteration over spillovers' destination tiles
    """

    tile = itemgetter(0, 1)
    coordxy = itemgetter(2, 3)
    values = itemgetter(4, 5)

    spillovers = sorted(spillovers, key=tile)
    tiles = itertools.groupby(spillovers, key=tile)

    g_spillover = list()
    tmpfiles = list()

    if processes == 1:

        for (row, col), seeds in tiles:
            seeds = [coordxy(seed) + values(seed) for seed in seeds]
            t_spillover, tmps = ContinuityTile(row, col, seeds, params)
            g_spillover.extend(t_spillover)
            tmpfiles.extend(tmps)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif' + params.tmp_suffix, '.tif'))

    else:

        def arguments():

            for (row, col), seeds in tiles:

                seeds = [coordxy(seed) + values(seed) for seed in seeds]
                yield (
                    ContinuityTile,
                    row,
                    col,
                    seeds,
                    params,
                    kwargs
                )

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments())

            with click.progressbar(pooled, length=ntiles) as iterator:
                for t_spillover, tmps in iterator:
                    g_spillover.extend(t_spillover)
                    tmpfiles.extend(tmps)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif' + params.tmp_suffix, '.tif'))

    return g_spillover

def ContinuityFirstIteration(params, processes, **kwargs):
    """
    First tile iteration with empty seed list.
    """

    tilefile = params.tiles.filename(**kwargs)
    # config.tileset().filename(ax_tiles, **kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    ContinuityTile,
                    row,
                    col,
                    [],
                    params,
                    kwargs
                )

    g_spillover = list()
    tmpfiles = list()

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for t_spillover, tmps in iterator:
                g_spillover.extend(t_spillover)
                tmpfiles.extend(tmps)

        # with click.progressbar(pooled, length=len(arguments)) as bar:
        #     for t_spillover in bar:
        #         g_spillover.extend(t_spillover)

    for tmpfile in tmpfiles:
        os.rename(tmpfile, tmpfile.replace('.tif' + params.tmp_suffix, '.tif'))

    return g_spillover

def ContinuityAnalysisMax(
        params,
        processes=1,
        maxiter=10,
        **kwargs):
    """
    Calculate landcover continuity from river channel
    """

    logger = logging.getLogger(__name__)

    count = 1
    tile = itemgetter(0, 1)
    g_tiles = set()

    click.echo('Iteration %02d --' % count)
    seeds = ContinuityFirstIteration(params, processes=processes, **kwargs)

    while seeds:

        count += 1

        if count > maxiter:
            logger.warning('Stopping after %d iterations', maxiter)
            break

        seeds = [s for s in seeds if tile(s) in config.tileset().tileindex]
        tiles = {tile(s) for s in seeds}
        g_tiles.update(tiles)
        click.echo('Iteration %02d -- %d spillovers, %d tiles' % (count, len(seeds), len(tiles)))

        seeds = ContinuityIteration(params, seeds, len(tiles), processes=processes, **kwargs)
