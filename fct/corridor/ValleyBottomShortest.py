# coding: utf-8

"""
Valley bottom extraction procedure - shortest path exploration

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
from collections import namedtuple
import itertools
from operator import itemgetter
from multiprocessing import Pool

import numpy as np
import click

import rasterio as rio
import fiona

from ..config import config
from ..cli import starcall
from .. import transform as fct
from .. import speedup
from ..tileio import (
    PadRaster,
    border
)

ShortestParams = namedtuple(
    'ShortestParams', (
        'dataset_height',
        'dataset_distance',
        'max_dz',
        'min_distance',
        'max_distance',
        'jitter',
        'tmp'
    )
)

def ValleyBottomTile(axis, row, col, seeds, params):
    """
    Valley bottom shortest path exploration
    """

    elevations, profile = PadRaster(row, col, 'dem', padding=1)
    transform = profile['transform']
    nodata = profile['nodata']
    height, width = elevations.shape

    output_height = config.tileset().tilename('ax_shortest_height', axis=axis, row=row, col=col)
    output_distance = config.tileset().tilename('ax_shortest_distance', axis=axis, row=row, col=col)
    output_state = config.tileset().tilename('ax_shortest_state', axis=axis, row=row, col=col)

    if os.path.exists(output_height):

        heights, _ = PadRaster(row, col, 'ax_shortest_height', axis=axis, padding=1)
        distance, _ = PadRaster(row, col, 'ax_shortest_distance', axis=axis, padding=1)
        state, _ = PadRaster(row, col, 'ax_shortest_state', axis=axis, padding=1)

    else:

        heights = np.full((height, width), nodata, dtype='float32')
        distance = np.full((height, width), nodata, dtype='float32')
        state = np.zeros((height, width), dtype='uint8')
        state[elevations == nodata] = 255

    coord = itemgetter(0, 1)
    pixels = fct.worldtopixel(np.array([coord(seed) for seed in seeds], dtype='float32'), transform)
    intile = np.array([0 <= i < height and 0 <= j < width for i, j in pixels])

    seed_heights = np.array([seed[2] for seed in seeds], dtype='float32')
    seed_distance = np.array([seed[3] for seed in seeds], dtype='float32')

    pixels = pixels[intile]
    seed_heights = seed_heights[intile]
    seed_distance = seed_distance[intile]

    recorded_distance = distance[pixels[:, 0], pixels[:, 1]]
    shortest = (recorded_distance == nodata) | (seed_distance < recorded_distance)

    pixels = pixels[shortest]
    heights[pixels[:, 0], pixels[:, 1]] = seed_heights[shortest]
    distance[pixels[:, 0], pixels[:, 1]] = seed_distance[shortest]

    state[pixels[:, 0], pixels[:, 1]] = 1
    control = np.copy(state)

    reference = elevations - heights
    del heights

    speedup.valley_bottom_shortest(
        elevations,
        state,
        reference,
        distance,
        max_dz=params.max_dz,
        min_distance=params.min_distance,
        max_distance=params.max_distance,
        jitter=params.jitter)

    spillovers = [
        (i, j) for i, j in border(height, width)
        if control[i, j] != 2 and state[i, j] == 2
    ]

    del control

    # restore unresolved cells' state
    state[state == 1] = 2

    heights = elevations - reference
    heights[(state == 0) | (state == 255)] = nodata
    distance[(state == 0) | (state == 255)] = nodata

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

            return (prow, pcol) + tuple(xy[k]) + (heights[i, j], distance[i, j])

        spillovers = [
            attributes(k, ij)
            for k, ij in enumerate(spillovers)
        ]

    del elevations
    del reference

    heights = heights[1:-1, 1:-1]
    distance = distance[1:-1, 1:-1]
    height, width = heights.shape

    transform = transform * transform.translation(1, 1)
    profile.update(
        dtype='float32',
        height=height,
        width=width,
        transform=transform,
        compress='deflate')

    output_height += params.tmp
    output_distance += params.tmp
    output_state += params.tmp

    with rio.open(output_height, 'w', **profile) as dst:
        dst.write(heights, 1)

    with rio.open(output_distance, 'w', **profile) as dst:
        dst.write(distance, 1)

    profile.update(dtype='uint8', nodata=255)

    with rio.open(output_state, 'w', **profile) as dst:
        dst.write(state, 1)

    return spillovers, (output_height, output_distance, output_state)

def ValleyBottomIteration(axis, params, spillovers, ntiles, processes=1, **kwargs):
    """
    Multiprocessing wrapper for ValleyBottomTile
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
            t_spillover, tmps = ValleyBottomTile(axis, row, col, seeds, params)
            g_spillover.extend(t_spillover)
            tmpfiles.extend(tmps)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif' + params.tmp, '.tif'))

    else:

        def arguments():

            for (row, col), seeds in tiles:

                seeds = [coordxy(seed) + values(seed) for seed in seeds]
                yield (ValleyBottomTile, axis, row, col, seeds, params, kwargs)

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments())
            with click.progressbar(pooled, length=ntiles) as iterator:
                for t_spillover, tmps in iterator:
                    g_spillover.extend(t_spillover)
                    tmpfiles.extend(tmps)

            # with click.progressbar(pooled, length=len(arguments)) as bar:
            #     for t_spillover in bar:
            #         g_spillover.extend(t_spillover)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif' + params.tmp, '.tif'))

    return g_spillover

def ValleyBottomDefaultParameters():
    """
    Default parameters
    """

    return dict(
        dataset_height='ax_shortest_height',
        dataset_distance='ax_shortest_distance',
        max_dz=20.0,
        min_distance=20,
        max_distance=5000,
        jitter=0.4,
        tmp='.tmp'
    )

def ValleyBottom(axis, processes=1, **kwargs):
    """
    Valley bottom extraction procedure - shortest path exploration
    """

    parameters = ValleyBottomDefaultParameters()

    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = ShortestParams(**parameters)

    def generate_seeds(feature):

        for point in feature['geometry']['coordinates']:
            x, y = point[:2]
            row, col = config.tileset().index(x, y)
            yield (row, col, x, y, 0.0, 0.0)

    network_shapefile = config.filename('ax_talweg', axis=axis)

    with fiona.open(network_shapefile) as fs:

        seeds = [
            seed
            for feature in fs
            for seed in generate_seeds(feature)
        ]

    count = 0
    tile = itemgetter(0, 1)
    g_tiles = set()

    while seeds:

        count += 1
        tiles = {tile(s) for s in seeds}
        g_tiles.update(tiles)
        click.echo('Iteration %02d -- %d spillovers, %d tiles' % (count, len(seeds), len(tiles)))

        seeds = ValleyBottomIteration(axis, params, seeds, len(tiles), processes)

    tiles = {tile(s) for s in seeds}
    g_tiles.update(tiles)

    click.secho('Ok', fg='green')

    output = config.tileset().filename('ax_shortest_tiles', axis=axis)

    with open(output, 'w') as fp:
        for row, col in sorted(g_tiles):
            fp.write('%d,%d\n' % (row, col))
