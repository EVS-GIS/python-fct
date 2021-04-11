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
import itertools
from operator import itemgetter
from multiprocessing import Pool

import numpy as np
import click

import rasterio as rio
import fiona

from ..config import (
    LiteralParameter,
    DatasetParameter
)
from ..cli import starcall
from .. import transform as fct
from .. import speedup
from ..tileio import (
    PadRaster,
    border
)

from .ValleyBottomFeatures import (
    MASK_EXTERIOR,
    MASK_FLOOPLAIN_RELIEF,
    MASK_VALLEY_BOTTOM,
    MASK_HOLE,
    MASK_SLOPE,
    MASK_TERRACE
)

DOMAIN_EXTERIOR = 255
DOMAIN_INTERIOR = 0
DOMAIN_REFERENCE = 1
DISTANCE_NODATA = -99999.0

class Parameters:
    """
    Connected valley bottom extraction parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='output')
    distance = DatasetParameter(
        'distance to drainage pixels (raster)',
        type='input')
    mask = DatasetParameter(
        'valley bottom features (raster)',
        type='input')
    output_mask = DatasetParameter(
        'connected valley bottom raster',
        type='output')
    output_distance = DatasetParameter(
        'shortest path distance to drainage pixels (temporary processing data)',
        type='output')
    output_final = DatasetParameter(
        'true valley bottom (raster)',
        type='output')

    distance_max = LiteralParameter(
        'maximum exploration distance (from drainage pixels)')
    jitter = LiteralParameter(
        'apply jitter on performing shortest path exploration')
    tmp_suffix = LiteralParameter(
        'temporary files suffix')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.distance = 'nearest_distance'
            self.mask = 'valley_bottom_features'
            self.output_mask = 'valley_bottom_connected'
            self.output_distance = 'valley_bottom_connected_distance'
            self.output_final = 'valley_bottom_final'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.distance = dict(key='ax_nearest_distance', axis=axis)
            self.mask = dict(key='ax_valley_bottom_features', axis=axis)
            self.output_mask = dict(key='ax_valley_bottom_connected', axis=axis)
            self.output_distance = dict(key='ax_valley_bottom_connected_distance', axis=axis)
            self.output_final = dict(key='ax_valley_bottom_final', axis=axis)

        self.distance_max = 0.0
        self.jitter = 0.4
        self.tmp_suffix = '.tmp'

def ValleyBottomConnectedTile(row, col, seeds, params, **kwargs):

    if not params.mask.tilename(row=row, col=col, **kwargs).exists():
        return list(), tuple()

    mask, profile = PadRaster(row, col, params.mask, padding=1, **kwargs)
    transform = profile['transform']
    # nodata = profile['nodata'] # MASK_EXTERIOR
    height, width = mask.shape

    output_mask = str(params.output_mask.tilename(row=row, col=col, **kwargs))
    output_distance = str(params.output_distance.tilename(row=row, col=col, **kwargs))

    if os.path.exists(output_mask):

        domain, _ = PadRaster(row, col, params.output_mask, padding=1, **kwargs)
        distance, _ = PadRaster(row, col, params.output_distance, padding=1, **kwargs)

    else:

        drainage_distance, _ = PadRaster(row, col, params.distance, padding=1, **kwargs)

        domain = np.full_like(mask, DOMAIN_EXTERIOR, dtype='uint8')
        domain[(mask == MASK_VALLEY_BOTTOM) | (mask == MASK_FLOOPLAIN_RELIEF)] = DOMAIN_INTERIOR
        domain[(mask != MASK_EXTERIOR) & (np.abs(drainage_distance) < 1)] = DOMAIN_REFERENCE

        distance = np.zeros_like(mask, dtype='float32')

    # apply seed values

    if seeds:

        coord = itemgetter(0, 1)
        pixels = fct.worldtopixel(np.array([coord(seed) for seed in seeds], dtype='float32'), transform)
        intile = np.array([0 <= i < height and 0 <= j < width for i, j in pixels])
        seed_distance = np.array([seed[2] for seed in seeds], dtype='float32')

        pixels = pixels[intile]
        seed_distance = seed_distance[intile]

        recorded_distance = distance[pixels[:, 0], pixels[:, 1]]
        shortest = (recorded_distance == DISTANCE_NODATA) | (seed_distance < recorded_distance)
        pixels = pixels[shortest]

        domain[pixels[:, 0], pixels[:, 1]] = DOMAIN_REFERENCE
        distance[pixels[:, 0], pixels[:, 1]] = seed_distance[shortest]

    # calculate connected subdomain

    connected = speedup.shortest_value(
        domain,
        np.copy(domain),
        DOMAIN_EXTERIOR,
        DOMAIN_REFERENCE,
        distance,
        params.distance_max,
        params.jitter)

    distance[connected == DOMAIN_EXTERIOR] = DISTANCE_NODATA

    # extract spillovers

    spillovers = [
        (i, j) for i, j in border(height, width)
        if domain[i, j] == DOMAIN_INTERIOR and connected[i, j] == DOMAIN_REFERENCE
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

            return (prow, pcol) + tuple(xy[k]) + (distance[i, j],)

        spillovers = [
            attributes(k, ij)
            for k, ij in enumerate(spillovers)
        ]

    # write output

    connected = connected[1:-1, 1:-1]
    distance = distance[1:-1, 1:-1]

    output_mask += params.tmp_suffix
    output_distance += params.tmp_suffix

    height, width = connected.shape
    transform = transform * transform.translation(1, 1)
    profile.update(
        dtype='uint8',
        nodata=DOMAIN_EXTERIOR,
        height=height,
        width=width,
        transform=transform,
        compress='deflate')

    with rio.open(output_mask, 'w', **profile) as dst:
        dst.write(connected, 1)

    profile.update(dtype='float32', nodata=DISTANCE_NODATA)

    with rio.open(output_distance, 'w', **profile) as dst:
        dst.write(distance, 1)

    # return spillovers and destination filenames

    return spillovers, (output_mask, output_distance)

def ValleyBottomConnectedIteration(params, spillovers, ntiles, processes=1, **kwargs):
    """
    Multiprocessing wrapper for ValleyBottomConnectedTile
    """

    tile = itemgetter(0, 1)
    values = itemgetter(2, 3, 4)

    spillovers = sorted(spillovers, key=tile)
    tiles = itertools.groupby(spillovers, key=tile)

    g_spillover = list()
    tmpfiles = list()

    def arguments():

        for (row, col), seeds in tiles:

            seeds = [values(seed) for seed in seeds]

            yield (
                ValleyBottomConnectedTile,
                row,
                col,
                seeds,
                params,
                kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        with click.progressbar(pooled, length=ntiles) as iterator:
            for t_spillover, tmps in iterator:
                g_spillover.extend(t_spillover)
                tmpfiles.extend(tmps)

    for tmpfile in tmpfiles:
        os.rename(tmpfile, tmpfile.replace('.tif' + params.tmp_suffix, '.tif'))

    return g_spillover

def ValleyBottomConnectedFirstIteration(params, processes, **kwargs):
    """
    First tile iteration with empty seed list.
    """

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    ValleyBottomConnectedTile,
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

def ValleyBottomFinalTile(row, col, params, **kwargs):

    mask_raster = params.mask.tilename(row=row, col=col, **kwargs)
    connected_raster = params.output_mask.tilename(row=row, col=col, **kwargs)
    output = params.output_final.tilename(row=row, col=col, **kwargs)

    if not mask_raster.exists():
        return

    with rio.open(connected_raster) as ds:
        connected = ds.read(1)

    # TODO detect holes from border
    # domain = DOMAIN_EXTERIOR
    # reference = border cells with DOMAIN_EXTERIOR
    # not reached DOMAIN_EXTERIOR cells -> holes

    with rio.open(mask_raster) as ds:

        mask = ds.read(1)

        mask[
            (connected == DOMAIN_REFERENCE) &
            ((mask == MASK_TERRACE) | (mask == MASK_SLOPE))
        ] = MASK_FLOOPLAIN_RELIEF

        mask[
            (connected != DOMAIN_REFERENCE) &
            (mask == MASK_VALLEY_BOTTOM)
        ] = MASK_TERRACE

        mask[
            (connected != DOMAIN_REFERENCE) &
            (mask == MASK_FLOOPLAIN_RELIEF)
        ] = MASK_SLOPE

        margins = np.copy(mask)
        margins[
            (mask == MASK_SLOPE) |
            (mask == MASK_TERRACE)
        ] = MASK_HOLE

        margins[:, 0] = MASK_EXTERIOR
        margins[:, -1] = MASK_EXTERIOR
        margins[0, :] = MASK_EXTERIOR
        margins[-1, :] = MASK_EXTERIOR
        
        speedup.reclass_margin(margins, MASK_HOLE, MASK_EXTERIOR, MASK_EXTERIOR)
        mask[margins == MASK_HOLE] = MASK_HOLE

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(mask, 1)

def ConnectedValleyBottom(params, processes=1, **kwargs):
    """
    Extract true valley bottom,
    excluding terraces, slopes and flat areas not connected to drainage network
    """

    count = 1
    tile = itemgetter(0, 1)
    g_tiles = set()

    click.echo('Iteration %02d --' % count)
    seeds = ValleyBottomConnectedFirstIteration(params, processes=processes, **kwargs)

    while seeds:

        count += 1
        tiles = {tile(s) for s in seeds}
        g_tiles.update(tiles)
        click.echo('Iteration %02d -- %d spillovers, %d tiles' % (count, len(seeds), len(tiles)))

        seeds = ValleyBottomConnectedIteration(params, seeds, len(tiles), processes, **kwargs)

def TrueValleyBottom(params: Parameters, processes: int = 1, **kwargs):

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    ValleyBottomFinalTile,
                    row,
                    col,
                    params,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass
