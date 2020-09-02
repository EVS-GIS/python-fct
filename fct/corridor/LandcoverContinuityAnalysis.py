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

from collections import namedtuple
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

from .. import speedup
from ..cli import starcall
from ..config import config

ContinuityParams = namedtuple('ContinuityParams', [
    # 'tileset',
    'landcover',
    'distance',
    'height',
    'output',
    'max_height',
    'max_class',
    'padding',
    'with_infrastructures'
])

def costf_default(landcover):
    """
    Default landcover cost function
    """

    cost = np.ones_like(landcover, dtype='float32')
    # cost[landcover == 0] = 0.05
    cost[landcover == 5] = 5.0
    cost[landcover == 6] = 5.0
    cost[landcover == 7] = 10.0
    cost[landcover == 8] = 100.0

    return cost

def ContinuityTile(axis, row, col, params, mode='init', **kwargs):
    """
    Tile Implementation
    """

    tileset = config.tileset()

    landcover_raster = tileset.filename(params.landcover, **kwargs)

    distance_raster = tileset.filename(params.distance, axis=axis, **kwargs)

    height_raster = tileset.filename(params.height, axis=axis, **kwargs)

    output = tileset.tilename(
        params.output,
        axis=axis,
        row=row,
        col=col,
        **kwargs)

    output_state = tileset.tilename(
        'ax_continuity_state',
        axis=axis,
        row=row,
        col=col,
        **kwargs)

    output_distance = tileset.tilename(
        'ax_continuity_distance',
        axis=axis,
        row=row,
        col=col,
        **kwargs)

    padding = params.padding
    height = tileset.height + 2*padding
    width = tileset.width + 2*padding
    tile = tileset.tileindex[row, col]

    with rio.open(height_raster) as ds1:

        i0, j0 = ds1.index(tile.x0, tile.y0)
        window1 = Window(j0 - padding, i0 - padding, width, height)
        nearest_height = ds1.read(1, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(distance_raster) as ds2:

            i, j = ds2.index(tile.x0, tile.y0)
            window2 = Window(j - padding, i - padding, width, height)
            nearest_distance = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)

        with rio.open(landcover_raster) as ds3:

            profile = ds3.profile.copy()
            nodata = ds3.nodata

            i, j = ds3.index(tile.x0, tile.y0)
            window3 = Window(j - padding, i - padding, width, height)
            landcover = ds3.read(1, window=window3, boundless=True, fill_value=nodata)

            transform = ds3.transform * ds3.transform.translation(
                i - padding,
                j - padding)

        if not params.with_infrastructures:
            # Remove infrastructures
            infrastructure_mask = (landcover == 8)
            landcover[infrastructure_mask] = 2

        out = np.array([])
        distance = np.array([])
        state = np.array([])

        def init_state():
            """
            Initialize state array with respect to processing mode

            - init mode: initialize from stream cells,
                         ie. cells having nearest_distance = 0

            - reiterate mode: lookup for resolved boundary cells,
                              ie. cells with previous state = 2 neighbouring nodata cells
            """

            nonlocal out
            nonlocal distance
            nonlocal state

            if mode == 'reiterate':

                state_raster = tileset.filename('ax_continuity_state', axis=axis, **kwargs)
                continuity_raster = tileset.filename(params.output, axis=axis, **kwargs)
                distance_raster = tileset.filename('ax_continuity_distance', axis=axis, **kwargs)

                with rio.open(state_raster) as ds4:

                    i, j = ds4.index(tile.x0, tile.y0)
                    window4 = Window(j - padding, i - padding, width, height)
                    state = ds4.read(1, window=window4, boundless=True, fill_value=ds4.nodata)

                with rio.open(continuity_raster) as ds4:

                    out = ds4.read(1, window=window4, boundless=True, fill_value=ds4.nodata)

                with rio.open(distance_raster) as ds4:

                    distance = ds4.read(1, window=window4, boundless=True, fill_value=ds4.nodata)

                count = speedup.continuity_analysis_restate(state, nearest_height, params.max_height)
                state[(nearest_height == ds1.nodata) | (nearest_height > params.max_height)] = 255

            else:

                state = np.uint8(nearest_distance == 0)
                state[(nearest_height == ds1.nodata) | (nearest_height > params.max_height)] = 255
                out = np.full_like(landcover, nodata)
                distance = np.zeros_like(landcover, dtype='float32')
                count = np.sum(state == 1)

            return count

        if init_state() > 0:

            # Continuity analysis on shortest path

            del nearest_distance
            cost = costf_default(landcover)

            for max_class, max_height in [
                    (3, 10.0),
                    (4, 15.0),
                    (5, 15.0),
                    (0, params.max_height)
                ]:

                state[state == 5] = 1

                # if max_class > params.max_class:
                #     break

                speedup.continuity_analysis(
                    landcover,
                    nearest_height,
                    out,
                    distance,
                    state,
                    cost=cost,
                    max_class=max_class,
                    min_distance=20.0,
                    max_distance=0.0,
                    max_height=max_height,
                    jitter=0.4)

            if not params.with_infrastructures:
                # Restore infrastructures
                out[infrastructure_mask & (out != nodata)] = 8

            # Restore water (landcover = 0) to water (0),
            # if within active channel (out = 1)

            out[(landcover == 0) & (out == 1)] = 0

        # Crop out nodata and padded border

        out[landcover == nodata] = nodata
        out = out[padding:-padding, padding:-padding]
        state = state[padding:-padding, padding:-padding]
        distance = distance[padding:-padding, padding:-padding]

        # Output

        height, width = out.shape
        transform = ds1.transform * ds1.transform.translation(j0, i0)
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

        profile.update(dtype='float32', nodata=ds2.nodata)

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

def LandcoverContinuityAnalysis(
        axis,
        processes=1,
        ax_tiles='ax_shortest_tiles',
        # tileset='landcover',
        landcover='landcover-bdt',
        distance='ax_nearest_distance',
        height='ax_nearest_height',
        output='ax_continuity',
        padding=200,
        max_height=20.0,
        max_class=0,
        with_infrastructures=True,
        **kwargs):
    """
    Calculate LandCover Continuity from River Channel

    Parameters
    ----------

    axis: int

        Axis identifier

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword Parameters
    ------------------

    # tileset: str, logical name

    #     Tileset to use,
    #     defaults to `landcover`

    landcover: str, logical name

        Landcover data raster,
        defaults to `landcover`

    distance: str, logical name

        Distance from drainage raster,
        defaults to `ax_talweg_distance`.

    height: str, logical name

        Height above drainage raster,
        defaults to `ax_nearest_height`

    output: str, logical name

        Continuity raster output,
        defaults to `ax_continuity`

    max_height: float

        Truncate landcover data with height above maxz,
        defaults to 20.0 m

    padding: int

        Number of pixels to pad tiles with,
        defaults to 200

    with_infrastructures: bool

        Whether to exclude landcover infrastructure class (8)
        from continuity analysis,
        defaults to True

    Other keywords are passed to dataset filename templates.
    """

    params = ContinuityParams(
        # tileset=tileset,
        landcover=landcover,
        distance=distance,
        height=height,
        output=output,
        padding=padding,
        max_height=max_height,
        max_class=max_class,
        with_infrastructures=with_infrastructures
    )

    # tile = itemgetter(0, 1)
    # xyvalue = itemgetter(2, 3, 4)

    # spillovers = sorted(seeds, key=tile)
    # tiles = itertools.groupby(spillovers, key=tile)

    # def arguments():

        # for (row, col), seeds in tiles:

        #     seeds = [xyvalue(seed) for seed in seeds]

        #     yield (
        #         ContinuityTile,
        #         axis,
        #         row,
        #         col,
        #         seeds,
        #         params,
        #         kwargs
        #     )

    tilefile = config.tileset().filename(ax_tiles, axis=axis, **kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    ContinuityTile,
                    axis,
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

def test(axis):

    # pylint:disable=import-outside-toplevel
    from ..tileio import buildvrt

    config.default()

    LandcoverContinuityAnalysis(axis=axis, processes=6)
    buildvrt('default', 'ax_continuity', axis=axis)
    buildvrt('default', 'ax_continuity_state', axis=axis)
    buildvrt('default', 'ax_continuity_distance', axis=axis)
    click.pause()
    LandcoverContinuityAnalysis(axis=axis, processes=6, mode='reiterate')
