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

from operator import itemgetter
from collections import namedtuple
import itertools
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

from .. import transform as fct
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

def ContinuityTile(
        axis,
        row,
        col,
        # seeds,
        params,
        **kwargs):
    """
    Tile Implementation
    """

    padding = params.padding
    # tileset = config.tileset(params.tileset)
    tileset = config.tileset()
    landcover_raster = tileset.filename(params.landcover, **kwargs)
    distance_raster = tileset.filename(params.distance, axis=axis, **kwargs)
    height_raster = tileset.filename(params.height, axis=axis, **kwargs)
    output = tileset.tilename(params.output, axis=axis, row=row, col=col, **kwargs)

    height = tileset.height + 2*params.padding
    width = tileset.width + 2*params.padding
    tile_index = tileset.tileindex
    tile = tile_index[row, col]

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
            i, j = ds3.index(tile.x0, tile.y0)
            window3 = Window(j - padding, i - padding, width, height)
            landcover = ds3.read(1, window=window3, boundless=True, fill_value=ds3.nodata)
            transform = ds3.transform * ds3.transform.translation(
                i - padding,
                j - padding)

        if not params.with_infrastructures:
            # Remove infrastructures
            infrastructure_mask = (landcover == 8)
            landcover[infrastructure_mask] = 2

        # coord = itemgetter(0, 1)
        # pixels = fct.worldtopixel(np.array([coord(seed) for seed in seeds], dtype='float32'), transform)
        # intile = np.array([0 <= i < height and 0 <= j < width for i, j in pixels])
        # pixels = pixels[intile]

        # state = np.zeros_like(landcover, dtype='uint8')
        # state[(nearest_height == ds1.nodata) | (nearest_height > params.max_height)] = 255
        # state[pixels[:, 0], pixels[:, 1]] = 1

        state = np.uint8(nearest_distance == 0)
        state[(nearest_height == ds1.nodata) | (nearest_height > params.max_height)] = 255
        del nearest_distance

        # TODO externalize cost function f(landcover)
        cost = np.ones_like(landcover, dtype='float32')
        # cost[landcover == 0] = 0.05
        cost[landcover == 5] = 5.0
        cost[landcover == 6] = 5.0
        cost[landcover == 7] = 10.0
        cost[landcover == 8] = 100.0

        # landcover = np.float32(landcover) + 1
        # landcover[distance == 0] = 0

        # Truncate data outside of valley bottom
        # landcover[(heights == ds1.nodata) | (heights > params.max_height)] = ds3.nodata

        # Shortest max analysis
        out = np.full_like(landcover, ds3.nodata)
        distance = np.zeros_like(landcover, dtype='float32')

        speedup.continuity_analysis(
            landcover,
            nearest_height,
            out,
            distance,
            state,
            cost=cost,
            max_class=params.max_class,
            min_distance=20.0,
            max_distance=0.0,
            max_height=params.max_height,
            jitter=0.4)

        # Reclass stream pixels as water pixels
        # out[landcover == 0] = 1
        # out = np.uint8(out) - 1

        if not params.with_infrastructures:
            # Restore infrastructures
            out[infrastructure_mask & (out != ds3.nodata)] = 8

        # Restore water (landcover = 0) to water (0),
        # if within active channel (out = 1)
        out[(landcover == 0) & (out == 1)] = 0

        # Crop out nodata and padded border
        out[landcover == ds3.nodata] = ds3.nodata
        out = out[padding:-padding, padding:-padding]

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

    # def generate_seeds(feature):

    #     for point in feature['geometry']['coordinates']:
    #         x, y = point[:2]
    #         row, col = config.tileset().index(x, y)
    #         yield (row, col, x, y, 0)

    # network_shapefile = config.filename('ax_talweg', axis=axis)

    # with fiona.open(network_shapefile) as fs:

    #     seeds = [
    #         seed
    #         for feature in fs
    #         for seed in generate_seeds(feature)
    #     ]

    # ContinuityIteration(axis, params, processes, **kwargs)
