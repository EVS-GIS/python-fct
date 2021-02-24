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

from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

# from .. import terrain_analysis as ta
from .. import speedup
from ..cli import starcall
from ..config import (
    config,
    DatasetParameter,
    LiteralParameter
)

class Parameters:
    """
    Continuity analysis parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    landcover = DatasetParameter('landcover raster map', type='input')
    distance = DatasetParameter('distance to talweg', type='input')
    height = DatasetParameter('height above talweg', type='input')

    output = DatasetParameter('continuity map', type='output')
    output_distance = DatasetParameter('distance to reference pixel', type='output')
    state = DatasetParameter('processing state raster', type='output')

    tileset = LiteralParameter('tileset')
    class_max = LiteralParameter('maximum landcover class (stop criterion)')
    height_max = LiteralParameter('maximum height above reference (stop criterion)')
    distance_min = LiteralParameter('minimum distance')
    distance_max = LiteralParameter('maximum distance')
    padding = LiteralParameter('tile padding in pixels')
    infrastructures = LiteralParameter('consider transport infrastructures (landcover class = 8)')
    jitter = LiteralParameter('apply jitter on performing shortest path raster exploration')
    tmp_suffix = LiteralParameter('temporary files suffix')

    def __init__(self):
        """
        Default parameter values
        """

        self.tileset = 'default'
        self.tiles = 'shortest_tiles'
        self.landcover = 'landcover_valley_bottom'
        self.distance = 'nearest_distance'
        self.height = 'nearest_height'
        self.output = 'continuity'
        self.output_distance = 'continuity_distance'
        self.state = 'continuity_state'
        self.class_max = 0
        self.height_max = 20.0
        self.distance_min = 100.0
        self.distance_max = 0
        self.padding = 200
        self.infrastructures = True
        self.jitter = 0.4
        self.tmp_suffix = '.tmp'

def WeightedContinuityAnalysisTile(
        row,
        col,
        params,
        **kwargs):
    """
    Tile Implementation
    """

    tileset = config.tileset(params.tileset)
    landcover_raster = params.landcover.filename(**kwargs)
    nearest_distance_raster = params.distance.filename(**kwargs)
    hand_raster = params.height.filename(**kwargs)
    output = params.output.tilename(row=row, col=col, **kwargs)

    padding = params.padding
    height = tileset.height + 2*padding
    width = tileset.width + 2*padding
    tile_index = tileset.tileindex
    tile = tile_index[row, col]

    with rio.open(hand_raster) as ds1:

        i0, j0 = ds1.index(tile.x0, tile.y0)
        window1 = Window(j0 - padding, i0 - padding, width, height)
        hand = ds1.read(1, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(nearest_distance_raster) as ds2:

            i, j = ds2.index(tile.x0, tile.y0)
            window2 = Window(j - padding, i - padding, width, height)
            nearest_distance = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)

        with rio.open(landcover_raster) as ds3:

            profile = ds3.profile.copy()

            i, j = ds3.index(tile.x0, tile.y0)
            window3 = Window(j - padding, i - padding, width, height)
            landcover = ds3.read(1, window=window3, boundless=True, fill_value=ds3.nodata)

        if not params.infrastructures:
            # Remove infrastructures
            infrastructure_mask = (landcover == 8)
            landcover[infrastructure_mask] = 2

        cost = np.ones_like(landcover, dtype='float32')
        # cost[landcover == 0] = 0.05
        # cost[landcover <= 5] = 1.0
        cost[landcover >= 6] = 10.0
        cost[landcover >= 7] = 100.0
        cost[landcover >= 8] = 1.0

        # landcover = np.float32(landcover) + 1
        # landcover[distance == 0] = 0

        # Truncate data outside of valley bottom
        # landcover[(hand == ds1.nodata) | (hand > params.height_max)] = ds3.nodata

        state = np.uint8(np.abs(nearest_distance) < 1)
        del nearest_distance

        state[
            (hand == ds1.nodata) |
            (hand > params.height_max)
        ] = 255

        # Shortest max analysis
        out = np.full_like(landcover, ds3.nodata)
        distance = np.zeros_like(landcover, dtype='float32')

        # ta.shortest_max(landcover, ds3.nodata, 0, cost, out, distance)

        speedup.continuity_analysis(
            landcover,
            hand,
            out,
            distance,
            state,
            cost=cost,
            max_class=params.class_max,
            min_distance=params.distance_min,
            max_distance=params.distance_max,
            max_height=params.height_max,
            jitter=params.jitter)

        # Reclass stream pixels as water pixels
        # out[landcover == 0] = 1
        # out = np.uint8(out) - 1

        if not params.infrastructures:
            # Restore infrastructures
            out[infrastructure_mask] = 8

        # Restore water (landcover = 0) to water (0),
        # if within active channel (out = 1)
        out[(landcover == 0) & (out == 1)] = 0

        # Crop out nodata
        out[landcover == ds3.nodata] = ds3.nodata

        height = height - 2*padding
        width = width - 2*padding
        transform = ds1.transform * ds1.transform.translation(j0, i0)

        profile.update(
            driver='GTiff',
            height=height,
            width=width,
            transform=transform,
            compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out[padding:-padding, padding:-padding], 1)

def WeightedContinuityAnalysis(
        params,
        processes=1,
        **kwargs):
    """
    Calculate landcover continuity from river channel

    @api    fct-corridor:continuity-weighted

    @input  landcover: landcover-bdt
    @input  distance: ax_talweg_distance
    @input  heigth: ax_nearest_height

    @param  with_infra: True
    @param  max_height: 20.0
    @param  padding: 200

    @output continuity_map_weighted: ax_continuity

    Parameters
    ----------

    axis: int

        Axis identifier

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword Parameters
    ------------------

    tileset: str, logical name

        Tileset to use,
        defaults to `landcover`

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

    maxz: float

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

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    WeightedContinuityAnalysisTile,
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
