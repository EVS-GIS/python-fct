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

from .. import terrain_analysis as ta
from ..cli import starcall
from ..config import config

def LateralContinuityTile(
        axis,
        row,
        col,
        dataset='landcover-bdt',
        maxz=20.0,
        padding=200,
        with_infrastructures=True):

    tileset = config.tileset('landcover')
    landcover_raster = config.filename(dataset)
    distance_raster = config.filename('ax_nearest_distance', axis=axis)
    hand_raster = config.filename('ax_relative_elevation', axis=axis)
    output = tileset.tilename('ax_continuity', axis=axis, row=row, col=col)

    height = tileset.height + 2*padding
    width = tileset.width + 2*padding
    tile_index = tileset.tileindex
    tile = tile_index[row, col]

    with rio.open(hand_raster) as ds1:

        i0, j0 = ds1.index(tile.x0, tile.y0)
        window1 = Window(j0 - padding, i0 - padding, width, height)
        hand = ds1.read(1, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(distance_raster) as ds2:

            i, j = ds2.index(tile.x0, tile.y0)
            window2 = Window(j - padding, i - padding, width, height)
            distance = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)

        with rio.open(landcover_raster) as ds3:

            profile = ds3.profile.copy()

            i, j = ds3.index(tile.x0, tile.y0)
            window3 = Window(j - padding, i - padding, width, height)
            landcover = ds3.read(1, window=window3, boundless=True, fill_value=ds3.nodata)

        if not with_infrastructures:
            # Remove infrastructures
            infrastructure_mask = (landcover == 8)
            landcover[infrastructure_mask] = 2

        cost = np.ones_like(landcover, dtype='float32')
        # cost[landcover == 0] = 0.05
        # cost[landcover <= 5] = 1.0
        cost[landcover >= 6] = 10.0
        cost[landcover >= 7] = 100.0
        cost[landcover >= 8] = 1.0

        landcover = np.float32(landcover) + 1
        landcover[distance == 0] = 0

        # Truncate data outside of valley bottom
        landcover[(hand == ds1.nodata) | (hand > maxz)] = ds3.nodata

        # Shortest max analysis
        out = np.zeros_like(landcover)
        distance = np.zeros_like(landcover)
        ta.shortest_max(landcover, ds3.nodata, 0, cost, out, distance)

        # Reclass stream pixels as water pixels
        out[landcover == 0] = 1
        out = np.uint8(out) - 1

        if not with_infrastructures:
            # Restore infrastructures
            out[infrastructure_mask] = 8

        # Restore water (landcover = 0+1) to water (0),
        # if within active channel (out = 1)
        out[(landcover == 1) & (out == 1)] = 0

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

def LateralContinuity(axis, processes=1, **kwargs):
    """
    Calculate LandCover Continuity from River Channel
    """

    tileset = config.tileset('landcover')

    arguments = list()

    for tile in tileset.tileindex.values():
        arguments.append((LateralContinuityTile, axis, tile.row, tile.col, kwargs))

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
