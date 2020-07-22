#!/usr/bin/env python
# coding: utf-8

"""
Raster buffer around stream active channel

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio

from .. import speedup
from ..tileio import (
    as_window,
    grow_window
)
from ..config import config
from ..cli import starcall

def BufferDistanceTile(
        axis,
        tile,
        buffer_width,
        padding=200,
        fill=2,
        tileset='default',
        dataset='ax_continuity',
        resolution=5.0):
    """
    Calculate a buffer area and distance from river active channel
    from landcover continuity raster.

    Output will have :
      1 = active channel
      2 (fill) = buffer area,
                 ie. space within buffer width from active channel
      0 = elsewhere (nodata)

    The calculated buffer is not limited to landcover data
    and may extend outside landcover data
    """

    rasterfile = config.filename(dataset, axis=axis)

    output_mask = config.tileset(tileset).tilename(
        'ax_buffer_mask',
        axis=axis,
        row=tile.row,
        col=tile.col)

    output_distance = config.tileset(tileset).tilename(
        'ax_buffer_distance',
        axis=axis,
        row=tile.row,
        col=tile.col)

    with rio.open(rasterfile) as ds:

        window = as_window(tile.bounds, ds.transform)
        landcover = ds.read(
            1,
            window=grow_window(window, padding),
            boundless=True,
            fill_value=ds.nodata)

        transform = ds.transform * ds.transform.translation(
            window.col_off,
            window.row_off)

        # Define a mask from active channel landcover classes
        mask = np.float32((landcover == 0) | (landcover == 1))

        distance = resolution * speedup.raster_buffer(
            mask,
            0.0,
            buffer_width / resolution,
            fill)

        mask = np.uint8(mask)
        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            dtype='uint8',
            nodata=0,
            compress='deflate',
            height=window.height,
            width=window.width,
            transform=transform
        )

        with rio.open(output_mask, 'w', **profile) as dst:
            dst.write(mask[padding:-padding, padding:-padding], 1)

        distance_nodata = -99999.0
        distance[mask == 0] = distance_nodata

        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            dtype='float32',
            nodata=distance_nodata,
            compress='deflate',
            height=window.height,
            width=window.width,
            transform=transform
        )

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance[padding:-padding, padding:-padding], 1)

def BufferDistance(axis, buffer_width, processes=1, **kwargs):
    """
    Calculate a buffer area and distance from river active channel
    from landcover continuity raster.
    """

    tileindex = config.tileset('landcover').tileindex
    tilefile = config.filename('ax_tiles', axis=axis)
    resolution = 5.0

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    def arguments():

        for _, row, col in tiles:
            # if (row, col) in tileindex:
            tile = tileindex[row, col]
            yield (
                BufferDistanceTile,
                axis,
                tile,
                buffer_width,
                kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tiles)) as iterator:
            for _ in iterator:
                pass
