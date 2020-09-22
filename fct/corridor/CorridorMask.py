# coding: utf-8

"""
Spatial Referencing :
Measure along Reference Axis, Space Discretization

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
from multiprocessing import Pool

import numpy as np
import click
import rasterio as rio

from ..config import config
from .. import speedup
from ..cli import starcall

def CorridorMaskTile(axis, row, col, buffer_width, **kwargs):

    tileset = config.tileset()

    def _tilename(dataset):
        return tileset.tilename(
            dataset,
            axis=axis,
            row=row,
            col=col)

    swaths_raster = _tilename('ax_valley_swaths')
    data_raster = _tilename('ax_continuity')
    output = _tilename('ax_corridor_mask')

    def create_mask(data):

        mask = np.float32(data <= 5)
        mask[data == ds.nodata] = 0

        return mask

    if not os.path.exists(swaths_raster):
        return

    with rio.open(swaths_raster) as ds:

        swaths = ds.read(1)
        swath_nodata = ds.nodata

    with rio.open(data_raster) as ds:

        # click.echo('Read Valley Bottom')

        # valley_bottom = speedup.raster_buffer(ds.read(1), ds.nodata, 6.0)
        data = ds.read(1)
        mask = create_mask(data)
        speedup.raster_buffer(mask, 0, buffer_width, 1)

        data[(mask == 0) | (swaths == swath_nodata)] = ds.nodata

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(data, 1)

def CorridorMask(axis, buffer_width, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):
    """
    Creates a raster buffer with distance buffer_width pixels
    around data pixels and crop out data outside of the resulting buffer
    """

    tilefile = config.tileset().filename(ax_tiles, axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                CorridorMaskTile,
                axis,
                row,
                col,
                buffer_width,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

# def HANDBufferTile(axis, row, col, buffer_width, **kwargs):

#     tileset = config.tileset()

#     def _tilename(dataset):
#         return tileset.tilename(
#             dataset,
#             axis=axis,
#             row=row,
#             col=col)

#     data_raster = _tilename('ax_nearest_height')
#     output = _tilename('ax_valley_mask')

#     def create_mask(data):

#         mask = np.float32(data <= 12.0)
#         mask[data == ds.nodata] = 0

#         return mask

#     with rio.open(data_raster) as ds:

#         # click.echo('Read Valley Bottom')

#         # valley_bottom = speedup.raster_buffer(ds.read(1), ds.nodata, 6.0)
#         data = ds.read(1)
#         mask = create_mask(data)
#         speedup.raster_buffer(mask, 0, buffer_width, 1)

#         data[mask == 0] = ds.nodata

#         profile = ds.profile.copy()
#         profile.update(compress='deflate')

#         with rio.open(output, 'w', **profile) as dst:
#             dst.write(data, 1)

# def HANDBuffer(axis, buffer_width, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):
#     """
#     Creates a raster buffer with distance buffer_width pixels
#     around data pixels and crop out data outside of the resulting buffer
#     """

#     tilefile = config.tileset().filename(ax_tiles, axis=axis)

#     def length():

#         with open(tilefile) as fp:
#             return sum(1 for line in fp)

#     def arguments():

#         with open(tilefile) as fp:
#             tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

#         for row, col in tiles:
#             yield (
#                 HANDBufferTile,
#                 axis,
#                 row,
#                 col,
#                 buffer_width,
#                 kwargs
#             )

#     with Pool(processes=processes) as pool:

#         pooled = pool.imap_unordered(starcall, arguments())

#         with click.progressbar(pooled, length=length()) as iterator:
#             for _ in iterator:
#                 pass
