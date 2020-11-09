# coding: utf-8

"""
Create Valley Bottom Mask from Height Raster

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

from ..config import config
from .. import speedup
from ..cli import starcall

ValleyBottomMaskParams = namedtuple('ValleyBottomMaskParams', [
    'height',
    'distance',
    'output',
    'buffer_width',
    'dist_resolution',
    'max_slope',
    'min_distance',
    'max_height'
])

# def ClipHeightTile(axis, row, col, params, **kwargs):

#     tileset = config.tileset()
#     height_raster = tileset.tilename(params.height, axis=axis, row=row, col=col, **kwargs)
#     distance_raster = tileset.tilename(params.distance, axis=axis, row=row, col=col, **kwargs)

#     with rio.open(height_raster) as ds:
#         hand = ds.read(1)
#         profile = ds.profile.copy()
#         nodata = ds.nodata

#     with rio.open(distance_raster) as ds:
#         distance = ds.read(1)

#     hand[
#         (hand < -params.max_slope * params.dist_resolution * distance)
#         & (params.dist_resolution * distance > params.min_distance)
#         | (hand > params.max_height)
#     ] = nodata

#     profile.update(compress='deflate')

#     with rio.open(height_raster, 'w', **profile) as dst:
#         dst.write(hand, 1)

def ValleyBottomMaskDefaultParameters():

    return dict(
        height='nearest_height',
        distance='nearest_distance',
        output='valley_mask',
        dist_resolution=1.0,
        max_slope=0.01,
        min_distance=1000.0,
        max_height=12.0,
        buffer_width=20.0
    )

# def ClipHeight(axis, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):

#     parameters = ClipDefaultParameters()

#     parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
#     kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
#     params = ClipParams(**parameters)

#     print(params)

#     tilefile = config.tileset().filename(ax_tiles, axis=axis, **kwargs)

#     def length():

#         with open(tilefile) as fp:
#             return sum(1 for line in fp)

#     def arguments():

#         with open(tilefile) as fp:
#             for line in fp:
#                 row, col = tuple(int(x) for x in line.split(','))
#                 yield (
#                     ClipHeightTile,
#                     axis,
#                     row,
#                     col,
#                     params,
#                     kwargs
#                 )

#     with Pool(processes=processes) as pool:

#         pooled = pool.imap_unordered(starcall, arguments())

#         with click.progressbar(pooled, length=length()) as iterator:
#             for _ in iterator:
#                 pass


def ValleyBottomMaskTile(axis, row, col, params, **kwargs):

    tileset = config.tileset()

    def _tilename(dataset):
        return tileset.tilename(
            dataset,
            axis=axis,
            row=row,
            col=col)

    # dem_raster = tileset.tilename('dem', row=row, col=col)
    height_raster = _tilename(params.height)
    distance_raster = _tilename(params.distance)
    output = _tilename(params.output)

    # with rio.open(dem_raster) as ds:
    #     dem_nodata = (ds.read(1) == ds.nodata)

    with rio.open(distance_raster) as ds:
        distance = ds.read(1)

    with rio.open(height_raster) as ds:

        hand = ds.read(1)

        mask = np.float32(
            ~(
                (hand < -params.max_slope * params.dist_resolution * distance)
                & (params.dist_resolution * distance > params.min_distance)
                | (hand > params.max_height)
            )
        )
        
        mask[hand == ds.nodata] = 0

        hand[mask == 0] = ds.nodata
        # hand[dem_nodata] = ds.nodata

        if params.buffer_width > 0:

            speedup.raster_buffer(mask, 0, params.buffer_width, 1)
            hand[(hand == ds.nodata) & (mask == 1)] = params.max_height

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(hand, 1)

def ValleyBottomMask(axis, ax_tiles='shortest_tiles', processes=1, **kwargs):
    """
    Creates a raster buffer with distance buffer_width pixels
    around data pixels and crop out data outside of the resulting buffer

    @api    fct-corridor:valleymask

    @input  tiles: ax_shortest_tiles
    @input  height: ax_nearest_height
    @input  distance: ax_nearest_distance

    @param  dist_resolution: 1.0
    @param  min_distance: 1000.0
    @param  max_slope: 0.01
    @param  max_height: 12.0
    @param  buffer_width: 40.0

    @output mask: ax_valley_mask
    """

    parameters = ValleyBottomMaskDefaultParameters()

    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = ValleyBottomMaskParams(**parameters)

    tilefile = config.tileset().filename(ax_tiles, axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                ValleyBottomMaskTile,
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
