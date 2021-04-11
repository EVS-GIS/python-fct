# coding: utf-8

"""
Height Above Nearest Drainage,
ie. elevations relative to (theoretical) stream network
aka. detrended DEM

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
from multiprocessing import Pool

import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import transform as fct
from .. import speedup
from ..config import (
    LiteralParameter,
    DatasetParameter
)
from ..rasterize import rasterize_linestringz
# from ..swath import nearest_value_and_distance
from ..measure.Measurement import nearest_value_and_distance
from ..cli import starcall

class Parameters:
    """
    Height above elevation profile parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    dem = DatasetParameter('elevation raster (DEM)', type='input')
    elevation_profile = DatasetParameter('elevation profile (shapefile)', type='input')
    mask = DatasetParameter('height raster defining domain mask', type='input')
    height = DatasetParameter('height above valley bottom', type='output')
    distance = DatasetParameter('distance to drainage pixels (raster)', type='output')
    nearest = DatasetParameter('nearest drainage axis (raster)', type='output')

    mask_height_max = LiteralParameter(
        'maximum height defining domain mask')
    buffer_width = LiteralParameter(
        'enlarge domain mask by buffer width expressed in real distance unit (eg. meters)')
    resolution = LiteralParameter(
        'raster resolution, ie. pixel size, in real distance unit (eg. meters)')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.dem = 'dem'

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.elevation_profile = dict(
                key='elevation_profile_valley_bottom',
                tiled=False,
                subdir='NETWORK/REF')
            self.mask = 'shortest_height'
            self.height = 'height_above_valley_bottom'
            self.distance = 'off' # 'nearest_distance'
            self.nearest = 'off' # 'nearest_drainage_axis'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.elevation_profile = dict(
                key='elevation_profile_valley_bottom',
                tiled=False,
                axis=axis)
            self.mask = dict(key='ax_shortest_height', axis=axis)
            self.height = dict(key='ax_height_above_valley_bottom', axis=axis)
            self.distance = 'off' # dict(key='ax_nearest_distance', axis=axis)
            self.nearest = 'off' # dict(key='ax_nearest_drainage_axis', axis=axis)

        self.mask_height_max = 20.0
        self.buffer_width = 0.0
        self.resolution = 5.0

def HeightAboveReferenceTile(
        row: int,
        col: int,
        params: Parameters,
        **kwargs):
    """
    Tile processing
    """

    elevation_raster = params.dem.tilename(row=row, col=col, **kwargs)
    profile_shapefile = params.elevation_profile.filename(**kwargs)

    if not profile_shapefile.exists():
        profile_shapefile = params.elevation_profile.filename(tileset=None, **kwargs)
        # config.filename(params.drainage, axis=axis, **kwargs)
        assert profile_shapefile.exists()

    # valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)
    mask_rasterfile = params.mask.tilename(row=row, col=col, **kwargs)
    # tileset.tilename(params.mask, axis=axis, row=row, col=col, **kwargs)

    output_height = params.height.tilename(row=row, col=col, **kwargs)

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        elevation_nodata = ds.nodata
        profile = ds.profile.copy()

    with rio.open(mask_rasterfile) as ds:

        mask = ds.read(1)
        height, width = mask.shape
        mask_nodata = ds.nodata

        if params.mask_height_max > 0:

            height_max = params.mask_height_max
            valid = (mask != ds.nodata) & (mask >= -height_max) & (mask <= height_max)
            mask[~valid] = ds.nodata

        if params.buffer_width > 0:

            speedup.raster_buffer(
                mask,
                ds.nodata,
                params.buffer_width / params.resolution)

        refaxis_pixels = list()

        # def accept(feature):

        #     properties = feature['properties']
        #     return properties['AXIS'] == axis

        # def intile(i, j):
        #     return all([i >= 0, i < height, j >= 0, j < width])

        def accept_pixel(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        # coord = itemgetter(0, 1, 2)
        # coord = itemgetter(0, 1)
        unique = set()

        with fiona.open(profile_shapefile) as fs:
            for feature in fs:

                # if accept(feature):

                axis = feature['properties']['AXIS']

                coordinates = np.array(
                    feature['geometry']['coordinates'],
                    dtype='float32')

                coordinates[:, :2] = fct.worldtopixel(coordinates[:, :2], ds.transform)

                for a, b in zip(coordinates[:-1], coordinates[1:]):

                    # if intile(a[0], a[1]):
                    #     a[2] = elevations[a[0], a[1]]

                    # if intile(b[0], b[1]):
                    #     b[2] = elevations[b[0], b[1]]

                    for i, j, z in rasterize_linestringz(a, b):
                        if accept_pixel(i, j) and (i, j) not in unique:
                            # distance[i, j] = 0
                            # measure[i, j] = m
                            # z = elevations[i, j]
                            refaxis_pixels.append((i, j, z, axis))
                            unique.add((i, j))

        if refaxis_pixels:

            nearest, reference, distance = nearest_value_and_distance(
                np.array(refaxis_pixels),
                mask,
                mask_nodata)

            distance = distance * params.resolution
            distance[mask == ds.nodata] = ds.nodata

            hand = elevations - reference
            hand[(mask == ds.nodata) | (elevations == elevation_nodata)] = mask_nodata

        else:

            hand = distance = np.full((height, width), ds.nodata, dtype='float32')

        profile.update(compress='deflate')

        with rio.open(output_height, 'w', **profile) as dst:
            dst.write(hand, 1)

        if not params.distance.none:

            output_distance = params.distance.tilename(row=row, col=col, **kwargs)

            with rio.open(output_distance, 'w', **profile) as dst:
                dst.write(distance, 1)

        if not params.nearest.none:

            output_nearest = params.nearest.tilename(axis=axis, row=row, col=col, **kwargs)

            profile.update(dtype='uint32', nodata=0)

            with rio.open(output_nearest, 'w', **profile) as dst:
                dst.write(nearest, 1)

def HeightAboveReference(
        params: Parameters,
        processes: int = 1,
        **kwargs):
    """
    Calculate distance and height above nearest drainage
    Other keywords are passed to dataset filename templates.
    """

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:
                row, col = (int(x) for x in line.split(','))
                yield (
                    HeightAboveReferenceTile,
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
