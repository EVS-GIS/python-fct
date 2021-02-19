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

import os
# from collections import namedtuple
from operator import itemgetter
from multiprocessing import Pool

import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import terrain_analysis as ta
from .. import speedup
from ..config import (
    config,
    LiteralParameter,
    DatasetParameter
)
from ..rasterize import rasterize_linestringz
# from ..swath import nearest_value_and_distance
from .Measurement import nearest_value_and_distance
from ..cli import starcall

class Parameters:
    """
    Height above nearest drainage (HAND) parameters
    """

    dem = DatasetParameter('elevation raster (DEM)', type='input')
    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    drainage = DatasetParameter('drainage network shapefile', type='input')
    mask = DatasetParameter('height raster defining domain mask', type='input')
    height = DatasetParameter('height raster (HAND)', type='output')
    distance = DatasetParameter('distance to reference pixels (raster)', type='output')
    nearest = DatasetParameter('nearest drainage axis (raster)', type='output')

    mask_height_max = LiteralParameter(
        'maximum height defining domain mask')
    buffer_width = LiteralParameter(
        'enlarge domain mask by buffer width expressed in real distance unit (eg. meters)')
    resolution = LiteralParameter(
        'raster resolution, ie. pixel size, in real distance unit (eg. meters)')

    def __init__(self):
        """
        Default parameter values,
        with reference elevation = talweg
        """

        self.dem = 'dem'
        self.tiles = 'shortest_tiles'
        self.drainage = 'network-cartography-ready'
        self.mask = 'shortest_height'
        self.height = 'nearest_height'
        self.distance = 'nearest_distance'
        self.nearest = 'nearest_drainage_axis'
        self.mask_height_max = 20.0
        self.buffer_width = 0.0
        self.resolution = 5.0

def DrapeLineString(coordinates, elevations, **kwargs):
    """
    Set coordinates z from `elevations` values
    """

    elevation_raster = config.tileset().filename(elevations, **kwargs)

    with rio.open(elevation_raster) as ds:

        z = np.array(list(ds.sample(coordinates[:, :2], 1)))
        coordinates[:, 2] = z[:, 0]

def HeightAboveNearestDrainageTile(
        axis,
        row,
        col,
        params,
        **kwargs):
    """
    Tile processing
    """

    # tileset = config.tileset()

    elevation_raster = params.dem.tilename(row=row, col=col, **kwargs)
    # tileset.tilename(params.dem, row=row, col=col, **kwargs)
    
    drainage_shapefile = params.drainage.filename(axis=axis, **kwargs)
    # tileset.filename(params.drainage, axis=axis, **kwargs)
    
    if not os.path.exists(drainage_shapefile):
        drainage_shapefile = params.drainage.filename(axis=axis, tileset=None, **kwargs)
        # config.filename(params.drainage, axis=axis, **kwargs)
        assert os.path.exists(drainage_shapefile)

    # valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)
    mask_rasterfile = params.mask.tilename(axis=axis, row=row, col=col, **kwargs)
    # tileset.tilename(params.mask, axis=axis, row=row, col=col, **kwargs)

    output_height = params.height.tilename(axis=axis, row=row, col=col, **kwargs)
    # tileset.tilename(params.height, axis=axis, row=row, col=col, **kwargs)
    output_distance = params.distance.tilename(axis=axis, row=row, col=col, **kwargs)
    # tileset.tilename(params.distance, axis=axis, row=row, col=col, **kwargs)
    output_nearest = params.nearest.tilename(axis=axis, row=row, col=col, **kwargs)

    with rio.open(mask_rasterfile) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        if params.mask_height_max > 0:

            height_max = params.mask_height_max
            valid = (mask != ds.nodata) & (mask >= -height_max) & (mask <= height_max)
            mask[~valid] = ds.nodata

        if params.buffer_width > 0:
            
            speedup.raster_buffer(
                mask,
                ds.nodata,
                params.buffer_width / params.resolution)

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        refaxis_pixels = list()

        # def accept(feature):

        #     properties = feature['properties']
        #     return properties['AXIS'] == axis

        # def intile(i, j):
        #     return all([i >= 0, i < height, j >= 0, j < width])

        def accept_pixel(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        # coord = itemgetter(0, 1, 2)
        coord = itemgetter(0, 1)
        unique = set()

        with rio.open(elevation_raster) as ds2:

            elevations = ds2.read(1)

            with fiona.open(drainage_shapefile) as fs:
                for feature in fs:

                    # if accept(feature):

                    axis = feature['properties']['AXIS']

                    coordinates = np.array([
                        coord(p) + (0.0, ) for p in feature['geometry']['coordinates']
                    ], dtype='float32')

                    # override z from elevation raster
                    # just in case we forgot to drape stream network on DEM
                    DrapeLineString(coordinates, params.dem.name, **kwargs)

                    coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], ds.transform, gdal=False)

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
                ds.nodata)

            distance = distance * params.resolution
            distance[mask == ds.nodata] = ds.nodata

            hand = elevations - reference
            hand[(mask == ds.nodata) | (elevations == ds2.nodata)] = ds.nodata

        else:

            hand = distance = np.full((height, width), ds.nodata, dtype='float32')

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        with rio.open(output_height, 'w', **profile) as dst:
            dst.write(hand, 1)

        profile.update(dtype='uint32', nodata=0)

        with rio.open(output_nearest, 'w', **profile) as dst:
            dst.write(nearest, 1)

def HeightAboveNearestDrainage(
        axis,
        params,
        processes=1,
        # ax_tiles='ax_tiles',
        # elevation='dem',
        # drainage='ax_drainage_network',
        # mask='ax_flow_height',
        # height='ax_nearest_height',
        # distance='ax_nearest_distance',
        # buffer_width=30.0,
        # resolution=5.0,
        **kwargs):
    """
    Calculate distance and height above nearest drainage

    @api    fct-corridor:hand
    @input  tiles: ax_shortest_tiles
    @input  dem: dem
    @input  drainage: ax_drainage_network
    @input  mask: ax_flow_height
    @output height: ax_nearest_height
    @output distance: ax_nearest_distance
    @params buffer_width: 30.0
    @params resolution: 5.0

    Parameters
    ----------

    axis: int

        Axis identifier

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword Parameters
    ------------------

    buffer_width: float

        Width (real world units) of the buffer
        used to expand domain mask,
        defaults to 30.0 m

    resolution: float

        Raster resolution (real world units),
        used to scale distance,
        defaults to 5.0 m

    ax_tiles: str, logical name

        Axis list of intersecting tiles

    elevation: str, logical name

        Absolute elevation raster (DEM)

    drainage: str, logical name

        Drainage network for reference.
        streams-tiled | ax_drainage_network | ax_talweg

    mask: str, logical name

        Mask raster,
        which defines the domain area to process
        from data/nodata values.
        ax_flow_height | ax_valley_bottom | ax_nearest_height

    height: str, logical name

        Output raster for calculated height
        above nearest drainage

    distance: str, logical name

        Output raster for calculated distance
        from nearest drainage

    Other keywords are passed to dataset filename templates.
    """

    # params = HandParams(
    #     elevation=elevation,
    #     drainage=drainage,
    #     mask=mask,
    #     height=height,
    #     distance=distance,
    #     buffer_width=buffer_width,
    #     resolution=resolution
    # )

    tilefile = params.tiles.filename(axis=axis, **kwargs)
    # config.tileset().filename(ax_tiles, axis=axis, **kwargs)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:
                row, col = tuple(int(x) for x in line.split(','))
                yield (
                    HeightAboveNearestDrainageTile,
                    axis,
                    row,
                    col,
                    params,
                    kwargs
                )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
