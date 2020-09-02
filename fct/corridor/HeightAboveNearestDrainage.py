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
from collections import namedtuple
from operator import itemgetter
from multiprocessing import Pool

import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import terrain_analysis as ta
from .. import speedup
from ..config import config
from ..rasterize import rasterize_linestringz
from ..metrics import nearest_value_and_distance
from ..cli import starcall

HandParams = namedtuple('HandParams', [
    'elevation',
    'drainage',
    'mask',
    'height',
    'distance',
    'buffer_width',
    'resolution'
])

def DrapeLineString(coordinates, dataset, **kwargs):

    elevation_raster = config.tileset().filename(dataset, **kwargs)

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
    see DistanceAndHeightAboveNearestDrainage
    """

    tileset = config.tileset()

    elevation_raster = tileset.tilename(params.elevation, row=row, col=col, **kwargs)
    
    drainage_shapefile = tileset.filename(params.drainage, axis=axis, **kwargs)
    
    if not os.path.exists(drainage_shapefile):
        drainage_shapefile = config.filename(params.drainage, axis=axis, **kwargs)
        assert os.path.exists(drainage_shapefile)

    # valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)
    mask_rasterfile = tileset.tilename(params.mask, axis=axis, row=row, col=col, **kwargs)

    output_height = tileset.tilename(params.height, axis=axis, row=row, col=col, **kwargs)
    output_distance = tileset.tilename(params.distance, axis=axis, row=row, col=col, **kwargs)

    with rio.open(mask_rasterfile) as ds:

        mask = ds.read(1)
        height, width = mask.shape
        
        if params.buffer_width > 0:
            speedup.raster_buffer(mask, ds.nodata, params.buffer_width / params.resolution)

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        refaxis_pixels = list()

        def accept(feature):

            properties = feature['properties']
            return properties['AXIS'] == axis

        # def intile(i, j):
        #     return all([i >= 0, i < height, j >= 0, j < width])

        def accept_pixel(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        coord = itemgetter(0, 1, 2)
        unique = set()

        with rio.open(elevation_raster) as ds2:

            elevations = ds2.read(1)

            with fiona.open(drainage_shapefile) as fs:
                for feature in fs:

                    if accept(feature):

                        coordinates = np.array([
                            coord(p) for p in feature['geometry']['coordinates']
                        ], dtype='float32')

                        # override z from elevation raster
                        # just in case we forgot to drape stream network on DEM
                        DrapeLineString(coordinates, params.elevation, **kwargs)

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
                                    refaxis_pixels.append((i, j, z))
                                    unique.add((i, j))

        # output_refaxis = os.path.join(axdir, 'REF', 'REFAXIS_POINTS.shp')
        # schema = {
        #     'geometry': 'Point',
        #     'properties': [
        #         ('GID', 'int'),
        #         ('I', 'float'),
        #         ('J', 'float'),
        #         ('Z', 'float')
        #     ]
        # }
        # crs = fiona.crs.from_epsg(2154)
        # options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

        # if os.path.exists(output_refaxis):
        #     mode = 'a'
        # else:
        #     mode = 'w'

        # with fiona.open(output_refaxis, mode, **options) as fst:
        #     for k, (i, j, z) in enumerate(refaxis_pixels):
        #         geom = {'type': 'Point', 'coordinates': ds.xy(i, j)}
        #         properties = {'GID': k, 'I': i, 'J': j, 'Z': float(z)}
        #         fst.write({'geometry': geom, 'properties': properties})

        if not refaxis_pixels:
            return

        reference, distance = nearest_value_and_distance(
            np.array(refaxis_pixels),
            mask,
            ds.nodata)

        # scale distance by raster resolution
        distance = params.resolution * distance
        distance[mask == ds.nodata] = ds.nodata

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        # del distance

        # elevations, _ = ReadRasterTile(row, col, 'dem1')

        hand = elevations - reference
        hand[mask == ds.nodata] = ds.nodata

        # clip heights
        hand[((hand < -5.0) & (distance > 1000.0)) | (hand > 15.0)] = ds.nodata

        with rio.open(output_height, 'w', **profile) as dst:
            dst.write(hand, 1)

def HeightAboveNearestDrainage(
        axis,
        processes=1,
        ax_tiles='ax_tiles',
        elevation='dem',
        drainage='ax_drainage_network',
        mask='ax_flow_height',
        height='ax_nearest_height',
        distance='ax_nearest_distance',
        buffer_width=30.0,
        resolution=5.0,
        **kwargs):
    """
    Calculate distance and height above nearest drainage

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

    params = HandParams(
        elevation=elevation,
        drainage=drainage,
        mask=mask,
        height=height,
        distance=distance,
        buffer_width=buffer_width,
        resolution=resolution
    )

    tilefile = config.tileset().filename(ax_tiles, axis=axis, **kwargs)

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

def HeightAboveTalweg(axis, **kwargs):
    """
    Default parameters for HAND with talweg reference
    """

    parameters = dict(
        processes=6,
        elevation='dem',
        ax_tiles='ax_shortest_tiles',
        drainage='ax_talweg',
        mask='ax_shortest_height',
        height='ax_nearest_height',
        distance='ax_nearest_distance',
        buffer_width=30.0,
        resolution=5.0
    )

    parameters.update(kwargs)

    click.secho('--%12s:' % 'Parameters', fg='cyan')
    click.secho('  %12s: %d' % ('axis', axis))

    for parameter, value in parameters.items():
        click.echo('  %12s: %s' % (parameter, value))

    HeightAboveNearestDrainage(axis, **parameters)
