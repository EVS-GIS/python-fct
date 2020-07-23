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

def HeightAboveNearestDrainageTile(axis, row, col, dataset='ax_flow_height'):
    """
    see DistanceAndHeightAboveNearestDrainage
    """

    tileset = config.tileset('landcover')

    network_shapefile = config.filename('streams-tiled')
    elevation_raster = tileset.tilename('tiled', row=row, col=col)
    # valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)
    valley_bottom_rasterfile = tileset.tilename(dataset, axis=axis, row=row, col=col)

    output_relative_z = tileset.tilename('ax_relative_elevation', axis=axis, row=row, col=col)
    output_stream_distance = tileset.tilename('ax_nearest_distance', axis=axis, row=row, col=col)

    with rio.open(valley_bottom_rasterfile) as ds:

        click.echo('Read Valley Bottom')

        valley_bottom = ds.read(1)
        speedup.raster_buffer(valley_bottom, ds.nodata, 6.0)
        height, width = valley_bottom.shape

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        refaxis_pixels = list()

        click.echo('Map Stream Network')

        def accept(feature):

            properties = feature['properties']
            return properties['AXIS'] == axis

        def accept_pixel(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        coord = itemgetter(0, 1, 2)
        unique = set()

        with rio.open(elevation_raster) as ds2:
            elevations = ds2.read(1)

        with fiona.open(network_shapefile) as fs:
            for feature in fs:

                if accept(feature):

                    coordinates = np.array([
                        coord(p) for p in feature['geometry']['coordinates']
                    ], dtype='float32')

                    coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], ds.transform, gdal=False)

                    for a, b in zip(coordinates[:-1], coordinates[1:]):
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

        click.echo('Calculate Reference & Distance Raster')

        reference, distance = nearest_value_and_distance(
            np.array(refaxis_pixels),
            valley_bottom,
            ds.nodata)

        distance = 5.0 * distance

        click.echo('Write output')

        distance[valley_bottom == ds.nodata] = ds.nodata

        with rio.open(output_stream_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        del distance

        # elevations, _ = ReadRasterTile(row, col, 'dem1')

        relative = elevations - reference
        relative[valley_bottom == ds.nodata] = ds.nodata

        with rio.open(output_relative_z, 'w', **profile) as dst:
            dst.write(relative, 1)

def DistanceAndHeightAboveNearestDrainage(axis, **kwargs):
    """
    Calculate distance and height above nearest drainage,
    based on theoretical drainage derived from DEM.
    """

    tilefile = config.filename('ax_tiles', axis=axis)

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    with click.progressbar(tiles) as iterator:
        for _, row, col in iterator:

            HeightAboveNearestDrainageTile(axis, row, col, **kwargs)
