# coding: utf-8

"""
DEM Burning
Match mapped stream network and DEM by adjusting stream's elevation

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
from collections import defaultdict, Counter
import numpy as np
import click

import rasterio as rio
import fiona
from shapely.geometry import (
    asShape,
    box,
    shape,
    mapping
)

from shapely.ops import unary_union

from ..config import (
    config,
    LiteralParameter,
    DatasetParameter,
    DatasourceParameter
)

from .. import terrain_analysis as ta
from ..config import config
from ..rasterize import rasterize_linestringz

class Parameters:
    """
    Burns DEM from hydrologic network parameters
    """
    hydro_network = DatasourceParameter('reference hydrologic network shapefile')

    hydro_network_buffer = DatasetParameter('reference hydrologic network buffered by field', type='input')
    elevations = DatasetParameter('filled-resolved elevation raster (DEM)', type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.hydro_network = 'hydrography'
        self.hydro_network_buffer = 'stream-network-cartography-buffered'
        self.elevations = 'dem-drainage-resolved'

def HydroBuffer(params):
    hydro_network = params.hydro_network.filename()
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)

    with fiona.open(hydro_network, 'r') as source:
        # Create output shapefile schema
        schema = source.schema.copy()
        schema['properties']['buffer'] = 'float'
        schema['geometry'] = 'Polygon'
        print (schema)

        options = dict(
            driver=source.driver,
            schema=schema,
            crs=source.crs)
        
        # Create output shapefile
        with fiona.open(hydro_network_buffered, 'w', **options) as output:
            for feature in source:
                properties = feature['properties']
                geometry = shape(feature['geometry'])
                
                # Extract buffer value from attribute field
                buffer_value = properties['buffer']

                if geometry.geom_type == 'Polygon':
                    # Generate buffer geometry for polygon
                    print(geometry.geom_type)
                    print (properties)
                
                # Generate buffer geometry
                buffer_geometry = geometry.buffer(buffer_value)
                
                # Create buffered feature
                buffered_feature = {
                    'type': 'Feature',
                    'properties': properties,
                    'geometry': mapping(buffer_geometry)
                }
                
                # Write buffered feature to output shapefile
                output.write(buffered_feature)
    

def BurnTileBuffer(params, row, col, burn_delta=0.0):
    """
    DOCME
    """

    elevation_raster = params.elevations.tilename(row=row, col=col)
    # config.tileset().tilename(dataset, row=row, col=col)
    hydrography = params.hydrography.tilename(row=row, col=col)
    # config.tileset().tilename('stream-network-draped', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        height, width = elevations.shape

        if os.path.exists(hydrography):

            with fiona.open(hydrography) as fs:
                for feature in fs:

                    geom = np.array(feature['geometry']['coordinates'], dtype=np.float32)
                    geom[:, :2] = np.fliplr(ta.worldtopixel(geom, ds.transform, gdal=False))

                    for a, b in zip(geom[:-1], geom[1:]):
                        for px, py, z in rasterize_linestringz(a, b):
                            if all([py >= 0, py < height, px >= 0, px < width, not np.isinf(z)]):
                                elevations[py, px] = z - burn_delta
        else:

            click.secho('File not found: %s' % hydrography, fg='yellow')

    return elevations




def DispatchHydrographyToTiles():

    src = '/var/local/fct/RMC/TILES2/HYDROGRAPHY_TILED.shp'
    tileindex = config.tileset().tileindex

    def rowcol(feature):
        return feature['properties']['ROW'], feature['properties']['COL']

    with fiona.open(src) as fs:
        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)
        features = sorted(list(fs), key=rowcol)

    groups = itertools.groupby(features, key=rowcol)

    with click.progressbar(groups, length=len(tileindex)) as progress:
        for (row, col), features in progress:
            with fiona.open(config.tileset().filename('hydrography', row=row, col=col), 'w', **options) as fst:
                for feature in features:
                    fst.write(feature)

def BurnTile(params, row, col, burn_delta=0.0):
    """
    DOCME
    """

    elevation_raster = params.elevations.tilename(row=row, col=col)
    # config.tileset().tilename(dataset, row=row, col=col)
    hydrography = params.hydrography.tilename(row=row, col=col)
    # config.tileset().tilename('stream-network-draped', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        height, width = elevations.shape

        if os.path.exists(hydrography):

            with fiona.open(hydrography) as fs:
                for feature in fs:

                    geom = np.array(feature['geometry']['coordinates'], dtype=np.float32)
                    geom[:, :2] = np.fliplr(ta.worldtopixel(geom, ds.transform, gdal=False))

                    for a, b in zip(geom[:-1], geom[1:]):
                        for px, py, z in rasterize_linestringz(a, b):
                            if all([py >= 0, py < height, px >= 0, px < width, not np.isinf(z)]):
                                elevations[py, px] = z - burn_delta
        else:

            click.secho('File not found: %s' % hydrography, fg='yellow')

    return elevations

