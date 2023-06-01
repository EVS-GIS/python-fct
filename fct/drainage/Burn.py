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
from multiprocessing import Pool
from ..cli import starcall_nokwargs

import rasterio as rio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio import features
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
    hydro_network_buffer_tiled = DatasetParameter('reference hydrologic network buffered by field clipped by tiles', type='input')
    elevations = DatasetParameter('filled-resolved elevation raster (DEM)', type='input')
    burned_dem = DatasetParameter('burned elevation raster (DEM)', type='output')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.hydro_network = 'hydrography'
        self.hydro_network_buffer = 'stream-network-cartography-buffered'
        self.hydro_network_buffer_tiled = 'stream-network-cartography-buffered-tiled'
        self.elevations = 'dem-drainage-resolved'
        self.burned_dem = 'burned_dem'

def HydroBuffer(params):
    hydro_network = params.hydro_network.filename()
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)

    with fiona.open(hydro_network, 'r') as source:
        # Create output shapefile schema
        schema = source.schema.copy()
        schema['properties']['buffer'] = 'float'
        schema['geometry'] = 'Polygon'

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

def HydroBufferTile(row, col, params, overwrite=True, tileset='default'):
    elevations = params.elevations.tilename(row=row, col=col, tileset=tileset)
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)
    hydro_network_buffer_tiled = params.hydro_network_buffer.tilename(row=row, col=col, tileset=tileset)

    with rio.open(elevations) as raster:
        valid_data = raster.read_masks(1)
        geoms = list(features.shapes(valid_data, transform=raster.transform))

        with fiona.open(hydro_network_buffered) as source:

            options = dict(
                driver=source.driver,
                schema=source.schema,
                crs=source.crs)

            with fiona.open(hydro_network_buffer_tiled, 'w', **options) as dst:
                for feature in source:
                    buff_geom = shape(feature['geometry'])
                    
                    # Clip each LineString individually
                    clipped_geometries = [buff_geom.intersection(shape(sub_geom[0])) for sub_geom in geoms]

                    for item in clipped_geometries:
                        clipped_geom = {
                            'type': 'Feature',
                            'properties': feature['properties'],
                            'geometry': mapping(item)
                        }
                    
                    # Write buffered feature to output vector
                    dst.write(clipped_geom)

# def HydroBufferTile(row, col, params, overwrite=True, tileset='default'):
#     hydro_network = params.hydro_network.filename()
#     hydro_network_buffered = params.hydro_network_buffer.tilename(row=row, col=col, tileset=tileset)
#     elevations = params.elevations.tilename(row=row, col=col, tileset=tileset)

#     # Load the raster mask
#     with rio.open(elevations) as raster:
#         nodata_value = raster.nodata
#         # mask = raster.read_masks(1)
#         # Calculate the out_shape based on the bounds
#         # out_shape = (raster.height, raster.width)
#         print([feature['geometry'] for feature in fiona.open(hydro_network)])
#         maskhydro, _ = mask(raster, shapes=[feature['geometry'] for feature in fiona.open(hydro_network)], invert=True)

#         with fiona.open(hydro_network) as source:
#             options = dict(
#                 driver=source.driver,
#                 schema=source.schema,
#                 crs=source.crs)
            
#         #     features = [shape(f['geometry']) for f in source]

#         #     data_mask = geometry_mask(features, out_shape, raster.transform, invert=True)

#             # Convert the data mask to a rasterio mask
#             # mask = rio.features.geometry_mask([data_mask], out_shape=out_shape, transform=raster.transform)

#             # Appliquer le masque nodata à la géométrie
#             # masked_geom = data_mask & mask.astype(bool)

#             with fiona.open(hydro_network_buffered, 'w', **options) as output:
#                 for feature in source:
#                     geom = shape(feature['geometry'])

#                     # Découper les linestrings en utilisant le masque
#                     if maskhydro.any() and maskhydro.shape == raster.shape and maskhydro.intersects(geom):
#                         intersected = maskhydro.intersection(geom)
#                         if intersected.geom_type == 'LineString':
#                             intersected = [intersected]

#                         # Écrire les nouvelles géométries découpées dans le fichier shapefile de sortie
#                         output.write({
#                             'geometry': mapping(intersected),
#                             'properties': feature['properties']
#                         })

    # with fiona.open(hydro_network, 'r') as source:
    #     polyline_features = list(source)
    #     options = dict(
    #         driver=source.driver,
    #         schema=source.schema,
    #         crs=source.crs)
    
    

    # with fiona.open(hydro_network_buffered, 'w', **options) as dst:
    #     for feature in polyline_features:
    #         properties = feature['properties']
    #         geometry = shape(feature['geometry'])

    #         transformed_geometry = rio.transform.xy(raster_transform, *geometry.coords.xy)
    #         clipped, _ = mask(raster_data, [transformed_geometry], nodata=raster_nodata, crop=True)
    #         clipped = clipped[0]
    #         clipped[clipped != 0] = raster_nodata

    #         clipped_feature = {
    #             'type': 'Feature',
    #             'properties': properties,
    #             'geometry': clipped
    #         }

    #         dst.write(clipped_feature)

    # with fiona.open(hydro_network, 'r') as source:
    #     # Create output shapefile schema
    #     schema = source.schema.copy()
    #     schema['properties']['buffer'] = 'float'
    #     schema['geometry'] = 'Polygon'

    #     options = dict(
    #         driver=source.driver,
    #         schema=schema,
    #         crs=source.crs)
        
    #     # Create output shapefile
    #     with fiona.open(hydro_network_buffered, 'w', **options) as output:
    #         for feature in source:
    #             properties = feature['properties']
    #             geometry = shape(feature['geometry'])
                
    #             # Extract buffer value from attribute field
    #             buffer_value = properties['buffer']

    #             if geometry.geom_type == 'Polygon':
    #                 # Generate buffer geometry for polygon
    #                 print(geometry.geom_type)
    #                 print (properties)
                
    #             # Generate buffer geometry
    #             buffer_geometry = geometry.buffer(buffer_value)
                
    #             # Create buffered feature
    #             buffered_feature = {
    #                 'type': 'Feature',
    #                 'properties': properties,
    #                 'geometry': mapping(buffer_geometry)
    #             }
                
    #             # Write buffered feature to output shapefile
    #             output.write(buffered_feature)

def BurnTileBuffer(row, col, params, burn_delta=0.0, overwrite=True, tileset='default'):
    """
    DOCME
    """

    elevations = params.elevations.tilename(row=row, col=col, tileset=tileset)
    hydrography = params.hydro_network_buffer.filename(tileset=None)
    burned = params.burned_dem.filename(row=row, col=col, tileset=tileset)

    if os.path.exists(burned) and not overwrite:
        click.secho('Output already exists: %s' % burned, fg='yellow')
        return

    with rio.open(elevations) as ds:

        dem_data = ds.read(1)
        transform = ds.transform
        height, width = ds.shape

        if os.path.exists(hydrography):

            with fiona.open(hydrography) as fs:

                # Generate a mask for the polygon extent
                mask = geometry_mask(fs, transform=transform, invert=True,
                                 out_shape=(height, width), all_touched=True)

                # # Modify the DEM values within the polygon extent
                # dem_data = np.where(mask, dem_data - burn_delta, dem_data)

                # Create a new raster to store the modified DEM values
                burned_data = rio.open(
                    burned,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=1,
                    dtype=mask.dtype,
                    crs=ds.crs,
                    transform=transform,
                )

                burned_data.write(mask, 1)
                burned_data.close()

    #             for feature in fs:

    #                 geom = np.array(feature['geometry']['coordinates'], dtype=np.float32)
    #                 geom[:, :2] = np.fliplr(ta.worldtopixel(geom, ds.transform, gdal=False))

    #                 for a, b in zip(geom[:-1], geom[1:]):
    #                     for px, py, z in rasterize_linestringz(a, b):
    #                         if all([py >= 0, py < height, px >= 0, px < width, not np.isinf(z)]):
    #                             elevations[py, px] = z - burn_delta
    #     else:

    #         click.secho('File not found: %s' % hydrography, fg='yellow')

    # return elevations

def BurnBufferDEM(
        params,
        burn_delta,
        overwrite=True,
        tileset='default',
        processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                BurnTileBuffer,
                row,
                col,
                params,
                burn_delta,
                overwrite,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass


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

