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
import fiona
from shapely.geometry import (
    shape,
    MultiPolygon,
    mapping
)

from ..config import (
    config,
    DatasetParameter,
    DatasourceParameter
)

from .. import terrain_analysis as ta
from ..config import config
from ..rasterize import rasterize_linestringz, rasterize_linestring

class Parameters:
    """
    Burns DEM from hydrologic network parameters
    """
    hydrography_strahler_fieldbuf = DatasetParameter('reference stream network with strahler order and buffer field to compute buffer before burn DEM', type='input')
    hydro_network_buffer = DatasetParameter('reference hydrologic network buffered by field', type='input')
    elevations = DatasetParameter('filled-resolved elevation raster (DEM)', type='input')
    burned_dem = DatasetParameter('burned elevation raster (DEM)', type='output')
    tileset_10k = DatasetParameter('10k default tileset', type='input')
    tileset_10kbis = DatasetParameter('10k bis tileset', type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.hydrography_strahler_fieldbuf = 'hydrography-strahler-fieldbuf'
        self.hydro_network_buffer = 'stream-network-cartography-buffered'
        self.elevations = 'dem-drainage-resolved'
        self.burned_dem = 'burned-dem'
        self.tileset_10k = '10k-tileset'
        self.tileset_10kbis = '10kbis-tileset'
        

def HydroBuffer(params, overwrite=True):
    """
    Creates a buffer from a hydro network.

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydro_network (str): The filename of the hydro network.
        - hydro_network_buffer (str): The filename for the buffered output.
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.

    Returns:
    - None

    """
    # paths to files
    hydrography_strahler_fieldbuf = params.hydrography_strahler_fieldbuf.filename(tileset=None)
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)

    # check overwrite
    if os.path.exists(hydro_network_buffered) and not overwrite:
        click.secho('Output already exists: %s' % hydro_network_buffered, fg='yellow')
        return

    with fiona.open(hydrography_strahler_fieldbuf, 'r') as source:
        # Create output schema
        schema = source.schema.copy()
        schema['geometry'] = 'Polygon'

        options = dict(
            driver=source.driver,
            schema=schema,
            crs=source.crs)

        # Create output vector
        with fiona.open(hydro_network_buffered, 'w', **options) as output:
            for feature in source:
                properties = feature['properties']
                geometry = shape(feature['geometry'])

                # Check if feature is a MultiPolygon
                if geometry.geom_type == 'MultiPolygon':
                    # Iterate over the individual polygons in the MultiPolygon
                    for polygon in geometry:
                        # Create buffered feature for each polygon
                        buffered_feature = {
                            'type': 'Feature',
                            'properties': properties,
                            'geometry': mapping(polygon)
                        }
                        # Write buffered feature to output
                        output.write(buffered_feature)
                else:
                    # Extract buffer value from attribute field
                    buffer_value = properties['buffer']

                    # Generate buffer geometry
                    buffer_geometry = geometry.buffer(buffer_value)

                    # Create buffered feature
                    buffered_feature = {
                        'type': 'Feature',
                        'properties': properties,
                        'geometry': mapping(buffer_geometry)
                    }

                    # Write buffered feature to output
                    output.write(buffered_feature)

def BurnTileBuffer(row, col, params, burn_delta=5, overwrite=True, tileset='default'):
    """
    Clips hydro network buffer tiles based on a specified row and column,
    and saves the clipped tiles to a tiled hydro network buffer file.

    Parameters:
    - row (int): The row number of the tile to clip.
    - col (int): The column number of the tile to clip.
    - params (object): parameters.
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.
    - tileset (str): Optional. The name of the tileset to use. Default is 'default'.

    Returns:
    None
    """

    # Get file paths
    elevations = params.elevations.tilename(row=row, col=col, tileset=tileset)
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)
    burned = params.burned_dem.tilename(row=row, col=col, tileset=tileset)
    # change path with tileset params
    if tileset == 'default' or tileset == '10k':
        tileset_file = params.tileset_10k.filename(tileset=None)
    elif tileset == '10kbis' : 
        tileset_file = params.tileset_10kbis.filename(tileset=None)
    else :
        None

    # check overwrite
    if os.path.exists(burned) and not overwrite:
        click.secho('Output already exists: %s' % burned, fg='yellow')
        return

    # Open the tileset file
    with fiona.open(tileset_file) as tileset_data:
        for tile in tileset_data:
            geom_tile = shape(tile['geometry'])
            properties_tile = tile['properties']

            if properties_tile['ROW'] == row and properties_tile['COL'] == col:
                # Tile found, continue

                    # Open the hydro network buffer file
                    with fiona.open(hydro_network_buffered) as hydro:

                        # Get the polygon geometries
                        hydro_all_geoms = [feature['geometry'] for feature in hydro]
                        # Merge the polygons into a single geometry
                        hydro_all_merge = MultiPolygon([shape(feat) for feat in hydro_all_geoms])

                        if hydro_all_geoms and hydro_all_merge.intersects(geom_tile):

                            options = dict(
                                driver=hydro.driver,
                                schema=hydro.schema,
                                crs=hydro.crs)
                            
                            clipped_geometries = []

                            for feature in hydro:
                                hydro_geom = shape(feature['geometry'])
                                hydro_properties = feature['properties']
                                # Perform intersection to clip the geometry
                                clipped = hydro_geom.intersection(geom_tile)

                                # Check if the intersection result is a MultiPolygon
                                if clipped.geom_type == 'MultiPolygon':
                                    clipped_geometries.extend(clipped)

                                else:
                                    # If the intersection result is a single Polygon, convert it to MultiPolygon
                                    clipped_geometries.append(clipped)
                            
                            # Create a MultiPolygon geometry from the clipped geometries
                            multi_clipped_geometries = MultiPolygon(clipped_geometries)
                            
                            # copy dem with burn
                            # Open the elevation file
                            with rio.open(elevations) as dem:
                                # Create a mask for the polygons
                                mask = geometry_mask(multi_clipped_geometries, out_shape=dem.shape, transform=dem.transform, invert=True)

                                # Read the DEM data
                                dem_data = dem.read(1)

                                # Create a new raster with the modified values
                                with rio.open(burned, 'w', **dem.meta) as output:
                                    # Apply the reduction amount to the intersecting cells
                                    burn_data = dem_data.copy()
                                    burn_data[mask] -= burn_delta

                                    # Write the modified DEM data to the output raster
                                    output.write(burn_data, 1)
                        # copy input dem
                        else:
                            # Open the elevation file
                            with rio.open(elevations) as dem:
                                # Read the DEM data
                                dem_data = dem.read(1)

                                # Create a new raster with the modified values
                                with rio.open(burned, 'w', **dem.meta) as output:
                                    # Apply the reduction amount to the intersecting cells
                                    burn_data = dem_data.copy()

                                    # Write the modified DEM data to the output raster
                                    output.write(burn_data, 1)


def BurnBuffer(
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

def BurnLinesTile(params, row, col, burn_delta=50, overwrite=True, tileset='default'):
    """
    Burn DEM along the hydrologic network lines by delta, create a new raster that keep the inital DEM data with the burned cells along the lines.
    Used to fit the flow direction ant the flow accumulation the the input hydrologic network

    Parameters:
    - params (object): parameters, paths to files.
    - row (int): The row number of the tile to clip.
    - col (int): The column number of the tile to clip.
    - burn_delta (float): Cell reduction value along the lines
    - overwrite (bool): Optional. Specifies whether to overwrite existing tiled buffer files. Default is True.
    - tileset (str): Optional. The name of the tileset to use. Default is 'default'.

    Returns:
    None
    """

    # paths
    elevation_raster = params.elevations.tilename(row=row, col=col, tileset=tileset)
    hydrography = params.hydrography_strahler_fieldbuf.filename(tileset=None)
    burned = params.burned_dem.tilename(row=row, col=col, tileset=tileset)

    # check overwrite
    if os.path.exists(burned) and not overwrite:
        click.secho('Output already exists: %s' % burned, fg='yellow')
        return

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        height, width = elevations.shape

        # New raster with the modified values
        with rio.open(burned, 'w', **ds.meta) as output:
            # copy initial data to keep all the data with the burned ones
            burn_data = elevations.copy()

            with fiona.open(hydrography) as fs:
                for feature in fs:
                    
                    # get the coordinates of the line in a numpy array
                    geom = np.array(feature['geometry']['coordinates'], dtype=np.float32)
                    # transform the coordinate of the line in pixel coordinates
                    geom[:, :2] = np.fliplr(ta.worldtopixel(geom, ds.transform, gdal=False))
                    # a and b are pixel coordinates couple following along the line features
                    for a, b in zip(geom[:-1], geom[1:]):
                        # select the coordinate in geom to draw the line with the minimum of pixels (Bresenham's line algorithm), one data point per intersected cell
                        for px, py in rasterize_linestring(a, b):
                            if all([py >= 0, py < height, px >= 0, px < width]):
                                # burn the pixel selected
                                burn_data[py, px] = elevations[py, px] - burn_delta
                    # Write the burn DEM data
                    output.write(burn_data, 1)

def BurnLines(
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
                BurnLinesTile,
                params,
                row,
                col,
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