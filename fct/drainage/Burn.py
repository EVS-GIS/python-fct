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
    tileset_10k = DatasetParameter('10k default tileset', type='input')
    tileset_10kbis = DatasetParameter('10k bis tileset', type='input')
    hydrography_strahler_fieldbuf = DatasetParameter('reference stream network with strahler order and buffer field to compute buffer before burn DEM', type='input')
    hydrography_strahler_fieldbuf_complete = DatasetParameter('reference stream network with strahler order and buffer field completed to compute buffer before burn DEM', type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.hydro_network = 'hydrography'
        self.hydro_network_buffer = 'stream-network-cartography-buffered'
        self.hydro_network_buffer_tiled = 'stream-network-cartography-buffered-tiled'
        self.elevations = 'dem-drainage-resolved'
        self.burned_dem = 'burned_dem'
        self.tileset_10k = '10k-tileset'
        self.tileset_10kbis = '10kbis-tileset'
        self.hydrography_strahler_fieldbuf = 'hydrography-strahler-fieldbuf'
        self.hydrography_strahler_fieldbuf_complete = 'hydrography-strahler-fieldbuf-complete'


def find_head_lines(lines):
    head_idx = []

    num_lines = len(lines)
    for i in range(num_lines):
        line = lines[i]
        first_point = line[0]

        has_upstream = False

        for j in range(num_lines):
            if j == i:
                continue
            line = lines[j]
            last_point = line[len(line)-1]

            if first_point == last_point:
                has_upstream = True

        if not has_upstream:
            head_idx.append(i)

    return head_idx

def find_prev_lines(curr_idx, lines):
    prev_idx = []

    num_lines = len(lines)

    line = lines[curr_idx]
    first_point = line[0]

    for i in range(num_lines):
        if i == curr_idx:
            continue
        line = lines[i]
        last_point = line[len(line)-1]

        if first_point == last_point:
            prev_idx.append(i)

    return prev_idx

def find_next_line(curr_idx, lines):
    num_lines = len(lines)

    line = lines[curr_idx]
    last_point = line[len(line)-1]

    next_idx = None

    for i in range(num_lines):
        if i == curr_idx:
            continue
        line = lines[i]
        first_point = line[0]

        if last_point == first_point:
            next_idx = i
            break

    return next_idx

def find_sibling_line(curr_idx, lines):
    num_lines = len(lines)

    line = lines[curr_idx]
    last_point = line[len(line)-1]

    sibling_idx = None

    for i in range(num_lines):
        if i == curr_idx:
            continue
        line = lines[i]
        last_point2 = line[len(line)-1]

        if last_point == last_point2:
            sibling_idx = i
            break

    return sibling_idx


# source code https://here.isnew.info/strahler-stream-order-in-python.html
def strahler_order(params):
    hydro_network = params.hydro_network.filename()
    hydrography_strahler_fieldbuf = params.hydrography_strahler_fieldbuf.filename(tileset=None)
    hydrography_strahler_fieldbuf_complete = params.hydrography_strahler_fieldbuf_complete.filename(tileset=None)

    # load network
    with fiona.open(hydro_network, 'r') as source:

        schema = source.schema.copy()
        lines = []
        strahler_field_name = "strahler"
        strahler_field_type = 'int'
        buffer_field_name = 'buffer'
        buffer_field_type = 'float'

        # Add the new field to the schema
        schema['properties'][strahler_field_name] = strahler_field_type
        schema['properties'][buffer_field_name] = buffer_field_type

        # Create a new dataset with the updated schema
        with fiona.open(hydrography_strahler_fieldbuf, 'w', driver=source.driver, crs=source.crs, schema=schema) as dst:
            # Copy the features from the source shapefile to the destination shapefile
            for feature in source:
                # Create a new feature with the new field
                new_properties = feature['properties']
                new_properties[strahler_field_name] = 0  # Set the strahler field value to 0
                new_properties[buffer_field_name] = 0 # Set the buffer field value to 0

                geom = shape(feature['geometry'])
                # copy line coordinates to find head line
                line = geom.coords
                lines.append(line)
        
                # Create the new feature and write it to the destination shapefile
                new_feature = {
                    'geometry': feature['geometry'],
                    'properties': new_properties
                }
                dst.write(new_feature)

        # save head lines index
        head_idx = find_head_lines(lines)

    # read dataset with the updated schema
    with fiona.open(hydrography_strahler_fieldbuf, 'r') as source_buff:
        options = dict(
            driver=source_buff.driver,
            schema=source_buff.schema,
            crs=source_buff.crs)
        
        # copy feature to save modification before saving
        source_buff_copy = [feature for feature in source_buff]

        # iterate through head lines
        for idx in head_idx:
            curr_idx = idx
            curr_ord = 1
            # head lines order = 1
            source_buff_copy[curr_idx]['properties'][strahler_field_name] = curr_ord
            # go downstream from each head lines
            while True:
                # find next line downstream
                next_idx = find_next_line(curr_idx, lines)
                # stop iteration if no next line
                if not next_idx:
                    break
                # copy next line feature and order
                next_feat = source_buff_copy[next_idx]
                next_ord = next_feat['properties'][strahler_field_name]
                # find sibling line
                sibl_idx = find_sibling_line(curr_idx, lines)
                # if sibling line exist
                if sibl_idx is not None:
                    # copy sibling line feature and order
                    sibl_feat = source_buff_copy[sibl_idx]
                    sibl_ord = sibl_feat['properties'][strahler_field_name]
                    # determinate order base on sibling, next and current line
                    if sibl_ord > curr_ord:
                        break
                    elif sibl_ord < curr_ord:
                        if next_ord == curr_ord:
                            break
                    else:
                        curr_ord += 1
                # update order in feature copy dict
                source_buff_copy[next_idx]['properties'][strahler_field_name] = curr_ord
                # go further downstream
                curr_idx = next_idx

        # write final features from updated features copy
        with fiona.open(hydrography_strahler_fieldbuf_complete, 'w', **options) as modif:
            for feature in source_buff_copy:
                modified_feature = {
                        'type': 'Feature',
                        'properties': feature['properties'],
                        'geometry': feature['geometry'],
                    }

                modif.write(modified_feature)

def HydroBuffer(params):
    """
    Creates a buffered shapefile from a hydro network shapefile.

    Parameters:
    - params (object): An object containing the parameters for buffering.
        - hydro_network (str): The filename of the hydro network shapefile.
        - hydro_network_buffer (str): The filename for the buffered output shapefile.

    Returns:
    - None

    """
    hydro_network = params.hydro_network.filename()
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)

    with fiona.open(hydro_network, 'r') as source:
        # Create output schema
        schema = source.schema.copy()
        schema['properties']['buffer'] = 'float'
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

def ClipBufferTile(row, col, params, overwrite=True, tileset='default'):
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
    hydro_network_buffered = params.hydro_network_buffer.filename(tileset=None)
    hydro_network_buffer_tiled = params.hydro_network_buffer_tiled.tilename(row=row, col=col, tileset=tileset)
    if tileset == 'default' or tileset == '10k':
        tileset_file = params.tileset_10k.filename(tileset=None)
    elif tileset == '10kbis' : 
        tileset_file = params.tileset_10kbis.filename(tileset=None)
    else :
        None

    if os.path.exists(hydro_network_buffer_tiled) and not overwrite:
            click.secho('Output already exists: %s' % hydro_network_buffer_tiled, fg='yellow')
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
                            
                            # Open the tiled hydro network buffer file for writing
                            with fiona.open(hydro_network_buffer_tiled, 'w', **options) as dst:
                                for feature in hydro:
                                    hydro_geom = shape(feature['geometry'])
                                    hydro_properties = feature['properties']

                                # Check if hydro_geom intersects with geom_tile
                                # if not hydro_geom.is_empty and hydro_geom.intersects(geom_tile):
                                    # Perform intersection to clip the geometry
                                    clipped_geometries = hydro_geom.intersection(geom_tile)

                                    # check if clipped_geometries not empty
                                    # if clipped_geometries:
                                    # Create a new feature with the clipped geometry
                                    clipped_geom = {
                                        'type': 'Feature',
                                        'properties': hydro_properties,
                                        'geometry': mapping(clipped_geometries)
                                    }
                                
                                    # Write the clipped feature to the tiled buffer file
                                    dst.write(clipped_geom)


def ClipBuffer(
        params,
        overwrite=True,
        tileset='default',
        processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                ClipBufferTile,
                row,
                col,
                params,
                overwrite,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass


def BurnTileBuffer(row, col, params, burn_delta=5, overwrite=True, tileset='default'):
    """
    Burns hydro network buffer onto the elevation tile.

    Parameters:
    - row (int): The row number of the tile.
    - col (int): The column number of the tile.
    - params (object): necessary parameters.
    - burn_delta (float): Optional. The reduction amount to apply to the intersecting cells. Default is 5.
    - overwrite (bool): Optional. Specifies whether to overwrite existing burned DEM files. Default is True.
    - tileset (str): Optional. The name of the tileset to use. Default is 'default'.

    Returns:
    None
    """

    # Get file paths
    elevations = params.elevations.tilename(row=row, col=col, tileset=tileset)
    hydro_network_buffer_tiled = params.hydro_network_buffer_tiled.tilename(row=row, col=col, tileset=tileset)
    burned = params.burned_dem.tilename(row=row, col=col, tileset=tileset)

    if os.path.exists(burned) and not overwrite:
            click.secho('Output already exists: %s' % burned, fg='yellow')
            return

    if os.path.exists(hydro_network_buffer_tiled) == False:
            return
    
    # Open the hydro network buffer tiled file
    with fiona.open(hydro_network_buffer_tiled) as hydro_buff:
        # Get the polygon geometries
        hydro_geoms = [feature['geometry'] for feature in hydro_buff]

        # Merge the polygons into a single geometry
        hydro_merge = MultiPolygon([shape(poly) for poly in hydro_geoms])

        # Open the elevation file
        with rio.open(elevations) as dem:
            # Create a mask for the polygons
            mask = geometry_mask(hydro_merge, out_shape=dem.shape, transform=dem.transform, invert=True)

            # Read the DEM data
            dem_data = dem.read(1)

            # Create a new raster with the modified values
            with rio.open(burned, 'w', **dem.meta) as output:
                # Apply the reduction amount to the intersecting cells
                burn_data = dem_data.copy()
                burn_data[mask] -= burn_delta

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

