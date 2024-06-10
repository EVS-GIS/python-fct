"""
Drape stream network on DEM

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import defaultdict, Counter
import numpy as np
import click

import fiona
from shapely.geometry import (
    asShape,
    box
)
import rasterio as rio

from ..config import (
    config,
    DatasetParameter
)

class Parameters():

    elevations = DatasetParameter('elevation raster (DEM)', type='input')
    stream_network = DatasetParameter('stream network', type='input')
    draped = DatasetParameter('draped network', type='output')

    def __init__(self):
        """
        Default paramater values
        """

        self.elevations = 'dem'
        self.stream_network = 'stream-network-cartography'
        self.draped = 'stream-network-draped'

def DrapeNetworkAndAdjustElevations(params):
    """
    Drape hydrography vectors on DEM
    and adjust elevation profile to ensure
    monotonic decreasing z across network.
    """

    # dem_vrt = '/var/local/fct/RMC/DEM_RGE5M_TILES2.vrt'
    # vectorfile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/HYDROGRAPHY.shp'
    # output = '/var/local/fct/RMC/TILES2/HYDROGRAPHY.shp'

    dem_vrt = params.elevations.filename()
    # config.tileset().filename(dataset)
    networkfile = params.stream_network.filename(tileset=None)
    # config.filename('stream-network-cartography') # filename ok
    output = params.draped.filename(tileset=None)
    # config.filename('stream-network-draped') # filename ok

    graph = defaultdict(list)
    indegree = Counter()
    features = list()

    click.secho('Drape Stream Vectors on DEM', fg='cyan')

    with rio.open(dem_vrt) as ds:
        with fiona.open(networkfile) as fs:

            feature_count = len(fs)
            options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)

            with click.progressbar(fs) as progress:
                
                for feature in progress:

                    a = feature['properties']['NODEA']
                    b = feature['properties']['NODEB']
                    coordinates = np.array(feature['geometry']['coordinates'])

                    z = np.array(list(ds.sample(coordinates[:, :2], 1)))
                    coordinates[:, 2] = np.ravel(z)
                    feature['geometry']['coordinates'] = coordinates
                    
                    idx = len(features)
                    graph[a].append((b, idx))
                    indegree[b] += 1
                    features.append(feature)

    click.secho('Adjust Elevation Profile', fg='cyan')

    nodez = defaultdict(lambda: float('inf'))
    queue = [node for node in graph if indegree[node] == 0]

    with fiona.open(output, 'w', **options) as fst:
        with click.progressbar(length=feature_count) as progress:
            while queue:

                source = queue.pop(0)

                for node, idx in graph[source]:

                    feature = features[idx]
                    coordinates = feature['geometry']['coordinates']
                    a = feature['properties']['NODEA']
                    b = feature['properties']['NODEB']

                    zmin = nodez[a]

                    for k, z in enumerate(coordinates[:, 2]):
                                
                        if z != ds.nodata and z <= zmin:
                            zmin = z
                        
                        # Clamp z to upstream elevation
                        coordinates[k, 2] = zmin

                    if not np.isinf(feature['geometry']['coordinates']).any():
                        fst.write(feature)
                    progress.update(1)

                    nodez[b] = zmin
                    indegree[node] -= 1

                    if indegree[node] == 0:
                        queue.append(node)

def SplitStreamNetworkIntoTiles(params, tileset='default'):

    tileset_data = config.tileset(tileset)
    networkfile = params.draped.filename(tileset=None)
    # config.filename('stream-network-draped') # filename ok

    with fiona.open(networkfile) as fs:

        properties = fs.schema['properties'].copy()
        properties.update(
            ROW='int',
            COL='int'
        )

        schema = dict(
            geometry=fs.schema['geometry'],
            properties=properties)

        options = dict(driver=fs.driver, crs=fs.crs, schema=schema)

        with click.progressbar(tileset_data.tiles(), length=len(tileset_data)) as iterator:
            for tile in iterator:

                output = params.draped.tilename(row=tile.row, col=tile.col, tileset=tileset)
                # config.tileset().tilename('stream-network-draped', row=tile.row, col=tile.col)
                with fiona.open(output, 'w', **options) as dst:

                    tile_geom = box(*tile.bounds)

                    for feature in fs.filter(bbox=tile.bounds):

                        intersection = asShape(feature['geometry']).intersection(tile_geom)

                        if intersection.geometryType() == 'LineString':

                            props = feature['properties']
                            props.update(ROW=tile.row, COL=tile.col)
                            dst.write({
                                'geometry': intersection.__geo_interface__,
                                'properties': props
                            })

                        elif intersection.geometryType() in ('MultiLineString', 'GeometryCollection'):

                            for geom in intersection.geoms:
                                if geom.geometryType() == 'LineString':

                                    props = feature['properties']
                                    props.update(ROW=tile.row, COL=tile.col)
                                    dst.write({
                                        'geometry': geom.__geo_interface__,
                                        'properties': props
                                    })