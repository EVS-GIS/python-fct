# coding: utf-8

import os
from collections import defaultdict, Counter
import numpy as np
import click
import itertools
import math

import rasterio as rio

import fiona
import terrain_analysis as ta

from config import tileindex, filename
from rasterize import rasterize_linestringz

def DrapeNetworkAndAdjustElevations():
    """
    Drape hydrography vectors on DEM
    and adjust elevation profile to ensure
    monotonic decreasing z across network.
    """

    dem_vrt = '/var/local/fct/RMC/DEM_RGE5M_TILES2.vrt'
    vectorfile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/HYDROGRAPHY.shp'
    output = '/var/local/fct/RMC/TILES2/HYDROGRAPHY.shp'

    graph = defaultdict(list)
    indegree = Counter()
    features = list()

    click.secho('Drape Stream Vectors on DEM', fg='cyan')

    with rio.open(dem_vrt) as ds:
        with fiona.open(vectorfile) as fs:

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

                    fst.write(feature)
                    progress.update(1)

                    nodez[b] = zmin
                    indegree[node] -= 1

                    if indegree[node] == 0:
                        queue.append(node)

def DispatchHydrographyToTiles():

    src = '/var/local/fct/RMC/TILES2/HYDROGRAPHY_TILED.shp'

    def rowcol(feature):
        return feature['properties']['ROW'], feature['properties']['COL']

    with fiona.open(src) as fs:
        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)
        features = sorted(list(fs), key=rowcol)

    groups = itertools.groupby(features, key=rowcol)

    with click.progressbar(groups, length=len(tileindex())) as progress:
        for (row, col), features in progress:
            with fiona.open(filename('hydrography', row=row, col=col), 'w', **options) as fst:
                for feature in features:
                    fst.write(feature)

def BurnTile(tileset, row, col, burn_delta=0.0):
    """
    DOCME
    """

    elevation_raster = filename(tileset, row=row, col=col)
    hydrography = filename('hydrography', row=row, col=col)

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

    return elevations
