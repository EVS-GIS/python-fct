# coding: utf-8

"""
Sequence :

4. Accumulate/Resolve Acc Graph/InletAreas
5. FlowAccumulation (*)
6. StreamToFeature (*)

(*) Possibly Parallel Steps

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
import itertools
from operator import itemgetter

import numpy as np
import click

import fiona
import fiona.crs
import rasterio as rio

from .. import speedup
from .. import transform as fct
from .. import terrain_analysis as ta
from ..config import config

from multiprocessing import Pool
from ..cli import starcall_nokwargs

from ..config import (
    config,
    DatasetParameter,
    DatasourceParameter
)

class Parameters:
    """
    Prepare hydrologic network
    """
    sources = DatasetParameter('stream sources from the reference hydrologic network', type='input')
    elevations = DatasetParameter('DEM', type='input')
    flow = DatasetParameter('flow direction raster', type='input')
    acc = DatasetParameter('accumulation raster (drainage area)', type='input')
    streamsources = DatasetParameter('streams from sources', type='output')
    inlets = DatasetParameter('tile inlets (point) shapefile', type='input')
    inletsource = DatasetParameter('tile inlets source', type='input')
    noflowsources = DatasetParameter('no-flow from sources', type='output')

    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.sources = 'river-network-sources'
        self.elevations = 'dem'
        self.flow = 'flow'
        self.acc = 'acc'
        self.streamsources = 'streams-from-sources'
        self.inlets = 'inlets'
        self.inletsource = 'inlet-sources'
        self.noflowsources = 'noflow-from-sources'

def tileindex(tileset='default'):
    """
    Return default tileindex
    """
    return config.tileset(tileset).tileindex

def CreateSourcesGraph(params, tileset='default'):
    """
    DOCME
    """

    tile_index = tileindex(tileset)
    dem_rasterfile = params.elevations.filename(tileset=tileset)
    sources = params.sources.filename(tileset=None)

    click.secho('Build sources graph', fg='cyan')

    graph = dict()
    indegree = Counter()

    dem = rio.open(dem_rasterfile)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[(row, col)].gid
            inlet_shapefile = params.inlets.tilename(row=row, col=col, tileset=tileset)
            flow_raster = params.flow.tilename(row=row, col=col, tileset=tileset)

            with rio.open(flow_raster) as ds:

                flow = ds.read(1)
                height, width = flow.shape

                with fiona.open(inlet_shapefile) as fs:
                    for feature in fs:

                        # connect outlet->inlet

                        from_tile = feature['properties']['FROM']
                        area = 0.0
                        from_i, from_j = dem.index(feature['properties']['FROMX'], feature['properties']['FROMY'])
                        i, j = dem.index(*feature['geometry']['coordinates'])
                        graph[(from_tile, from_i, from_j)] = (tile, i, j, area)
                        indegree[(tile, i, j)] += 1

                        # connect inlet->tile outlet

                        loci, locj = ds.index(*feature['geometry']['coordinates'])
                        locti, loctj = ta.outlet(flow, loci, locj)
                        ti, tj = dem.index(*ds.xy(locti, loctj))

                        if (locti, loctj) == (loci, locj):
                            continue
                        
                        if ti >= 0 and tj >= 0:
                            graph[(tile, i, j)] = (tile, ti, tj, 0)
                            indegree[(tile, ti, tj)] += 1

                with fiona.open(sources) as fs:
                    for feature in fs:

                        loci, locj = ds.index(*feature['geometry']['coordinates'])

                        if not all([loci >= 0, loci < height, locj >= 0, locj < width]):
                            continue

                        # connect exterior->inlet

                        i, j = dem.index(*feature['geometry']['coordinates'])
                        area = 1.0
                        graph[(-2, i-1, j-1)] = (tile, i, j, area)
                        indegree[(tile, i, j)] += 1

                        # connect inlet->tile outlet

                        locti, loctj = ta.outlet(flow, loci, locj)

                        if (locti, loctj) == (loci, locj):
                            continue

                        ti, tj = dem.index(*ds.xy(locti, loctj))
                        
                        if ti >= 0 and tj >= 0:
                            graph[(tile, i, j)] = (tile, ti, tj, 0)
                            indegree[(tile, ti, tj)] += 1


    dem.close()

    click.secho('Created graph with %d nodes' % len(graph), fg='green')

    return graph, indegree

def TileInletSources(params, tile, keys, areas, tileset='default'):
    """
    Output inlet points,
    attributed with the total upstream drained area.
    """

    row = tile.row
    col = tile.col
    gid = tile.gid
    
    output = params.inletsource.tilename(row=row, col=col, tileset=tileset)

    crs = fiona.crs.from_epsg(config.srid)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    dem_file = config.tileset(tileset).filename('dem')
    dem = rio.open(dem_file)

    cum_areas = defaultdict(lambda: 0.0)

    for key in keys:
        cum_areas[key[1:]] += areas.get(key[1:], 0)

    with fiona.open(output, 'w', **options) as dst:
        for i, j in cum_areas:

            x, y = dem.xy(i, j)
            area = cum_areas[i, j]
            if area > 0.0:
                geom = {'type': 'Point', 'coordinates': [x, y]}
                props = {'TILE': gid}
                feature = {'geometry': geom, 'properties': props}
                dst.write(feature)

    dem.close()


def InletSources(params, tileset='default'):
    """
    Accumulate areas across tiles
    and output per tile inlet shapefiles
    with contributing area flowing into tile.
    """

    tile_index = tileindex(tileset)
    tiles = {tile.gid: tile for tile in tile_index.values()}

    graph, indegree = CreateSourcesGraph(params, tileset=tileset)
    
    # Check a random tile just to get pixels x and y size
    flow_raster = params.flow.tilename(row=tiles.get(1).row, col=tiles.get(1).col, tileset=tileset)
    with rio.open(flow_raster) as ds:
        pixelSizeX = ds.profile['transform'][0]
        pixelSizeY =-ds.profile['transform'][4]
        
    coeff = (pixelSizeX*pixelSizeY)*1e-6

    click.secho('Accumulate areas', fg='cyan')
    areas, res = speedup.graph_acc(graph, coeff)

    keys = sorted(graph.keys() | indegree.keys(), key=itemgetter(0))
    groups = itertools.groupby(keys, key=itemgetter(0))

    click.secho('Write inlet shapefiles', fg='cyan')
    with click.progressbar(groups, length=len(tile_index)) as iterator:
        for tile_gid, keys in iterator:

            if tile_gid in tiles:
                tile = tiles[tile_gid]
                TileInletSources(params, tile, keys, areas, tileset)

def StreamToFeatureFromSourcesTile(params, row, col, min_drainage, tileset='default'):
    """
    DOCME
    """

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]

    flow_raster = params.flow.tilename(row=row, col=col, tileset=tileset)
    acc_raster = params.acc.tilename(row=row, col=col, tileset=tileset)
    sources = params.inletsource.tilename(row=row, col=col, tileset=tileset)
    output = params.streamsources.tilename(row=row, col=col, tileset=tileset)

    if not os.path.exists(sources):
        click.secho(
            '\nMissing inlet-sources for tile (%d, %d)' % (row, col),
            fg='yellow')
        return

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('GID', 'int'),
            ('HEAD', 'int:1'),
            ('ROW', 'int:4'),
            ('COL', 'int:4')
        ]
    }
    crs = fiona.crs.from_epsg(config.srid)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)

        height, width = flow.shape

        def intile(i, j):
            return all([i >= 0, i < height, j >= 0, j < width])

        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

        with fiona.open(sources) as fs:
            for feature in fs:

                x, y = feature['geometry']['coordinates']
                i, j = ds.index(x, y)

                while intile(i, j) and streams[i, j] == 0:

                    streams[i, j] = 1
                    direction = flow[i, j]

                    if direction == -1 or direction == 0:
                        break

                    n = int(np.log2(direction))
                    i = i + ci[n]
                    j = j + cj[n]

        with fiona.open(output, 'w', **options) as dst:

            for current, (segment, head) in enumerate(speedup.stream_to_feature(streams, flow)):

                coords = ta.pixeltoworld(np.fliplr(np.int32(segment)), ds.transform, gdal=False)
                dst.write({
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': coords},
                    'properties': {
                        'GID': current,
                        'HEAD': 1 if head else 0,
                        'ROW': row,
                        'COL': col
                    }
                })

def AggregateStreamsFromSources(params, tileset='default'):
    """
    Aggregate Streams Shapefile
    """

    tile_index = tileindex(tileset)
    output = params.streamsources.filename(tileset=tileset)

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('GID', 'int'),
            ('HEAD', 'int:1'),
            ('ROW', 'int:4'),
            ('COL', 'int:4')
        ]
    }
    crs = fiona.crs.from_epsg(config.srid)
    options = dict(driver=driver, crs=crs, schema=schema)

    gid = itertools.count(1)

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tile_index) as progress:

            for row, col in progress:

                shapefile = output = params.streamsources.tilename(row=row, col=col, tileset=tileset)

                if os.path.exists(shapefile):
                    with fiona.open(shapefile) as fs:
                        for feature in fs:
                            feature['properties']['GID'] = next(gid)
                            dst.write(feature)

def NoFlowPixelsTile(params, row, col, min_drainage, tileset='default'):

    flow_raster = params.flow.tilename(row=row, col=col, tileset=tileset)
    acc_raster = params.acc.tilename(row=row, col=col, tileset=tileset)
    stream_features = params.streamsources.tilename(row=row, col=col, tileset=tileset)
    output = params.noflowsources.tilename(row=row, col=col, tileset=tileset)

    if not os.path.exists(stream_features):
        click.secho(
            '\nMissing streams-from-sources for tile (%d, %d)' % (row, col),
            fg='yellow')
        return

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('ROW', 'int:4'),
            ('COL', 'int:4')
        ]
    }
    crs = fiona.crs.from_epsg(config.srid)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        height, width = flow.shape

        # streams = np.zeros_like(flow, dtype='int16')
        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

        with fiona.open(stream_features) as fs:
            for feature in fs:

                coordinates = np.array(feature['geometry']['coordinates'], dtype='float32')
                pixels = fct.worldtopixel(coordinates, ds.transform)
                
                for i, j in pixels:
                    if 0 <= i < height and 0 <= j < width:
                        streams[i, j] = 1

                # pixels = np.array([
                #     (i, j)
                #     for i, j in fct.worldtopixel(coordinates, ds.transform)
                #     if 0 <= i < height and 0 <= j < width
                # ])

                # streams[pixels[:, 0], pixels[:, 1]] = 1

        with fiona.open(output, 'w', **options) as dst:

            pixels = speedup.noflow(streams, flow)

            if pixels:

                coordinates = ta.pixeltoworld(
                    np.int32(pixels),
                    ds.transform,
                    gdal=False)

                for current, point in enumerate(coordinates):
                    dst.write({
                        'type': 'Feature',
                        'geometry': {'type': 'Point', 'coordinates': point},
                        'properties': {'GID': current, 'ROW': row, 'COL': col}
                    })

def AggregateNoFlowPixels(params, tileset='default'):
    """
    Aggregate No Flow Shapefiles
    """

    tile_index = tileindex(tileset)
    output = params.noflowsources.filename(tileset=tileset)

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('ROW', 'int'),
            ('COL', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(config.srid)
    options = dict(driver=driver, crs=crs, schema=schema)

    gid = itertools.count(1)

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tile_index) as progress:
            for row, col in progress:

                filename = params.noflowsources.tilename(row=row, col=col, tileset=tileset)
                
                if os.path.exists(filename):
                    with fiona.open(filename) as fs:
                        for feature in fs:
                            feature['properties']['GID'] = next(gid)
                            dst.write(feature)

    count = next(gid) - 1
    click.secho('Found %d not-flowing stream nodes' % count, fg='cyan')


def StreamToFeatureFromSources(params, min_drainage, tileset='default', processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                StreamToFeatureFromSourcesTile,
                params,
                row,
                col,
                min_drainage,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
            
def NoFlowPixels(params, min_drainage, tileset='default', processes=1):
    
    def arguments():

        for tile in config.tileset(tileset).tiles():
            row = tile.row
            col = tile.col
            yield (
                NoFlowPixelsTile,
                params,
                row,
                col,
                min_drainage,
                tileset
            )

    arguments = list(arguments())

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall_nokwargs, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass