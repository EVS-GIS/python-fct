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

import click
import os
import glob
import rasterio as rio
from rasterio.windows import Window
from rasterio.warp import Resampling
import fiona
import fiona.crs
import numpy as np
from collections import namedtuple, defaultdict, Counter
from heapq import heappush, heappop

import richdem as rd
import speedup
import terrain_analysis as ta
import itertools

from config import tileindex, filename

def CreateOutletsGraph():
    """
    DOCME
    """

    tile_index = tileindex()
    DEM = filename('dem', 'input')

    click.secho('Build outlets graph', fg='cyan')

    graph = dict()
    indegree = Counter()

    dem = rio.open(DEM)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[(row, col)].gid
            inlet_shapefile = filename('inlets-agg', row=row, col=col)
            flow_raster = filename('flow', row=row, col=col)

            with rio.open(flow_raster) as ds:

                flow = ds.read(1)
                height, width = flow.shape

                with fiona.open(inlet_shapefile) as fs:
                    for feature in fs:

                        # connect outlet->inlet

                        from_tile = feature['properties']['FROM']
                        area = feature['properties']['LCA']
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

                with fiona.open(filename('exterior-inlets')) as fs:
                    for feature in fs:

                        loci, locj = ds.index(*feature['geometry']['coordinates'])

                        if not all([loci >= 0, loci < height, locj >= 0, locj < width]):
                            continue

                        # connect exterior->inlet

                        i, j = dem.index(*feature['geometry']['coordinates'])
                        area = feature['properties']['AREAKM2'] / 25e-6
                        graph[(-2, i-1, j-1)] = (tile, i, j, area)
                        indegree[(tile, i, j)] += 1

                        # connect inlet->tile outlet

                        locti, loctj = ta.outlet(flow, loci, locj)
                        ti, tj = dem.index(*ds.xy(locti, loctj))

                        if (locti, loctj) == (loci, locj):
                            continue
                        
                        if ti >= 0 and tj >= 0:
                            graph[(tile, i, j)] = (tile, ti, tj, 0)
                            indegree[(tile, ti, tj)] += 1

    dem.close()

    click.secho('Created graph with %d nodes' % len(graph), fg='green')

    return graph, indegree

    # queue = [pixel for pixel in graph if indegree[pixel] == 0]
    # areas = defaultdict(lambda: 0)
    # seen = set()

    # with click.progressbar(length=len(indegree)) as progress:
    
    #     while queue:

    #         tile, i, j = queue.pop(0)

    #         if (tile, i, j) in seen:
    #             continue

    #         progress.update(1)
    #         seen.add((tile, i, j))

    #         if (tile, i, j) in graph:

    #             tile, i, j, area = graph[(tile, i, j)]
    #             areas[(tile, i, j)] += area*25e-6 # convert to km^2
    #             indegree[(tile, i, j)] -= 1

    #             if indegree[(tile, i, j)] == 0:
    #                 queue.append((tile, i, j))

    # return areas


def TileInletAreas(row, col, graph, indegree, areas):
    """
    Output inlet points,
    attributed with the total upstream drained area.
    """

    tile_index = tileindex()
    DEM = filename('dem', 'input')

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('AREAKM2', 'float')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    dem = rio.open(DEM)

    tile = tile_index[(row, col)].gid
    cum_areas = defaultdict(lambda: 0.0)
    count = Counter()

    for key in graph.keys() | indegree.keys():
        if key[0] == tile:
            cum_areas[key[1:]] += areas.get(key[1:], 0)
            count[key[1:]] += 1

    # with fiona.open(inlet_filename(row, col)) as fs:
    #     for feature in fs:
            
    #         i, j = dem.index(*feature['geometry']['coordinates'])
    #         assert((i, j) in index)

    with fiona.open(filename('inlet-areas', row=row, col=col), 'w', **options) as dst:
        for i, j in cum_areas:

            x, y = dem.xy(i, j)
            area = cum_areas[(i, j)]
            geom = {'type': 'Point', 'coordinates': [x, y]}
            props = {'TILE': tile, 'AREAKM2': area}
            feature = {'geometry': geom, 'properties': props}
            dst.write(feature)

    dem.close()


def InletAreas():
    """
    DOCME
    """

    tile_index = tileindex()

    graph, indegree = CreateOutletsGraph()

    click.secho('Accumulate areas', fg='cyan')
    areas, res = speedup.graph_acc(graph)

    click.secho('Write inlet shapefiles', fg='cyan')
    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            TileInletAreas(row, col, graph, indegree, areas)    

def FlowAccumulation(row, col):

    tile_index = tileindex()

    tile = tile_index[(row, col)].gid

    flow_raster = filename('flow', row=row, col=col)
    inlet_shapefile = filename('inlet-areas', row=row, col=col)
    output = filename('acc', row=row, col=col)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        out = np.full_like(flow, 25e-6, dtype=np.float32)
        height, width = flow.shape

        with fiona.open(inlet_shapefile) as fs:
            with click.progressbar(fs) as progress:
                for feature in progress:

                    i, j = ds.index(*feature['geometry']['coordinates'])
                    out[i, j] += feature['properties']['AREAKM2']

        # with fiona.open(filename('exterior-inlets')) as fs:
        #     for feature in fs:
        #         i, j = ds.index(*feature['geometry']['coordinates'])
        #         if all([i >= 0, i < height, j >= 0, j < width]):
        #             out[i, j] += feature['properties']['AREAKM2']

        speedup.flow_accumulation(flow, out)

        click.secho('Save to %s' % output, fg='green')

        profile = ds.profile.copy()
        profile.update(compress='deflate', nodata=0, dtype=np.float32)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def StreamToFeature(row, col, min_drainage):
    """
    DOCME
    """

    flow_raster = filename('flow', row=row, col=col)
    acc_raster = filename('acc', row=row, col=col)
    output = filename('streams', row=row, col=col)

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('GID', 'int'),
            ('HEAD', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)

        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

        with fiona.open(output, 'w', **options ) as dst:

            for current, (segment, head) in enumerate(speedup.stream_to_feature(streams, flow)):

                coords = ta.pixeltoworld(np.fliplr(np.int32(segment)), ds.transform, gdal=False)
                dst.write({
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': coords},
                    'properties': {'GID': current, 'HEAD': 1 if head else 0}
                })

def AggregateStreams():

    tile_index = tileindex()
    output = filename('streams-agg')

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('GID', 'int'),
            ('HEAD', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    gid = itertools.count(1)

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tile_index) as progress:
            for row, col in progress:
                with fiona.open(filename('streams', row=row, col=col)) as fs:
                    for feature in fs:
                        feature['properties']['GID'] = next(gid)
                        dst.write(feature)
