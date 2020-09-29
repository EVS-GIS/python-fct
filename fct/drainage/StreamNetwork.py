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
import rasterio as rio
import fiona
import fiona.crs

from .. import speedup
from .. import terrain_analysis as ta
from ..config import config

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

def CreateOutletsGraph(exterior='exterior-inlets'):
    """
    DOCME
    """

    tile_index = tileindex()
    elevation_raster = config.tileset().filename('dem')

    click.secho('Build outlets graph', fg='cyan')

    graph = dict()
    indegree = Counter()

    # provide world/pixel geotransform
    dem = rio.open(elevation_raster)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[(row, col)].gid
            inlet_shapefile = config.tileset().tilename('inlets', row=row, col=col)
            flow_raster = config.tileset().tilename('flow', row=row, col=col)

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

                if exterior and exterior != 'off':

                    with fiona.open(config.datasource(exterior).filename) as fs:
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


def TileInletAreas(tile, keys, areas):
    """
    Output inlet points,
    attributed with the total upstream drained area.
    """

    row = tile.row
    col = tile.col
    gid = tile.gid

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

    # provide world/pixel geotransform
    dem_file = config.tileset().filename('dem')
    dem = rio.open(dem_file)

    cum_areas = defaultdict(lambda: 0.0)

    for key in keys:
        cum_areas[key[1:]] += areas.get(key[1:], 0)

    inlet_shapefile = config.tileset().tilename('inlets', row=row, col=col)
    emitted = set()

    with fiona.open(config.tileset().tilename('inlet-areas', row=row, col=col), 'w', **options) as dst:
        with fiona.open(inlet_shapefile) as fs:
            for feature in fs:

                x, y = feature['geometry']['coordinates']
                i, j = dem.index(x, y)

                if (i, j) in emitted:
                    continue

                assert (i, j) in cum_areas
                area = cum_areas[i, j]

                geom = {'type': 'Point', 'coordinates': [x, y]}
                props = {'TILE': gid, 'AREAKM2': area}
                feature = {'geometry': geom, 'properties': props}
                dst.write(feature)

                emitted.add((i, j))

    dem.close()

def InletAreas(exterior):
    """
    Accumulate areas across tiles
    and output per tile inlet shapefiles
    with contributing area flowing into tile.
    """

    tile_index = tileindex()
    tiles = {tile.gid: tile for tile in tile_index.values()}

    graph, indegree = CreateOutletsGraph(exterior)

    click.secho('Accumulate areas', fg='cyan')
    areas, res = speedup.graph_acc(graph)

    keys = sorted(graph.keys() | indegree.keys(), key=itemgetter(0))
    groups = itertools.groupby(keys, key=itemgetter(0))

    click.secho('Write inlet shapefiles', fg='cyan')
    with click.progressbar(groups, length=len(tile_index)) as progress:
        for tile_gid, keys in progress:

            if tile_gid in tiles:
                tile = tiles[tile_gid]
                TileInletAreas(tile, keys, areas)

def FlowAccumulation(row, col, overwrite):

    tile_index = tileindex()

    tile = tile_index[(row, col)].gid

    flow_raster = config.tileset().tilename('flow', row=row, col=col)
    inlet_shapefile = config.tileset().tilename('inlet-areas', row=row, col=col)
    output = config.tileset().tilename('acc', row=row, col=col)

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        out = np.full_like(flow, 25e-6, dtype='float32')
        height, width = flow.shape

        if os.path.exists(inlet_shapefile):
            with fiona.open(inlet_shapefile) as fs:
                for feature in fs:

                    i, j = ds.index(*feature['geometry']['coordinates'])
                    out[i, j] += feature['properties']['AREAKM2']

        # with fiona.open(filename('exterior-inlets')) as fs:
        #     for feature in fs:
        #         i, j = ds.index(*feature['geometry']['coordinates'])
        #         if all([i >= 0, i < height, j >= 0, j < width]):
        #             out[i, j] += feature['properties']['AREAKM2']

        speedup.flow_accumulation(flow, out)

        # click.secho('Save to %s' % output, fg='green')

        if os.path.exists(inlet_shapefile):
            with fiona.open(inlet_shapefile) as fs:
                for feature in fs:

                    x, y = feature['geometry']['coordinates']
                    i, j = ds.index(x, y)
                    out[i, j] = feature['properties']['AREAKM2']

        profile = ds.profile.copy()
        profile.update(compress='deflate', nodata=0, dtype=np.float32)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def StreamToFeature(row, col, min_drainage):
    """
    DOCME
    """

    flow_raster = config.tileset().tilename('flow', row=row, col=col)
    acc_raster = config.tileset().tilename('acc', row=row, col=col)
    output = config.tileset().tilename('dem-drainage-network', row=row, col=col)

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
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)

        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

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

def NoFlowPixels(row, col, min_drainage):

    flow_raster = config.tileset().tilename('flow', row=row, col=col)
    acc_raster = config.tileset().tilename('acc', row=row, col=col)
    output = config.tileset().tilename('noflow', row=row, col=col)

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('ROW', 'int:4'),
            ('COL', 'int:4')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)

        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

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

def AggregateNoFlowPixels():
    """
    Aggregate No Flow Shapefiles
    """

    tile_index = tileindex()
    output = config.tileset().filename('noflow')

    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('GID', 'int'),
            ('ROW', 'int'),
            ('COL', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    gid = itertools.count(1)

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tile_index) as progress:
            for row, col in progress:
                with fiona.open(config.tileset().tilename('noflow', row=row, col=col)) as fs:
                    for feature in fs:
                        feature['properties']['GID'] = next(gid)
                        dst.write(feature)

    count = next(gid) - 1
    click.secho('Found %d not-flowing stream nodes' % count, fg='cyan')


def AggregateStreams():
    """
    Aggregate Streams Shapefile
    """

    tile_index = tileindex()
    output = config.tileset().filename('dem-drainage-network')

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
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    gid = itertools.count(1)

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tile_index) as progress:
            for row, col in progress:
                with fiona.open(config.tileset().tilename('dem-drainage-network', row=row, col=col)) as fs:
                    for feature in fs:
                        feature['properties']['GID'] = next(gid)
                        dst.write(feature)

def AggregateStreamSegments():
    """
    DOCME
    """

    source = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHT_RGE5M.gpkg'
    layer = 'RHT_RGE5M_ALL'
    output = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHT_RGE5M_REGROUP.shp'

    graph = dict()
    indegree = Counter()

    with fiona.open(source, layer=layer) as fs:

        length = len(fs)

        with click.progressbar(fs) as processing:
            for feature in processing:

                a = feature['properties']['NODEA']
                b = feature['properties']['NODEB']
                axis = feature['properties']['AXIS']
                row = feature['properties']['ROW']
                col = feature['properties']['COL']

                graph[a] = [b, (axis, row, col), feature['id']]
                indegree[b] += 1

        # schema['properties'].update(GID='int')
        driver = 'ESRI Shapefile'
        options = dict(driver=driver, crs=fs.crs, schema=fs.schema)

        def getgroup(node):

            if node in graph:
                return graph[node][1]

            return None

        def group_iterator():

            stack = [node for node in graph if indegree[node] == 0]
            processed = set()

            while stack:

                node = stack.pop(0)

                if not node in graph:
                    continue

                if node in processed:
                    continue

                processed.add(node)
                current_nodes = [node]

                # move to downstream node
                node, current_group, _ = graph[node]

                while getgroup(node) == current_group:

                    current_nodes.append(node)
                    indegree[node] -= 1
                    processed.add(node)

                    # move to downstream node
                    node = graph[node][0]

                # add terminal node to current track
                current_nodes.append(node)
                yield current_group, current_nodes

                # append node to stack for further processing
                # in case we reached a confluence
                # or a split between two tiles

                indegree[node] -= 1
                if indegree[node] == 0:
                    stack.append(node)

        def group_iterator_reverse():
            """
            Would it be easier to write the iterator in the reverse order ?
            """

            reverse_graph = defaultdict(list)
            outdegree = Counter()

            for a, (b, group, fid) in graph.items():
                reverse_graph[b].append((a, group, fid))
                outdegree[a] += 1

            stack = list()
            seen_edges = set()

            for node in reverse_graph:
                if outdegree[node] == 0:
                    for upstream, group, _ in reverse_graph[node]:
                        stack.append((node, group))

            while stack:

                node, current_group = stack.pop()
                current_nodes = [node]

                while node is not None:

                    next_node = None

                    for upstream, group, _ in reverse_graph[node]:
 
                        if group == current_group:
                            current_nodes.append(upstream)
                            seen_edges.add((upstream, group))
                            next_node = upstream
                        else:
                            if (node, group) not in seen_edges:
                                stack.append((node, group))
                                seen_edges.add((node, group))

                    node = next_node

                yield current_group, list(reversed(current_nodes))

        feature_count = 0

        with fiona.open(output, 'w', **options) as dst:

            with click.progressbar(group_iterator(), length=length) as processing:
                for current, (group, nodes) in enumerate(processing):

                    # axis, row, col = group
                    a = nodes[0]
                    b = nodes[-1]
                    coordinates = list()

                    _, _, fid = graph[a]
                    feature = fs.get(fid)
                    properties = feature['properties']
                    coordinates = feature['geometry']['coordinates']
                    feature_count += 1

                    for node in nodes[1:-1]:

                        _, _, fid = graph[node]
                        feature = fs.get(fid)
                        coordinates.extend(feature['geometry']['coordinates'][1:])
                        feature_count += 1

                        # try:
                        #     _, _, _, segment = graph[node]
                        #     coordinates.extend(segment[1:])
                        # except KeyError:
                        #     pass

                    geometry = {
                        'type': 'LineString',
                        'coordinates': coordinates
                    }

                    properties.update(GID=current, NODEB=b)
                    feature = dict(geometry=geometry, properties=properties)

                    dst.write(feature)
                    processing.update(len(nodes)-1)

    assert(feature_count == length)

def VerifyAggregateSegments():

    output = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHT_RGE5M_REGROUP.shp'
    counts = Counter()

    with fiona.open(output) as fs:
        with click.progressbar(fs) as processing:
            for feature in processing:
                axis = feature['properties']['AXIS']
                row = feature['properties']['ROW']
                col = feature['properties']['COL']
                group = (axis, row, col)
                counts[group] += 1

    many = [group for group in counts if counts[group] > 1]

    print(len(many))
