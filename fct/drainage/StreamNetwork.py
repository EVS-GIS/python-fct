"""
Vectorize stream network from drainage rasters

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import defaultdict, Counter
import itertools

import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from .. import speedup
from .. import terrain_analysis as ta
from ..config import (
    config,
    DatasetParameter,
    LiteralParameter
)

# def tileindex():
#     """
#     Return default tileindex
#     """
#     return config.tileset().tileindex

class Parameters():
    """
    Drainage area and network extraction parameters
    """

    elevations = DatasetParameter(
        'filled-resolved elevation raster (DEM)',
        type='input')
    flow = DatasetParameter(
        'flow direction raster',
        type='input')
    acc = DatasetParameter(
        'accumulation raster (drainage area)',
        type='input')
    drainage_network = DatasetParameter(
        'drainage network shapefile',
        type='output')
    min_drainage = LiteralParameter(
        'minimum drainage area for stream extraction expressed in square kilometers')

    def __init__(self):
        """
        Default paramater values
        """

        self.elevations = 'dem'
        self.flow = 'flow'
        self.acc = 'acc'
        self.drainage_network = 'dem-drainage-network' # 'ax_drainage_network' ?
        self.min_drainage = 5.0

def StreamToFeatureTile(row, col, params):
    """
    DOCME
    """

    flow_raster = params.flow.tilename(row=row, col=col)
    # config.tileset().tilename('flow', row=row, col=col)
    acc_raster = params.acc.tilename(row=row, col=col)
    # config.tileset().tilename('acc', row=row, col=col)
    output = params.drainage_network.tilename(row=row, col=col)
    # config.tileset().tilename('dem-drainage-network', row=row, col=col)
    min_drainage = params.min_drainage

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

def AggregateStreams(params):
    """
    Aggregate Streams Shapefile
    """

    # tile_index = tileindex()
    tileset = config.tileset()
    output = params.drainage_network.filename()
    # config.tileset().filename('dem-drainage-network')

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
        with click.progressbar(tileset.tiles(), length=len(tileset)) as iterator:
            for row, col in iterator:
                
                network_tile = params.drainage_network.tilename(row=row, col=col)
                # config.tileset().tilename('dem-drainage-network', row=row, col=col)
                
                with fiona.open(network_tile) as fs:
                    for feature in fs:
                        feature['properties']['GID'] = next(gid)
                        dst.write(feature)

# =============================
# code to be reviewed below ...

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
