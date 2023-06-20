# coding: utf-8

"""
DOCME

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import Counter, defaultdict
import itertools
from heapq import heappop, heapify

import click
import fiona
import fiona.crs
from shapely.geometry import shape, Point
from shapely.ops import linemerge
from rtree import index

from ..config import config, DatasetParameter, LiteralParameter

class Parameters:
    """
    Prepare hydrologic network
    """
    network_identified = DatasetParameter('Theoric stream network with identified nodes', type='input')
    network_nodes = DatasetParameter('Theoric stream network nodes', type='input')
    sources_confluences = DatasetParameter('sources and confluences extracted from hydrologic network input', type='input')
    rhts = DatasetParameter('Theoric stream network with identified nodes and joined data from input network', type='output')


    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.network_identified_strahler = 'network-identified'
        self.network_nodes = 'network-nodes'
        self.sources_confluences = 'sources-confluences'
        self.rhts = 'rhts'

def create_index(points_list):
    # Create a spatial index
    idx = index.Index()

    # Insert points into the spatial index
    for i, point in enumerate(points_list):
        geom = point['geometry']
        idx.insert(i, geom.bounds)

    return idx

def find_nearest_point(target_point, idx, points_list):

    # Find the nearest point using the spatial index
    nearest_index = min(idx.nearest(target_point.bounds), key=lambda i: target_point.distance(points_list[i]))

    # Return the nearest point
    return points_list[nearest_index]


def JoinNetworkAttributes(params, tileset = 'default'):

    sources_shapefile = params.sources_confluences.filename(tileset=None)
    network_identified = params.network_identified.filename(tileset=tileset)
    network_nodes = params.network_nodes.filename(tileset=tileset)
    rhts = params.rhts.filename(tileset=tileset)

    strahler_field_name = 'strahler'

    # get first point coordinates from line with strahler == 1
    points_strahler_1 = []

    with fiona.open(network_nodes, 'r') as nodes:
        for point in nodes:
            if point['properties'][strahler_field_name] == 1:
                points_strahler_1.append(point)

    idx = create_index(points_strahler_1)

    # # Create an R-tree index
    # idx = index.Index()
    # # populate index
    # for fid, feature in enumerate(source):
    #     geom = shape(feature['geometry'])
    #     idx.insert(fid, geom.bounds)

    # select 
    with fiona.open(sources_shapefile, 'r') as source:

        schema = source.schema.copy

        node_id_name = 'NODE'
        hydro_id_name = 'CDENTITEHY'
        toponym_name = 'TOPONYME'
        axis_name = 'AXIS'
        hack_name = 'HACK'

        # Add the new field to the schema
        if not node_id_name in schema :
            schema['properties'][node_id_name] = 'int'
        if not axis_name in schema :
            schema['properties'][axis_name] = 'int' 
        if not hydro_id_name in schema :
                schema['properties'][hydro_id_name] = 'str' 
        if not toponym_name in schema :
            schema['properties'][toponym_name] = 'str' 
        if not hack_name in schema :
            schema['properties'][hack_name] = 'int'

        options = dict(
            driver=source.driver,
            schema=schema,
            crs=source.crs)

        # with fiona.open(rhts, 'w', **options) as dst:
            
        for feature in source:
            if feature['properties'][strahler_field_name] == 1:
                target_point = feature['geometry']
                nearest_point = find_nearest_point(target_point, idx, points_strahler_1)
            print(nearest_point)
                    

def UpdateLengthOrder(
        network_shapefile,
        output):
    """
    Update HACK et LENAXIS fields
    according to network connectivity and AXIS identifier

    Parameters
    ----------

    joined: str, logical name

        LineString dataset,
        output of procedure JoinNetworkAttributes()

    destination: str, logical name
        
        Output dataset
    """

    # network_shapefile = config.tileset().filename(joined)
    # output = config.tileset().filename(destination)

    # network_shapefile = config.filename(joined)
    # output = config.filename(destination)

    graph = dict()
    indegree = Counter()
    lengths = defaultdict(lambda: 0.0)
    newids = dict()
    gid = itertools.count(1)

    with fiona.open(network_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                axis = properties['AXIS']

                a = properties['NODEA']
                b = properties['NODEB']

                geometry = shape(feature['geometry'])
                lengths[axis] += geometry.length

                graph[a] = (b, axis)
                indegree[b] += 1

    def distance(node):

        if node in graph:
            _, axis = graph[node]
            return lengths[axis]

        return 0.0

    orders = dict()
    queue = [(-distance(node), node) for node in graph if indegree[node] == 0]
    heapify(queue)

    def set_order(nodes, k):

        for node in nodes:
            orders[node] = k

    while queue:

        _, node = heappop(queue)
        track = [node]

        while node in graph:

            next_node, axis = graph[node]

            if axis not in newids:
                newids[axis] = next(gid)

            if next_node in orders:
                set_order(track, orders[next_node] + 1)
                break

            track.append(next_node)
            node = next_node

        else:

            set_order(track, 1)

    with fiona.open(network_shapefile) as fs:

        # driver = fs.driver
        driver = 'ESRI Shapefile'
        schema = fs.schema
        crs = fs.crs

        schema['properties'].update({
            'HACK': 'int:3',
            'LENAXIS': 'float:8.0',
        })

        options = dict(driver=driver, crs=crs, schema=schema)

        with fiona.open(output, 'w', **options) as dst:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    properties = feature['properties']
                    a = properties['NODEA']
                    axis = properties['AXIS']

                    properties.update({
                        'AXIS': newids[axis],
                        'HACK': orders[a] if a in orders else None,
                        'LENAXIS': lengths[axis]
                    })

                    dst.write(feature)

def AggregateByAxis(network_shapefile, output):

    graph = dict()
    indegree = Counter()
    segments = defaultdict(list)

    with fiona.open(network_shapefile) as fs:
        
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                axis = properties['AXIS']

                a = properties['NODEA']
                b = properties['NODEB']

                segments[axis].append(feature['id'])

                graph[a] = (b, axis, feature['id'])
                indegree[b] += 1

        sources = [
            node for node in graph
            if indegree[node] == 0
        ]

        options = dict(
            driver=fs.driver,
            crs=fs.crs,
            schema=fs.schema
        )

        def get_nodeb(a, axis):

            node = a

            while node in graph:

                next_node, next_axis, _ = graph[node]

                if next_axis != axis:
                    break

                node = next_node

            return node
        
        with fiona.open(output, 'w', **options) as dst:
            with click.progressbar(sources) as iterator:
                for source in iterator:

                    _, axis, fid = graph[source]
                    feature = fs.get(fid)
                    properties = feature['properties']

                    geometry = linemerge([
                        shape(fs.get(fid)['geometry'])
                        for fid in segments[axis]
                    ])

                    a = properties['NODEA']
                    b = get_nodeb(a, axis)
                    properties.update({'NODEB': b})

                    feature['geometry']['coordinates'] = list(geometry.coords)
                    
                    dst.write(feature)
