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
    network_identified_strahler = DatasetParameter('Theoric stream network with identified nodes', type='input')
    sources_confluences = DatasetParameter('sources and confluences extracted from hydrologic network input', type='input')
    rhts = DatasetParameter('Theoric stream network with identified nodes and joined data from input network', type='output')


    def __init__(self, axis=None):
        """
        Default parameter values
        """
        self.network_identified_strahler = 'network-identified-strahler'
        self.sources_confluences = 'sources-confluences'
        self.rhts = 'rhts'

# work from python-fct not finish (for memory)
def JoinNetworkAttributes(params, distance_threshold_strahler1 = 50, distance_threshold_other_strahler = 2000, tileset = 'default'):

    sources_confluences = params.sources_confluences.filename(tileset=None)
    network_identified_strahler = params.network_identified_strahler.filename(tileset=tileset)
    rhts = params.rhts.filename(tileset=tileset)

    strahler_field_name = 'strahler'
    node_field_name = 'NODE'
    axis_field_name = 'AXIS'
    cdidentity_field_name= 'CDENTITEHY'
    toponyme_field_name = 'TOPONYME'
    hack_field_name = 'HACK'

    # get the fields in the schema from sources_confluences not existing in network_identified_strahler 
    with fiona.open(sources_confluences, 'r') as source:
        with fiona.open(network_identified_strahler, 'r') as network:
            schema_source_prop = source.schema['properties']
            new_schema_network = network.schema.copy()
            new_schema_network['properties'][node_field_name] = schema_source_prop[node_field_name]
            new_schema_network['properties'][axis_field_name] = schema_source_prop[axis_field_name]
            new_schema_network['properties'][cdidentity_field_name] = schema_source_prop[cdidentity_field_name]
            new_schema_network['properties'][toponyme_field_name] = schema_source_prop[toponyme_field_name]
            new_schema_network['properties'][hack_field_name] = schema_source_prop[hack_field_name]

            options = dict(
            schema = new_schema_network,
            driver = network.driver,
            crs = network.crs
            )

            max_strahler = max([feature['properties'][strahler_field_name] for feature in network])
            print(max_strahler)

    network_update = []

    for strahler in range(1, max_strahler+1):
        if strahler == 1:
            distance_threshold = distance_threshold_strahler1
        else:
            distance_threshold = distance_threshold_other_strahler
        print(strahler)
        with fiona.open(network_identified_strahler, 'r') as network:
            print(fiona.model.to_dict(network))

        # get sources_confluences with strahler == 1
        with fiona.open(sources_confluences, 'r') as source:
            strahler1 = [feature for feature in source if feature['properties'][strahler_field_name] == strahler]
        
        with fiona.open(network_identified_strahler, 'r') as network:
            network1 = [feature for feature in network if feature['properties'][strahler_field_name] == strahler]

        # Create an R-tree index
        strahler1_index = index.Index()
        # populate index
        for i, point in enumerate(strahler1):
            geom = shape(point['geometry'])
            strahler1_index.insert(i, geom.bounds)

        for line in network1:
            # initialisation to get nearest_point
            nearest_point = None
            nearest_distance = float('inf')
            line_properties = line['properties']
            geometry = shape(line['geometry'])
            # get line first point coordinates
            first_point = Point(geometry.coords[0])
            # create buffer around first point
            point_buffer = first_point.buffer(distance_threshold)
            # get all the point from source that intersect with buffer with index
            potential_matches = [idx for idx in strahler1_index.intersection(point_buffer.bounds)] # store idx of the index
            
            if potential_matches:
                # seach with index in potential_matches
                for idx in potential_matches:
                    source_potential_geom = shape(strahler1[idx]['geometry'])
                    # iterate through potential_matches strahler1, calculate distance and get the shortest by updating nearest_distance if the current strahler1 is closer
                    if first_point.distance(source_potential_geom) < nearest_distance:
                        nearest_point = strahler1[idx]
                        nearest_distance = first_point.distance(source_potential_geom)
                nearest_point_properties = nearest_point['properties']
                line_properties[node_field_name] = nearest_point_properties[node_field_name]
                line_properties[axis_field_name] = nearest_point_properties[axis_field_name]
                line_properties[cdidentity_field_name] = nearest_point_properties[cdidentity_field_name]
                line_properties[toponyme_field_name] = nearest_point_properties[toponyme_field_name]
                line_properties[hack_field_name] = nearest_point_properties[hack_field_name]
                rhts_feature = {
                        'type': 'Feature',
                        'properties': line_properties,
                        'geometry': line['geometry'],
                    }
                network_update.append(rhts_feature)

    with fiona.open(rhts, 'w', **options) as output:
        for feature in network_update:
            output.write(feature)








    # with fiona.open(network_identified_strahler, 'r') as network:

    #     options = dict(
    #         schema = new_schema_network,
    #         driver = network.driver,
    #         crs = network.crs
    #     )
    #     with fiona.open(rhts, 'w', **options) as output:
        
    #         for line in network:
    #             # initialisation to get nearest_point
    #             nearest_point = None
    #             nearest_distance = float('inf')
    #             line_properties = line['properties']
    #             # get lione with strahler == 1
    #             if line['properties'][strahler_field_name] == 1:
    #                 geometry = shape(line['geometry'])
    #                 # get line first point coordinates
    #                 first_point = Point(geometry.coords[0])
    #                 # create buffer around first point
    #                 point_buffer = first_point.buffer(distance_threshold_strahler1)
    #                 # get all the point from source that intersect with buffer with index
    #                 potential_matches = [idx for idx in strahler1_index.intersection(point_buffer.bounds)] # store idx of the index
                    
    #                 if potential_matches:
    #                     # seach with index in potential_matches
    #                     for idx in potential_matches:
    #                         source_potential_geom = shape(strahler1[idx]['geometry'])
    #                         # iterate through potential_matches strahler1, calculate distance and get the shortest by updating nearest_distance if the current strahler1 is closer
    #                         if first_point.distance(source_potential_geom) < nearest_distance:
    #                             nearest_point = strahler1[idx]
    #                             nearest_distance = first_point.distance(source_potential_geom)
    #                     nearest_point_properties = nearest_point['properties']
    #                     line_properties[node_field_name] = nearest_point_properties[node_field_name]
    #                     line_properties[axis_field_name] = nearest_point_properties[axis_field_name]
    #                     line_properties[cdidentity_field_name] = nearest_point_properties[cdidentity_field_name]
    #                     line_properties[toponyme_field_name] = nearest_point_properties[toponyme_field_name]
    #                     line_properties[hack_field_name] = nearest_point_properties[hack_field_name]
    #                     rhts_feature = {
    #                                     'type': 'Feature',
    #                                     'properties': line_properties,
    #                                     'geometry': line['geometry'],
    #                                 }
    #                     output.write(rhts_feature)
                    

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
