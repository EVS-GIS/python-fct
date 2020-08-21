# coding: utf-8

"""
DOCME

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
from collections import Counter
import click

import fiona
import fiona.crs
from ..config import config

def AggregateSegmentsByAxisAndTile(max_length=10e3):

    source = config.filename('streams') # filename ok ?
    output = config.tileset().filename('streams-tiled')

    graph = dict()
    indegree = Counter()

    with fiona.open(source) as fs:

        length = 0

        with click.progressbar(fs) as processing:
            for feature in processing:

                a = feature['properties']['NODEA']
                b = feature['properties']['NODEB']
                axis = feature['properties']['AXIS']
                axis_length = feature['properties']['LENAXIS']
                row = feature['properties']['ROW']
                col = feature['properties']['COL']

                if axis_length >= max_length:

                    graph[a] = [b, (axis, row, col), feature['id']]
                    indegree[b] += 1
                    length += 1

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

                    if node in processed:
                        break

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

        feature_count = 0

        with fiona.open(output, 'w', **options) as dst:

            with click.progressbar(group_iterator(), length=length) as iterator:
                for current, (group, nodes) in enumerate(iterator):

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
