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

from collections import Counter, defaultdict
import itertools
from heapq import heappop, heapify

import click
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..config import config

def JoinNetworkAttributes():

    # sources_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS_SOURCES.shp'
    # network_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS.shp'
    # output = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS_ATTR.shp'

    sources_shapefile = config.filename('sources')
    network_shapefile = config.filename('streams-attr')
    output = config.filename('streams-attr-sources')

    graph = dict()
    rgraph = defaultdict(list)
    indegree = Counter()
    axis_increment = 0


    with fiona.open(network_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                fid = feature['id']
                properties = feature['properties']
                a = properties['NODEA']
                b = properties['NODEB']

                graph[a] = (b, fid)
                indegree[b] += 1

    sources = dict()

    with fiona.open(sources_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                node = properties['NODE']
                axis = properties['AXIS']
                axis_increment = max(axis, axis_increment)
                sources[node] = properties

    def greater(c1, h1, c2, h2):

        if c2 is None or h2 is None:
            return True

        if c1 is None or h1 is None:
            return False

        if c1 == c2:
            return h1 < h2

        if c1.count('-') > c2.count('-'):
            return True

        if c1.count('-') == c2.count('-'):

            if c1[0] == 'V' and c2[0] != 'V':
                return True

            if c2[0] == 'V' and c1[0] != 'V':
                return False

            return c1 < c2

        return False

    def resolve_properties(node):

        if node in sources:
            return sources[node]

        if len(rgraph[node]) == 1:
            return rgraph[node][0][1]

        _cdentite = None
        _hack = None
        _properties = None

        for _, properties in rgraph[node]:

            if properties is None:
                continue

            cdentite = properties['CDENTITEHY']
            hack = properties['HACK']

            if greater(cdentite, hack, _cdentite, _hack):

                _cdentite, _hack = cdentite, hack
                _properties = properties

        if _properties is None:
            _properties = {
                'AXIS': next(axis_increment),
                'CDENTITEHY': None,
                'TOPONYME': None,
                'HACK': None
            }

        return _properties

    axis_increment = itertools.count(axis_increment+1)
    queue = [node for node in graph if indegree[node] == 0]
    features = dict()

    while queue:

        node = queue.pop()
        properties = resolve_properties(node)

        if node in graph:

            next_node, fid = graph[node]
            features[fid] = properties
            rgraph[next_node].append((node, properties))

            indegree[next_node] -= 1

            if indegree[next_node] == 0:
                queue.append(next_node)

    with fiona.open(network_shapefile) as fs:

        driver = fs.driver
        schema = fs.schema
        crs = fs.crs

        schema['properties'].update({
            'CDENTITEHY': 'str:8',
            'TOPONYME': 'str:254',
            'AXIS': 'int'
        })

        options = dict(driver=driver, crs=crs, schema=schema)

        with fiona.open(output, 'w', **options) as dst:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    fid = feature['id']

                    if fid not in features:
                        continue

                    properties = features[fid]

                    if properties is None:
                        feature['properties'].update({
                            'CDENTITEHY': None,
                            'TOPONYME': None,
                            'AXIS': None
                        })
                    else:
                        feature['properties'].update({
                            k: properties[k] for k in ('CDENTITEHY', 'TOPONYME', 'AXIS')
                        })

                    dst.write(feature)

def UpdateLengthOrder():

    # network_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS_ATTR.shp'
    # output = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/RHTS_HACK.shp'

    network_shapefile = config.filename('streams-attr-sources')
    output = config.filename('streams')

    graph = dict()
    indegree = Counter()
    lengths = defaultdict(lambda: 0.0)

    with fiona.open(network_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                geometry = asShape(feature['geometry'])
                axis = properties['AXIS']
                lengths[axis] += geometry.length

                a = properties['NODEA']
                b = properties['NODEB']
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

            if next_node in orders:
                set_order(track, orders[next_node] + 1)
                break

            track.append(next_node)
            node = next_node

        else:

            set_order(track, 1)

    with fiona.open(network_shapefile) as fs:

        driver = fs.driver
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
                        'HACK': orders[a] if a in orders else None,
                        'LENAXIS': lengths[axis]
                    })

                    dst.write(feature)
