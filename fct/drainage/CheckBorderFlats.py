# coding: utf-8

"""
Unused tile-border flat spillover resolution code

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import defaultdict
import numpy as np

import fiona
import fiona.crs

from ..config import config

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

def CheckConnectedFlats(directed, graph, graph_index, epsilon=0.001):

    extra_links = dict()
    exterior = (-1, 1)
    fixed = 0

    def isupstream(w1, w2):
        """
        Check whether watershed `w1` is upstream of `w2`
        with respect to uptream-dowstream graph `directed`
        """

        current = w1

        while current in directed:

            current = directed[current][0]
            if current == w2:
                return True

        return False

    for w1, w2 in graph:

        if w1 == exterior:
            continue

        # minz = graph[w1, w2]
        d1, z1 = directed[w1]
        d2, z2 = directed[w2]

        if abs(z1 - z2) < epsilon:

            minz = max(z1, z2) + epsilon

            if isupstream(w1, w2):

                for link in graph_index[w1]:
                    v1, v2 = sorted([w1, link])
                    graph[(v1, v2)] = max(graph[(v1, v2)], minz)

                fixed += 1

            elif isupstream(w2, w1):

                for link in graph_index[w2]:
                    v1, v2 = sorted([w2, link])
                    graph[(v1, v2)] = max(graph[(v1, v2)], minz)

                fixed += 1

            else:

                if (w1, w2) in extra_links:
                    extra_links[w1, w2] = max(extra_links[w1, w2], minz)
                else:
                    extra_links[w1, w2] = minz

    return fixed, extra_links

def CheckBorderFlats(directed, graph, row, col, dlinks, ulinks, epsilon=0.001):

    tile_index = tileindex()
    tile = tile_index[(row, col)].gid

    graph_index = defaultdict(list)
    for l1, l2 in graph.keys():
        graph_index[l1].append(l2)
        graph_index[l2].append(l1)

    dlinks = dict()
    ulinks = dict()

    def read_data(i, j):
        return np.load(config.filename('graph', row=i, col=j), allow_pickle=True)

    def isupstream(origin, target):

        current = origin
        while current in directed:
            current = directed[current][0]
            if current == target:
                return True
        return False

    data = read_data(row, col)

    def inspectborder(di, dj, side):

        if (row+di, col+dj) not in tile_index:
            return

        other_side = (side + 2) % 4
        other_data = read_data(row+di, col+dj)

        other_tile = tile_index[(row+di, col+dj)].gid
        elevations = data['z'][side]
        other_elevations = np.flip(other_data['z'][other_side])
        labels = data['labels'][side]
        other_labels = np.flip(other_data['labels'][other_side])
        width = len(elevations)

        for k in range(width):

            label = labels[k]
            watershed = (tile, label)
            downstream, minz = directed[watershed]

            for s in range(-1, 2):

                if (k+s) < 0 or (k+s) >= width:
                    continue

                other_watershed = (other_tile, other_labels[k+s])
                other_downstream, other_minz = directed[other_watershed]
                other_z = max(other_elevations[k+s], other_minz)

                if abs(other_z - minz) < epsilon:

                    if isupstream(watershed, other_watershed):

                        # for link in graph_index[watershed]:
                        #     v1, v2 = sorted([watershed, link])
                        #     graph[(v1, v2)] = max(graph[(v1, v2)], minz+epsilon)

                        # fixed += 1

                        if (watershed, other_watershed) in dlinks:
                            minz = max(minz+epsilon, dlinks[watershed, other_watershed])
                        else:
                            minz = minz+epsilon

                        dlinks[watershed, other_watershed] = minz

                    elif not isupstream(other_watershed, watershed):

                        w1, w2 = sorted([watershed, other_watershed])

                        if (w1, w2) in ulinks:
                            ulinks[w1, w2] = max(ulinks[w1, w2], minz, other_z)
                        else:
                            ulinks[w1, w2] = max(minz, other_z)

    def inspectcorner(di, dj, side):

        if (row+di, col+dj) not in tile_index:
            return

        other_side = (side + 2) % 4
        other_data = read_data(row+di, col+dj)

        other_tile = tile_index[(row+di, col+dj)].gid
        elevations = data['z'][side]
        other_elevations = other_data['z'][other_side]
        labels = data['labels'][side]
        other_labels = other_data['labels'][other_side]

        label = labels[0]
        watershed = (tile, label)
        other_watershed = (other_tile, other_labels[0])
        downstream, minz = directed[watershed]
        other_downstream, other_minz = directed[other_watershed]
        other_z = max(other_elevations[0], other_minz)

        if abs(other_z - minz) < epsilon:

            if isupstream(watershed, other_watershed):

                # for link in graph_index[watershed]:
                #     v1, v2 = sorted([watershed, link])
                #     graph[(v1, v2)] = max(graph[(v1, v2)], minz+epsilon)

                # fixed += 1

                if (watershed, other_watershed) in dlinks:
                    minz = max(minz+epsilon, dlinks[watershed, other_watershed])
                else:
                    minz = minz+epsilon

                dlinks[watershed, other_watershed] = minz

            elif not isupstream(other_watershed, watershed):

                w1, w2 = sorted([watershed, other_watershed])

                if (w1, w2) in ulinks:
                    ulinks[w1, w2] = max(ulinks[w1, w2], minz, other_z)
                else:
                    ulinks[w1, w2] = max(minz, other_z)

    inspectborder(-1, 0, 0) # top
    inspectborder(0, 1, 1) # right
    inspectborder(1, 0, 2) # bottom
    inspectborder(0, -1, 3) # left

    inspectcorner(-1, -1, 0) # top-left
    inspectcorner(-1, 1, 1) # top-right
    inspectcorner(1, 1, 2) # bottom-right
    inspectcorner(1, -1, 3) # bottom-left

    return dlinks, ulinks

def ResolveFlatLinks(directed, dlinks, ulinks, areas):

    # Build reverse (downstream to upstream) multi-graph
    graph = defaultdict(list)

    for watershed in directed:
        downstream, z = directed[watershed]
        graph[downstream].append((watershed, z))

    for upstream, downstream in dlinks:
        z = dlinks[upstream, downstream]
        graph[downstream].append((upstream, z))

    for w1, w2 in ulinks:

        z = ulinks[w1, w2]
        area1 = areas[w1]
        area2 = areas[w2]

        if area2 > area1:
            w1, w2 = w2, w1
            graph[w1].append((w2, z))

    # First pass :
    # Calculate minimum z for each watershed

    exterior = (-1, 1)
    queue = [exterior]
    resolved = dict()

    while queue:

        current = queue.pop(0)

        for upstream, z in graph[current]:

            if upstream in resolved:
                u, zu = resolved[upstream]
                if zu >= z:
                    continue

            resolved[upstream] = (current, z)
            queue.append(upstream)

    # Second pass
    # ensure epsilon gradient between watersheds

    graph = defaultdict(list)

    for watershed in resolved:
        downstream, z  = resolved[watershed]
        graph[downstream].append((watershed, z))

    queue = [(exterior, None)]
    epsilon = 0.001

    while queue:

        current, current_z = queue.pop(0)

        for upstream, z in graph[current]:

            if current_z is not None and (z - current_z) < epsilon:
                z = current_z + epsilon

            resolved[upstream] = (current, z)
            queue.append((upstream, z))

    return resolved
