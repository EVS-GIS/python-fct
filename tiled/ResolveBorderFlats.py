#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict, Counter
from functools import wraps
import numpy as np
import click
from heapq import heappush, heappop

import rasterio as rio
import fiona
import terrain_analysis as ta
import speedup

from areas import WatershedUnitAreas, WatershedCumulativeAreas
from config import tileindex, filename

def LabelBorderFlats(row, col, **kwargs):
    """
    DOCME
    """

    # elevation_raster = filename('filled', row=row, col=col)
    label_raster = filename('labels', row=row, col=col)

    # with rio.open(elevation_raster) as ds:

    elevations = PadElevations(row, col)
    flow = ta.flowdir(elevations)
    flat_labels = speedup.borderflat_labels(flow, elevations)

    with rio.open(label_raster) as ds:

        labels = ds.read(1)
        elevations = elevations[1:-1, 1:-1]
        flat_labels = flat_labels[1:-1, 1:-1]
        flatmask = (flat_labels > 0)
        flatindex = np.max(labels)
        labels[flatmask] = flatindex + flat_labels[flatmask]
        graph = speedup.label_graph(labels, flow, elevations)

        profile = ds.profile.copy()
        output = filename('flat_labels', row=row, col=col)
        with rio.open(output, 'w', **profile) as dst:
            dst.write(labels, 1)

        output = filename('flat_graph', row=row, col=col)
        np.savez(
            output,
            z=np.array([
                elevations[0, :],
                elevations[:, -1],
                np.flip(elevations[-1, :], axis=0),
                np.flip(elevations[:, 0], axis=0)]),
            labels=np.array([
                labels[0, :],
                labels[:, -1],
                np.flip(labels[-1, :], axis=0),
                np.flip(labels[:, 0], axis=0)]),
            flatindex=flatindex,
            graph=np.array(list(graph.items()))
        )

def ConnectTiles(row, col, **kwargs):
    """
    DOCME
    """

    TOP = 0
    RIGHT = 1
    BOTTOM = 2
    LEFT = 3

    tile_index = tileindex()
    tile = tile_index[row, col]
    exterior = (-1, 1)

    def read_data(i, j):
        return np.load(filename("flat_graph", row=i, col=j), allow_pickle=True)

    data = read_data(row, col)
    graph = dict()

    for (label1, label2), z in data['graph']:

        if int(label2) == 0:

            w1 = (tile.gid, int(label1))
            graph[w1, exterior] = z

        else:
    
            w1 = (tile.gid, int(label1))
            w2 = (tile.gid, int(label2))
            graph[w1, w2] = z

    def connect_undirected(w1, w2, z):

        if (w1, w2) in graph:
            graph[w1, w2] = min(z, graph[w1, w2])
        else:
            graph[w1, w2] = z

        if (w2, w1) in graph:
            graph[w2, w1] = min(z, graph[w2, w1])
        else:
            graph[w2, w1] = z

    def connect_side(di, dj, side):
        
        if (row+di, col+dj) not in tile_index:
            return

        other_tile = tile_index[row+di, col+dj]
        other_side = (side + 2) % 4
        other_data = read_data(row+di, col+dj)

        labels = data['labels'][side]
        other_labels = np.flip(other_data['labels'][other_side])
        elevations = data['z'][side]
        other_elevations = np.flip(other_data['z'][other_side])
        width = len(labels)

        for k in range(width):

            label = labels[k]
            z = elevations[k]
            watershed = (tile.gid, label)
            
            for s in range(-1:2):

                other_label = other_labels[k+s]
                other_z = other_elevations[k+s]
                other_watershed = (other_tile.gid, other_label)
                linkz = max(z, other_z)

                connect_undirected(watershed, other_watershed, linkz)

    def connect_corner(di, dj, side):
        
        if (row+di, col+dj) not in tile_index:
            return

        other_tile = tile_index[row+di, col+dj]
        other_side = (side + 2) % 4
        other_data = read_data(row+di, col+dj)

        labels = data['labels'][side]
        other_labels = other_data['labels'][other_side]
        elevations = data['z'][side]
        other_elevations = other_data['z'][other_side]

        label = labels[0]
        z = elevations[0]
        watershed = (tile.gid, label)

        other_label = other_labels[0]
        other_z = other_elevations[0]
        other_watershed = (other_tile.gid, other_label)
        linkz = max(z, other_z)

        connect_undirected(watershed, other_watershed, linkz)

    connect_side(-1, 0, TOP)
    connect_side(0, -1, LEFT)
    connect_corner(-1, -1, TOP)
    connect_corner(-1, 1, RIGHT)
    connect_corner(1, -1, LEFT)

    return graph

def BuildFlatSpilloverGraph():
    """
    DOCME
    """

    tile_index = tileindex()
    graph = dict()

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
            tile_graph = ConnectTiles(row, col)
            graph.update({k: tile_graph[k] for k in tile_graph.keys() - graph.keys()})
            graph.update({k: min(graph[k], tile_graph[k]) for k in graph.keys() & tile_graph.keys()})

    return graph

def ResolveMinimumZ(graph, epsilon=0.001):
    """
    DOCME
    """

    upgraph = defaultdict(list)
    # outdegree = Counter()
    exterior = (-1, 1)
    nodata = -99999.0

    for (w1, w2), z in graph.items():
        upgraph[w2].append((w1, z))
        # outdegree[w1] += 1

    # for watershed in [w for w in upgraph if outdegree[w] == 0]:
    #     z = ?
    #     upgraph[exterior].append((watershed, z))

    directed = dict()
    ulinks = dict()
    queue = [(nodata, exterior, None)]

    while queue:

        minz, watershed, downstream = heappop(queue)

        if watershed in directed:

            for neighbor, upz in upgraph[watershed]:
                if neighbor in directed:
                    z = directed[watershed][1]
                    neighbor_z = directed[neighbor][1]
                    # TODO use epsilon comparison ?
                    # if z == neighbor_z:
                    if abs(z - neighbor_z) < epsilon:
                        w1, w2 = sorted([watershed, neighbor])
                        ulinks[w1, w2] = z

            continue

        directed[watershed] = (downstream, minz)

        for upstream, upz in upgraph[watershed]:

            # upz = max(upz, minz)
            if upz < minz:
                upz = minz

            heappush(queue, (upz, upstream, watershed))

    return directed, ulinks

def EnsureEpsilonGradient(directed, ulinks, areas, epsilon=.001):
    """
    DOCME
    """

    upgraph = defaultdict(list)

    for watershed, (downstream, z) in directed.items():
        upgraph[downstream].append((watershed, z))

    # Decide extra edge direction
    # What if w1 is upstream of w2 or vice versa ?

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

    for (w1, w2), z in ulinks.items():

        if isupstream(w1, w2):
        
            upgraph[w2].append((w1, z))
        
        elif isupstream(w2, w1):
            
            upgraph[w1].append((w2, z))
        
        else:
        
            area1 = areas[w1]
            area2 = areas[w2]
            
            if area1 > area2:
                upgraph[w2].append((w1, z))
            else:
                upgraph[w1].append((w2, z))

    exterior = (-1, 1)
    nodata = -99999.0
    resolved = dict()

    queue = [(exterior, nodata)]

    while queue:

        watershed, z = queue.pop(0)

        for upstream, upz in upgraph[watershed]:
            
            if upz - z < epsilon:
                upz = z + epsilon
                resolved[upstream] = (watershed, upz)
            
            queue.append((upstream, upz))

    return resolved

def ResolveFlatSpillover():
    """
    DOCME
    """

    graph = BuildFlatSpilloverGraph()
    directed, ulinks = ResolveMinimumZ(graph)
    unitareas = WatershedUnitAreas()
    areas = WatershedCumulativeAreas(directed, unitareas)
    resolved = EnsureEpsilonGradient(directed, ulinks, areas)

    output = filename('resolved_minz')
    minz = [watershed + (directed[watershed][1],) for watershed in resolved]
    np.savez(output, minz=np.array(minz))

    click.secho('Saved to : %s' % output, fg='green')

def ApplyMinimumZ(row, col, **kwargs):
    
        """
    Ajuste l'altitude des dépressions en bordure de tuile,
    et calcule la carte des dépressions
    (différentiel d'altitude avec le point de débordement)
    """

    tile_index = tile_index()
    tile = tile_index[row, col]

    minz_file = filename('resolved_minz')
    minimum_z = np.load(minz_file)['minz']

    index = {int(w): z for t, w, z in minimum_z if int(t) == tile.gid}
    del minimum_z

    filled_raster = filename('prefilled', row=row, col=col)
    label_raster = filename('flat_labels', row=row, col=col)
    output = filename('resolved', row=row, col=col)

    with rio.open(filled_raster) as ds:

        nodata = ds.nodata
        profile = ds.profile.copy()

        with rio.open(label_raster) as ds2:
            labels = ds2.read(1)

        @np.vectorize
        def minimumz(x):
            """
            Map watershed to its minimum elevation

            %time minz = minimumz(labels)
            CPU times: user 13.3 s, sys: 1.46 s, total: 14.7 s
            Wall time: 16.9 s
            """
            if x == 0:
                return nodata
            return index[x]

        try:
            minz = np.float32(minimumz(labels))
        except KeyError:
            click.secho('Error while processing tile (%d, %d)' % (row, col))
            return

        del labels

        elevations = ds.read(1)
        filled = np.maximum(elevations, minz)
        filled[dem == nodata] = nodata

        del elevations
        del minz

        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(filled, 1)