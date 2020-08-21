# coding: utf-8

"""
DEM Depression Filling Procedure

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
from collections import defaultdict
from heapq import heappush, heappop
import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs

from ..config import config
from .. import terrain_analysis as ta
from .. import speedup
from .Burn import BurnTile

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

def silent(msg):
    pass

def LabelWatersheds(
        row, col,
        dataset='dem',
        burn=-1.0,
        exterior_data=9000.0,
        overwrite=True,
        verbose=False):
    """
    Identifie et numérote les bassins versants
    et les zones continues de même altitude,
    avec remplissage des creux
    """

    tile_index = tileindex()

    if (row, col) not in tile_index:
        return

    # outputs = config.fileset(['prefilled', 'labels', 'graph'], row=row, col=col)
    output_filled = config.tileset().tilename('dem-filled', row=row, col=col)
    output_labels = config.tileset().tilename('dem-watershed-labels', row=row, col=col)
    output_graph = config.tileset().tilename('dem-watershed-graph', row=row, col=col)

    if verbose:

        def info(msg):
            click.secho(msg, fg='cyan')

        def step(msg):
            click.secho(msg, fg='yellow')

    else:

        info = step = silent

    for output in [output_filled, output_labels, output_graph]:
        if os.path.exists(output) and not overwrite:
            info('Output already exists: %s' % output)
            return 

    info('Processing tile (%02d, %02d)' % (row, col))

    with rio.open(config.tileset().tilename(dataset, row=row, col=col)) as ds:

        profile = ds.profile.copy()
        nodata = ds.nodata

        if burn < 0:
            elevations = ds.read(1)
        else:
            elevations = BurnTile(dataset, row, col, burn)

    step('Label flats')

    labels, graph = ta.watershed_labels(elevations, nodata, exterior_data)
    labels = np.uint32(labels)

    step('Write filled DEM')

    profile.update(
        compress='deflate',
        tiled='yes'
    )

    with rio.open(output_filled, 'w', **profile) as dst:
        dst.write(elevations, 1)

    step('Write labels and watershed graph')

    profile.update(
        compress='deflate',
        tiled='yes',
        dtype=np.uint32,
        nodata=0)

    with rio.open(output_labels, 'w', **profile) as dst:
        dst.write(labels, 1)

    np.savez(
        output_graph,
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
        graph=np.array(list(graph.items()), dtype=object)
    )

def ResolveMinimumZ(graph, nodata, epsilon=0.002):
    """
    Walk over spillover graph from minimum z to maximum z,
    and calculate the minimum outlet elevation
    for all watersheds.
    """

    graph_index = defaultdict(list)
    for l1, l2 in graph.keys():
        graph_index[l1].append(l2)
        graph_index[l2].append(l1)

    exterior = (-1, 1)
    queue = [(nodata, exterior, None)]
    seen = set()
    # minimum_z = list()
    # flats = list()
    # flatz = dict()
    directed = dict()

    while queue:

        minz, watershed, downstream = heappop(queue)
        if watershed in seen:
            continue

        if downstream is not None:
            directed[watershed] = (downstream, minz)

        # minimum_z.append((z, watershed))
        seen.add(watershed)
        # if watershed in flatz and flatz[watershed] < z:
        #     flats.append(watershed)

        for link in graph_index[watershed]:
            
            l1, l2 = sorted((watershed, link))
            zlink = graph[(l1, l2)]

            if zlink < minz:
                # minz += epsilon
                # zlink = minz
                zlink = minz + epsilon
                # flatz[link] = min(zlink, flatz.get(link, float('inf')))
                # zlink = z

            heappush(queue, (zlink, link, watershed))

    # return minimum_z, flats
    return directed

def ResolveWatershedSpillover(overwrite):
    """
    Calcule le graph de débordement
    entre les différentes tuiles
    """

    output = config.tileset().filename('dem-watershed-spillover')

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    tile_index = tileindex()

    click.secho('Build spillover graph', fg='cyan')

    graph = dict()
    nodata = -99999.0

    def tiledatafn(row, col):
        return config.tileset().tilename('dem-watershed-graph', row=row, col=col)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
        
            # click.secho('Processing tile (%d, %d)' % (row, col), fg='cyan')
            this_graph = speedup.connect_tile(row, col, nodata, tile_index, tiledatafn)
            graph.update({k: this_graph[k] for k in this_graph.keys() - graph.keys()})
            graph.update({k: min(graph[k], this_graph[k]) for k in graph.keys() & this_graph.keys()})

    click.secho('Resolve Watershed\'s Minimum Z', fg='cyan')

    # graph_index = defaultdict(list)
    # for l1, l2 in graph.keys():
    #     graph_index[l1].append(l2)
    #     graph_index[l2].append(l1)

    # directed = resolve(graph, nodata)
    # unitareas = WatershedUnitAreas()

    # areas = WatershedCumulativeAreas(directed, unitareas)
    # fixed, extra_links = CheckConnectedFlats(directed, graph, graph_index)

    # for w1, w2 in extra_links:

    #     minz = extra_links[w1, w2]
    #     area1 = areas[w1]
    #     area2 = areas[w2]

    #     if area2 > area1:
    #         w1, w2 = w2, w1

    #     for link in graph_index[w1]:
    #         v1, v2 = sorted([w1, link])
    #         graph[(v1, v2)] = max(graph[(v1, v2)], minz)

    directed = ResolveMinimumZ(graph, nodata)
    # unitareas = WatershedUnitAreas()
    # areas = WatershedCumulativeAreas(directed, unitareas)

    # was_fixed = float('inf')
    # iterations = 0
    # max_iterations = 5
    # dlinks = dict()
    # ulinks = dict()
    
    # while was_fixed > 0 and iterations < max_iterations:

    #     fixed = 0
        
    #     for row, col in tile_index:
    #         CheckBorderFlats(directed, graph, row, col, dlinks, ulinks)
        
    #     # directed = ResolveMinimumZ(graph, nodata)
    #     directed = ResolveFlatLinks(directed, dlinks, ulinks, areas)
    #     fixed = len(dlinks) + len(ulinks)
        
    #     if fixed > was_fixed:
    #         break
        
    #     was_fixed = fixed
    #     iterations += 1

    # click.secho('Fixed border flats elevations with %d iterations' % iterations, fg='green')

    minz = [watershed + (directed[watershed][1],) for watershed in directed]
    np.savez(output, minz=np.array(minz))

    click.secho('Saved to : %s' % output, fg='green')

    # flats = [(t, w) for t, w in flats]
    # np.savez(os.path.join(workdir, 'FLATS.npz'), flats=np.asarray(flats))

def DispatchWatershedMinimumZ(row, col, **kwargs):
    """
    Ajuste l'altitude des dépressions en bordure de tuile,
    (différentiel d'altitude avec le point de débordement)
    """

    overwrite = kwargs.get('overwrite', False)

    tile_index = tileindex()
    tile = tile_index[row, col]

    minz_file = config.tileset().filename('dem-watershed-spillover')
    minimum_z = np.load(minz_file)['minz']

    index = {int(w): z for t, w, z in minimum_z if int(t) == tile.gid}

    filled_raster = config.tileset().tilename('dem-filled', row=row, col=col)
    label_raster = config.tileset().tilename('dem-watershed-labels', row=row, col=col)
    output = config.tileset().tilename('dem-filled-resolved', row=row, col=col)

    def info(msg):
        click.secho(msg, fg='yellow')

    if os.path.exists(output) and not overwrite:
        info('Output already exists: %s' % output)
        return

    with rio.open(filled_raster) as ds:

        dem = ds.read(1)
        profile = ds.profile.copy()

        with rio.open(label_raster) as ds2:
            labels = ds2.read(1)

        # def minimumz(labels):
        #     """
        #     Map watershed to its minimum elevation

        #     %time minz = minimumz(labels)
        #     CPU times: user 1min 24s, sys: 37.8 s, total: 2min 2s
        #     Wall time: 2min 2s
        #     """

        #     minz = np.zeros_like(dem)

        #     for label in np.unique(labels):
        #         minz[labels == label] = index[(tile_id, label)]

        #     minz[labels == 0] = nodata

        #     return minz

        # @np.vectorize
        # def minimumz(x):
        #     """
        #     Map watershed to its minimum elevation

        #     %time minz = minimumz(labels)
        #     CPU times: user 13.3 s, sys: 1.46 s, total: 14.7 s
        #     Wall time: 16.9 s
        #     """
        #     if x == 0:
        #         return ds.nodata
        #     return index[x]

        # try:
        #     minz = np.float32(minimumz(labels))
        # except KeyError:
        #     click.secho('Error while processing tile (%d, %d)' % (row, col))
        #     return

        # %timeit minimumz(labels)
        # 26 µs ± 250 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
        # %timeit speedup.minimumz(labels, index, nodata)
        # 7.9 µs ± 52 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
        # speedup.minimumz(labels, index, ds.nodata)
        # CPU times: user 1.3 s, sys: 64 ms, total: 1.36 s
        # Wall time: 1.36 s


        minz = speedup.minimumz(labels, index, ds.nodata)
        filled = np.maximum(dem, minz)
        filled[dem == ds.nodata] = ds.nodata

        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(filled, 1)
