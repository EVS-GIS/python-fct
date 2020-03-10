#!/usr/bin/env python
# coding: utf-8

"""
1ère étape du calcul du plan de drainage global :

- remplit les zones NODATA du MNT de résolution 5 m (RGE Alti 5 m)
  avec les valeurs interpolées de la BD Alti 25 m

- identifie et numérote les bassins versants
  et les zones continues de même altitude,
  avec remplissage des creux

- construit le graphe de connection
  entre bassins versants contigus

Séquence :

1. Extract and Path DEM
2. Fill Sinks and Label Flats
3. Resolve Global Spillover Graph
4. Apply Spillover Elevations and Map Flats

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

import terrain_analysis as ta
import speedup
from config import tileindex, filename, fileset, workdir

tile_height = 7150
tile_width = 9800

def silent(msg):
    pass

def TileExtendedBoundingBox(row, col, expand=20):

    template = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_PATCHED.tif' % (row, col))
    output = os.path.join(workdir, '../TILEBOXES.shp')

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('TILE', 'int')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    tile_index = tileindex()

    with rio.open(template) as ds:

        gid = tile_index[(row, col)].gid
        height, width = ds.shape
        xmin, ymax = ds.xy(-expand, -expand)
        xmax, ymin = ds.xy(height+expand, width+expand)

        box = [[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin]]
        geom = {'type': 'Polygon', 'coordinates': [box]}
        props = {'TILE': gid}

        if os.path.exists(output):
            mode = 'a'
        else:
            mode = 'w'

        with fiona.open(output, mode, **options) as dst:
            feature = {'geometry': geom, 'properties': props}
            dst.write(feature)

def ExtractAndPatchTile(row, col, overwrite, quiet):
    """
    1ère étape du calcul du plan de drainage global :

    - remplit les zones NODATA du MNT de résolution 5 m (RGE Alti 5 m)
      avec les valeurs interpolées de la BD Alti 25 m

    - identifie et numérote les bassins versants
      et les zones continues de même altitude,
      avec remplissage des creux

    - construit le graphe de connection
      entre bassins versants contigus
    """

    tile_index = tileindex()
    RGE = filename('rge', 'input')
    BDA = filename('bda', 'input')
    
    if (row, col) not in tile_index:
        return

    outputs = dict()
    outputs['patched'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_PATCHED.tif' % (row, col))
    # outputs['filled'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED.tif' % (row, col))
    # outputs['label'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_LABELS.tif' % (row, col))
    # outputs['graph'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_GRAPH.npz' % (row, col))

    if quiet:
        
        info = step = silent
    
    else:

        def info(msg):
            click.secho(msg, fg='cyan')

        def step(msg):
            click.secho(msg, fg='yellow')

    for output in outputs.values():
        if os.path.exists(output) and not overwrite:
            info('Output already exists: %s' % output)
            return 

    info('Processing tile (%02d, %02d)' % (row, col))

    tile = tile_index[(row, col)]

    step('Read and patch elevations')

    with rio.open(RGE) as ds:

        nodata = ds.nodata
        profile = ds.profile.copy()
        dst_transform = ds.transform * ds.transform.translation(col*tile_width, row*tile_height)

        window1 = Window(tile.col, tile.row, tile_width, tile_height)
        dem1 = ds.read(1, window=window1)

        with rio.open(BDA) as ds2:

            i2, j2 = ds2.index(*ds.xy(window1.row_off, window1.col_off))
            window2 = Window(j2, i2, tile_width//5, tile_height//5)
            
            dem2 = ds2.read(1, resampling=Resampling.bilinear, out_shape=dem1.shape,
                boundless=True, fill_value=ds2.nodata, window=window2)
            
            mask = (dem1 == ds.nodata) & (dem2 != ds2.nodata)
            dem1[mask] = dem2[mask]
            del mask

    del dem2

    profile.update(
        compress='deflate',
        transform=dst_transform,
        height=tile_height,
        width=tile_width
    )

    with rio.open(outputs['patched'], 'w', **profile) as dst:
        dst.write(dem1, 1)

def FillDepressions(row, col, overwrite, quiet):
    """
    1ère étape du calcul du plan de drainage global :

    - remplit les zones NODATA du MNT de résolution 5 m (RGE Alti 5 m)
      avec les valeurs interpolées de la BD Alti 25 m

    - identifie et numérote les bassins versants
      et les zones continues de même altitude,
      avec remplissage des creux

    - construit le graphe de connection
      entre bassins versants contigus
    """

    tile_index = tileindex()
    RGE = filename('rge', 'input')
    BDA = filename('bda', 'input')
    
    if (row, col) not in tile_index:
        return

    outputs = fileset(['prefilled', 'labels', 'graph'], row=row, col=col)

    if quiet:
        
        info = step = silent
    
    else:

        def info(msg):
            click.secho(msg, fg='cyan')

        def step(msg):
            click.secho(msg, fg='yellow')

    for output in outputs.values():
        if os.path.exists(output) and not overwrite:
            info('Output already exists: %s' % output)
            return 

    info('Processing tile (%02d, %02d)' % (row, col))

    tile = tile_index[(row, col)]

    with rio.open(filename('patched', row=row, col=col)) as ds:

        profile = ds.profile.copy()
        nodata = ds.nodata
        elevations = ds.read(1)

    step('Read and patch elevations')

    # with rio.open(RGE) as ds:

    #     nodata = ds.nodata
    #     profile = ds.profile.copy()
    #     dst_transform = ds.transform * ds.transform.translation(col*tile_width, row*tile_height)

    #     window1 = Window(tile.col, tile.row, tile_width, tile_height)
    #     dem1 = ds.read(1, window=window1)

    #     with rio.open(BDA) as ds2:

    #         i2, j2 = ds2.index(*ds.xy(window1.row_off, window1.col_off))
    #         window2 = Window(j2, i2, tile_width//5, tile_height//5)
            
    #         dem2 = ds2.read(1, resampling=Resampling.bilinear, out_shape=dem1.shape,
    #             boundless=True, fill_value=ds2.nodata, window=window2)
            
    #         mask = (dem1 == ds.nodata) & (dem2 != ds2.nodata)
    #         dem1[mask] = dem2[mask]
    #         del mask

    # del dem2

    # profile.update(
    #     compress='deflate',
    #     transform=dst_transform,
    #     height=tile_height,
    #     width=tile_width
    # )

    # with rio.open(outputs['patched'], 'w', **profile) as dst:
    #     dst.write(dem1, 1)

    # step('Fill depressions')

    # dx = dy = 5.0
    # filled = ta.fillsinks2(dem1, nodata, dx, dy, 0)
    # del dem1

    # step('Write filled DEM')

    # profile.update(
    #     compress='deflate',
    #     tiled='yes',
    #     transform=dst_transform,
    #     height=tile_height,
    #     width=tile_width
    # )

    step('Label flats')

    labels, graph, seeds = ta.watershed_labels(elevations, nodata)
    labels = np.uint32(labels)

    step('Write filled DEM')

    profile.update(
        compress='deflate',
        tiled='yes'
    )

    with rio.open(outputs['prefilled'], 'w', **profile) as dst:
        dst.write(elevations, 1)

    step('Write labels and watershed graph')

    profile.update(
        compress='deflate',
        tiled='yes',
        dtype=np.uint32,
        nodata=0)

    with rio.open(outputs['labels'], 'w', **profile) as dst:
        dst.write(labels, 1)

    np.savez(
        outputs['graph'],
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
        graph=np.array(list(graph.items()))
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

@click.group()
def cli():
    pass

@cli.command()
def boxes():
    """
    Generate tile extended bounding boxes
    """

    tile_index = tileindex()

    output = os.path.join(workdir, '../TILEBOXES.shp')

    if os.path.exists(output):
        for filename in glob.glob(os.path.join(workdir, '../TILEBOXES.*')):
            if os.path.exists(filename):
                os.unlink(filename)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
            TileExtendedBoundingBox(row, col, 20)

@cli.command()
@click.argument('row', type=int)
@click.argument('col', type=int)
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--quiet/--no-quiet', '-q', default=False, help='Suppress message output ?')
def tile(row, col, overwrite, quiet):
    """
    Calcule la tuile de coordonnées (ROW, COL)
    tel que défini dans `TILES.shp`
    """
    
    FillDepressions(row, col, overwrite, quiet)
    # ExtractAndPatchTile(row, col, overwrite, quiet)


def Starred(args):
    """
    Starred version of `function` for use with pool.imap_unordered()
    """

    return FillDepressions(*args)
    # return ExtractAndPatchTile(*args)

@cli.command()
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
@click.option('--quiet/--no-quiet', '-q', default=True, help='Suppress message output ?')
def batch(overwrite, processes, quiet):
    """
    Calcule toutes les tuiles définies dans `TILES.shp`
    """

    from multiprocessing import Pool

    tile_index = tileindex()

    click.secho('Running %d processes ...' % processes, fg='yellow')
    arguments = (tuple(key) + (overwrite, quiet) for key in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(Starred, arguments)
        with click.progressbar(pooled, length=len(tile_index)) as progress:
            for _ in progress:
                click.echo('\r')

def WatershedUnitAreas():

    tile_index = tileindex()
    areas = defaultdict(lambda: 0)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[row, col].gid
            label_raster = filename('labels', row=row, col=col)

            with rio.open(label_raster) as ds:
        
                labels = ds.read(1)
                this_areas = speedup.label_areas(labels)
                areas.update({(tile, w): area for w, area in this_areas.items()})

    return areas

def WatershedCumulativeAreas(directed, unitareas):

    areas = unitareas.copy()
    indegree = Counter()

    for watershed in directed:

        downstream, minz = directed[watershed]
        indegree[downstream] += 1

    queue = [w for w in directed if indegree[w] == 0]

    while queue:

        watershed = queue.pop(0)

        if watershed not in directed:
            continue

        downstream, minz = directed[watershed]

        areas[downstream] = areas[downstream] + areas[watershed]

        indegree[downstream] -= 1

        if indegree[downstream] == 0:
            queue.append(downstream)

    return areas

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
        return np.load(filename("graph", row=i, col=j), allow_pickle=True)

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

@cli.command()
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
# @click.option('--processes', '-j', default=1, help="Execute j parallel processes")
# @click.option('--quiet/--no-quiet', '-q', default=True, help='Suppress message output ?')
def spillover(overwrite):
    """
    Calcule le graph de débordement
    entre les différentes tuiles
    """

    output = filename('spillover')
    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    tile_index = tileindex()

    click.secho('Build spillover graph', fg='cyan')

    graph = dict()
    nodata = -99999.0

    def tiledatafn(row, col):
        return filename('graph', row=row, col=col)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
        
            # click.secho('Processing tile (%d, %d)' % (row, col), fg='cyan')
            this_graph = ta.connect_tile(row, col, nodata, tile_index, tiledatafn)
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
    unitareas = WatershedUnitAreas()
    areas = WatershedCumulativeAreas(directed, unitareas)

    was_fixed = float('inf')
    iterations = 0
    max_iterations = 5
    dlinks = dict()
    ulinks = dict()
    
    while was_fixed > 0 and iterations < max_iterations:

        fixed = 0
        
        for row, col in tile_index:
            CheckBorderFlats(directed, graph, row, col, dlinks, ulinks)
        
        # directed = ResolveMinimumZ(graph, nodata)
        directed = ResolveFlatLinks(directed, dlinks, ulinks, areas)
        fixed = len(dlinks) + len(ulinks)
        
        if fixed > was_fixed:
            break
        
        was_fixed = fixed
        iterations += 1

    click.secho('Fixed border flats elevations with %d iterations' % iterations, fg='green')

    minz = [watershed + (directed[watershed][1],) for watershed in directed]
    np.savez(output, minz=np.array(minz))

    click.secho('Saved to : %s' % output, fg='green')

    # flats = [(t, w) for t, w in flats]
    # np.savez(os.path.join(workdir, 'FLATS.npz'), flats=np.asarray(flats))

# def FinalizeTile(row, col, tile_id, overwrite):
def FinalizeTile(args):
    """
    Ajuste l'altitude des dépressions en bordure de tuile,
    et calcule la carte des dépressions
    (différentiel d'altitude avec le point de débordement)
    """

    row, col, tile_id, overwrite = args

    minz_file = filename('spillover')
    minimum_z = np.load(minz_file)['minz']

    index = {int(w): z for t, w, z in minimum_z if int(t) == tile_id}

    # with click.progressbar(tile_index) as progress:
    #     for row, col in progress:

    # tile_id = tile_index[(row, col)].gid

    reference_raster = filename('patched', row=row, col=col)
    filled_raster = filename('prefilled', row=row, col=col)
    label_raster = filename('labels', row=row, col=col)
    outputs = fileset(['filled', 'flats'], row=row, col=col)

    def info(msg):
        click.secho(msg, fg='yellow')

    for output in outputs.values():
        if os.path.exists(output) and not overwrite:
            info('Output already exists: %s' % output)
            return

    with rio.open(filled_raster) as ds:

        dem = ds.read(1)
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

        try:
            minz = np.float32(minimumz(labels))
        except KeyError:
            click.secho('Error while processing tile (%d, %d)' % (row, col))
            return

        filled = np.maximum(dem, minz)
        filled[dem == nodata] = nodata

        del dem
        del minz

        with rio.open(reference_raster) as ds2:
            reference = ds2.read(1)

        flats = filled - reference

        del reference

        flats[filled == nodata] = nodata

        profile.update(compress='deflate')

        with rio.open(outputs['filled'], 'w', **profile) as dst:
            dst.write(filled, 1)

        with rio.open(outputs['flats'], 'w', **profile) as dst:
            dst.write(flats, 1)

@cli.command()
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def finalize(overwrite, processes):
    """
    Ajuste l'altitude des dépressions en bordure de tuile,
    et calcule la carte des dépressions
    (différentiel d'altitude avec le point de débordement)
    """

    from multiprocessing import Pool

    tile_index = tileindex()

    arguments = (tuple(key) + (tile_index[key].gid, overwrite) for key in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(FinalizeTile, arguments)
        with click.progressbar(pooled, length=len(tile_index)) as progress:
            for _ in progress:
                click.echo('\r')               

# from options import batch

# @batch(cli)
# def finalize(args):
#     """
#     Ajuste l'altitude des dépressions en bordure de tuile,
#     et calcule la carte des dépressions
#     (différentiel d'altitude avec le point de débordement)
#     """

#     FinalizeTile(*args)

if __name__ == '__main__':
    cli()
