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
import rasterio as rio
from rasterio.windows import Window
from rasterio.warp import Resampling
import fiona
import fiona.crs
import numpy as np
from collections import namedtuple, defaultdict
from heapq import heappush, heappop
import terrain_analysis as ta

Tile = namedtuple('Tile', ('gid', 'row', 'col', 'x0', 'y0', 'i', 'j'))

BDA = '/media/crousson/Backup/REFERENTIELS/IGN/BDALTI_25M/BDALTI25M.tif'
RGE = '/media/crousson/Backup/REFERENTIELS/IGN/RGEALTI/2017/RGEALTI.tif'

workdir = '/media/crousson/Backup/PRODUCTION/RGEALTI/TILES'
tile_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/TILES.shp'
tile_index = dict()
tile_height = 7150
tile_width = 9800

def read_tile_index():

    with fiona.open(tile_shapefile) as fs:
        for feature in fs:
            row = feature['properties']['I']
            col = feature['properties']['J']
            tile_index[(row, col)] = Tile(*feature['properties'].values())

def silent(msg):
    pass

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

def FillSinks(row, col, overwrite, quiet):
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
    
    if (row, col) not in tile_index:
        return

    outputs = dict()
    outputs['patched'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_PATCHED.tif' % (row, col))
    outputs['filled'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED.tif' % (row, col))
    outputs['label'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_LABELS.tif' % (row, col))
    outputs['graph'] = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_GRAPH.npz' % (row, col))

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

    step('Fill depressions')

    dx = dy = 5.0
    filled = ta.fillsinks2(dem1, nodata, dx, dy, 0)
    del dem1

    step('Write filled DEM')

    profile.update(
        compress='deflate',
        tiled='yes',
        transform=dst_transform,
        height=tile_height,
        width=tile_width
    )

    step('Label flats')

    with rio.open(outputs['filled'], 'w', **profile) as dst:
        dst.write(filled, 1)

    labels, graph = ta.watershed_labels(filled, nodata)
    labels = np.uint32(labels)

    step('Write labels and watershed graph')

    profile.update(
        compress='deflate',
        tiled='yes',
        dtype=np.uint32,
        nodata=0,
        transform=dst_transform,
        height=tile_height,
        width=tile_width)

    with rio.open(outputs['label'], 'w', **profile) as dst:
        dst.write(labels, 1)

    np.savez(
        outputs['graph'],
        z=np.array([
            filled[0, :],
            filled[:, -1],
            np.flip(filled[-1, :], axis=0),
            np.flip(filled[:, 0], axis=0)]),
        labels=np.array([
            labels[0, :],
            labels[:, -1],
            np.flip(labels[-1, :], axis=0),
            np.flip(labels[:, 0], axis=0)]),
        graph=np.array(list(graph.items()))
    )

def resolve(graph, nodata):
    """
    Walk over spillover graph from minimum z to maximum z,
    and calculate the minimum outlet elevation
    for all watersheds.
    """

    # epsilon = 0.001

    graph_index = defaultdict(list)
    for l1, l2 in graph.keys():
        graph_index[l1].append(l2)
        graph_index[l2].append(l1)

    exterior = (-1,1)
    queue = [(nodata, exterior)]
    seen = set()
    minimum_z = list()
    # flats = list()
    # flatz = dict()

    while queue:

        z, watershed = heappop(queue)
        if watershed in seen:
            continue

        minimum_z.append((z, watershed))
        seen.add(watershed)
        # if watershed in flatz and flatz[watershed] < z:
        #     flats.append(watershed)

        for link in graph_index[watershed]:
            
            l1, l2 = sorted((watershed, link))
            zlink = graph[(l1, l2)]

            if zlink < z:
                # zlink = z + epsilon
                # flatz[link] = min(zlink, flatz.get(link, float('inf')))
                zlink = z

            heappush(queue, (zlink, link))

    # return minimum_z, flats
    return minimum_z

@click.group()
def cli():
    pass

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
    
    read_tile_index()
    FillSinks(row, col, overwrite, quiet)
    # ExtractAndPatchTile(row, col, overwrite, quiet)


def Starred(args):
    """
    Starred version of `function` for use with pool.imap_unordered()
    """

    return FillSinks(*args)
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

    read_tile_index()

    click.secho('Running %d processes ...' % processes, fg='yellow')
    arguments = (tuple(key) + (overwrite, quiet) for key in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(Starred, arguments)
        with click.progressbar(pooled, length=len(tile_index)) as progress:
            for _ in progress:
                click.echo('\r')

@cli.command()
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
# @click.option('--processes', '-j', default=1, help="Execute j parallel processes")
# @click.option('--quiet/--no-quiet', '-q', default=True, help='Suppress message output ?')
def spillover(overwrite):
    """
    Calcule le graph de débordement
    entre les différentes tuiles
    """

    output = os.path.join(workdir, 'MINIMUMZ.npz')
    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists: %s' % output, fg='yellow')
        return

    read_tile_index()

    def tiledatafn(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_GRAPH.npz' % (row, col))

    graph = dict()
    nodata = -99999.0

    click.secho('Build spillover graph', fg='cyan')

    with click.progressbar(tile_index) as progress:
        for row, col in progress:
        
            # click.secho('Processing tile (%d, %d)' % (row, col), fg='cyan')
            this_graph = ta.connect_tile(row, col, nodata, tile_index, tiledatafn)
            graph.update({k: this_graph[k] for k in this_graph.keys() - graph.keys()})
            graph.update({k: min(graph[k], this_graph[k]) for k in graph.keys() & this_graph.keys()})

    click.secho('Resolve Watershed\'s Minimum Z', fg='cyan')

    nodata = -99999.0
    minimum_z = resolve(graph, nodata)
    index = [(t, w, z) for z, (t, w) in minimum_z]

    np.savez(output, minz=np.array(index))

    click.secho('Saved to : %s' % output, fg='green')

    # flats = [(t, w) for t, w in flats]
    # np.savez(os.path.join(workdir, 'FLATS.npz'), flats=np.asarray(flats))

def FinalizeTile(args):
    """
    Ajuste l'altiude des dépressions en bordure de tuile,
    et calcule la carte des dépressions
    (différentiel d'altitude avec le point de débordement)
    """

    row, col, tile_id, overwrite = args

    minz_file = os.path.join(workdir, 'MINIMUMZ.npz')    
    minimum_z = np.load(minz_file)['minz']

    index = {(int(t), int(w)): z for t, w, z in minimum_z}

    # with click.progressbar(tile_index) as progress:
    #     for row, col in progress:

    # tile_id = tile_index[(row, col)].gid

    reference_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_PATCHED.tif' % (row, col))
    filled_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED.tif' % (row, col))
    label_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_LABELS.tif' % (row, col))
    outputs = dict(
        filled=os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED2.tif' % (row, col)),
        flats = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLATS.tif' % (row, col))
    )

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
            return index[(tile_id, x)]

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
    Ajuste l'altiude des dépressions en bordure de tuile,
    et calcule la carte des dépressions
    (différentiel d'altitude avec le point de débordement)
    """

    from multiprocessing import Pool

    read_tile_index()

    arguments = (tuple(key) + (tile_index[key].gid, overwrite) for key in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(FinalizeTile, arguments)
        with click.progressbar(pooled, length=len(tile_index)) as progress:
            for _ in progress:
                click.echo('\r')               

if __name__ == '__main__':
    cli()
