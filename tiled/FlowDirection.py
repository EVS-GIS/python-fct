#!/usr/bin/env python
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

import click
import os
import rasterio as rio
from rasterio.windows import Window
from rasterio.warp import Resampling
import fiona
import fiona.crs
import numpy as np
from collections import namedtuple, defaultdict, Counter
from heapq import heappush, heappop

import richdem as rd
from speedup import graph_acc, flow_accumulation
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

read_tile_index()

def FlowDirection(row, col):

    read_tile_index()

    output = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))

    def filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED2.tif' % (row, col))

    elevation_raster = filename(row, col)

    with rio.open(elevation_raster) as ds:

        height, width = ds.shape
        extended = np.zeros((height+2, width+2), dtype=np.float32)

        extended[1:-1, 1:-1] = ds.read(1)

        # top
        i = row-1
        j = col

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[0, 1:-1] = ds2.read(1, window=Window(0, height-1, width, 1)).reshape(width)

        else:

            extended[0, 1:-1] = ds.nodata

        # bottom
        i = row+1
        j = col

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[-1, 1:-1] = ds2.read(1, window=Window(0, 0, width, 1)).reshape(width)

        else:

            extended[-1, 1:-1] = ds.nodata

        # left
        i = row
        j = col-1

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[1:-1, 0] = ds2.read(1, window=Window(width-1, 0, 1, height)).reshape(height)

        else:

            extended[1:-1, 0] = ds.nodata

        # right
        i = row
        j = col+1

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[1:-1, -1] = ds2.read(1, window=Window(0, 0, 1, height)).reshape(height)

        else:

            extended[1:-1, -1] = ds.nodata

        # top-left
        i = row-1
        j = col-1

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[0, 0] = np.asscalar(ds2.read(1, window=Window(width-1, height-1, 1, 1)))

        else:

            extended[0, 0] = ds.nodata

        # bottom-left
        i = row+1
        j = col-1

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[-1, 0] = np.asscalar(ds2.read(1, window=Window(width-1, 0, 1, 1)))

        else:

            extended[-1, 0] = ds.nodata

        # top-right
        i = row-1
        j = col+1

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[0, -1] = np.asscalar(ds2.read(1, window=Window(0, height-1, 1, 1)))

        else:

            extended[0, -1] = ds.nodata

        # bottom-right
        i = row+1
        j = col+1

        if (i, j) in tile_index:
        
            other_raster = filename(i, j)
            with rio.open(other_raster) as ds2:
                extended[-1, -1] = np.asscalar(ds2.read(1, window=Window(0, 0, 1, 1)))

        else:

            extended[-1, -1] = ds.nodata

        extended = rd.rdarray(extended, no_data=ds.nodata)
        rd.BreachDepressions(extended, True, 'D8')
        rd.FillDepressions(extended, True, True, 'D8')
        flow = ta.flowdir(extended, ds.nodata)

        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype=np.int16, nodata=-1)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(flow[1:-1, 1:-1], 1)

def Outlets(row, col):
    """
    DOCME
    """

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('LCA', 'int'),
            ('FROM', 'int'),
            ('FROMX', 'float'),
            ('FROMY', 'float')]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    # read_tile_index()

    flow_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))

    def output_filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_INLETS.shp' % (row, col))

    gid = tile_index[(row, col)].gid
    tiles = defaultdict(list)

    with rio.open(flow_raster) as ds:

        height, width = ds.shape
        flow = ds.read(1)
        mask = np.ones_like(flow, dtype=np.uint8)
        outlets, targets = ta.tile_outlets(flow, mask)

        for current, (ti, tj) in enumerate(targets):

            top = (ti < 0)
            bottom = (ti >= height)
            left = (tj < 0)
            right = (tj >= width)

            if top:
                di = -1
            elif bottom:
                di = +1
            else:
                di = 0

            if left:
                dj = -1
            elif right:
                dj = +1
            else:
                dj = 0

            tiles[(row+di, col+dj)].append(current)

        cum_area = 0
        skipped = 0

        for trow, tcol in tiles:

            if (trow, tcol) not in tile_index:
                skipped += len(tiles[(trow, tcol)])
                continue

            target = tile_index[(trow, tcol)].gid
            output = output_filename(trow, tcol)

            if os.path.exists(output):
                mode = 'a'
            else:
                mode = 'w'

            with fiona.open(output, mode, **options) as dst:
                for idx in tiles[(trow, tcol)]:

                    (i, j), area = outlets[idx]
                    ti, tj = targets[idx]
                    x, y = ds.xy(i, j)
                    tx, ty = ds.xy(ti, tj)

                    cum_area += area
                
                    geom = {'type': 'Point', 'coordinates': [tx, ty]}
                    props = {
                        'TILE': target,
                        'LCA': area,
                        'FROM': gid,
                        'FROMX': x,
                        'FROMY': y
                    }

                    dst.write({'geometry': geom, 'properties': props})

    click.secho('\nSkipped %d outlets' % skipped, fg='yellow')
    click.secho('Tile (%02d, %02d) Coverage = %.1f %%' % (row, col, (cum_area / (height*width) * 100)), fg='green')
    return cum_area

def Accumulate():
    """
    DOCME
    """

    # read_tile_index()

    def inlet_filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_INLETS.shp' % (row, col))

    def flow_filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))

    click.secho('Build outlets graph', fg='cyan')

    graph = dict()
    indegree = Counter()

    rge = rio.open(RGE)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[(row, col)].gid
            inlet_shapefile = inlet_filename(row, col)
            flow_raster = flow_filename(row, col)

            with rio.open(flow_raster) as ds:

                flow = ds.read(1)

                with fiona.open(inlet_shapefile) as fs:
                    for feature in fs:

                        # connect outlet->inlet

                        from_tile = feature['properties']['FROM']
                        area = feature['properties']['LCA']
                        from_i, from_j = rge.index(feature['properties']['FROMX'], feature['properties']['FROMY'])
                        i, j = rge.index(*feature['geometry']['coordinates'])
                        graph[(from_tile, from_i, from_j)] = (tile, i, j, area)
                        indegree[(tile, i, j)] += 1

                        # connect inlet->tile outlet

                        ti, tj = ta.outlet(flow, i, j)
                        
                        if ti >= 0 and tj >= 0:
                            graph[(tile, i, j)] = (tile, ti, tj, 0)
                            indegree[(tile, ti, tj)] += 1

    rge.close()

    click.secho('Created graph with %d nodes' % len(graph), fg='green')
    click.secho('Accumulate areas', fg='cyan')

    return graph, indegree

    # queue = [pixel for pixel in graph if indegree[pixel] == 0]
    # areas = defaultdict(lambda: 0)
    # seen = set()

    # with click.progressbar(length=len(indegree)) as progress:
    
    #     while queue:

    #         tile, i, j = queue.pop(0)

    #         if (tile, i, j) in seen:
    #             continue

    #         progress.update(1)
    #         seen.add((tile, i, j))

    #         if (tile, i, j) in graph:

    #             tile, i, j, area = graph[(tile, i, j)]
    #             areas[(tile, i, j)] += area*25e-6 # convert to km^2
    #             indegree[(tile, i, j)] -= 1

    #             if indegree[(tile, i, j)] == 0:
    #                 queue.append((tile, i, j))

    # return areas


def InletAreas(graph, indegree, areas):
    """
    DOCME
    """

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('LCA', 'int'),
            ('FROM', 'int'),
            ('FROMX', 'float'),
            ('FROMY', 'float'),
            ('AREAKM2', 'float')]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    def inlet_filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_INLETS.shp' % (row, col))

    def output_filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_INLET_AREAS.shp' % (row, col))

    rge = rio.open(RGE)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            tile = tile_index[(row, col)].gid
            index = {key[1:]: areas.get(key[1:], 0) for key in graph.keys() | indegree.keys() if key[0] == tile}

            with fiona.open(inlet_filename(row, col)) as fs:
                with fiona.open(output_filename(row, col), 'w', **options) as dst:

                    for feature in fs:

                        i, j = rge.index(*feature['geometry']['coordinates'])
                        assert((i, j) in index)
                        feature['properties']['AREAKM2'] = index[(i, j)]
                        dst.write(feature)

    rge.close()

def FlowAccumulation(row, col):

    tile = tile_index[(row, col)].gid

    flow_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))
    inlet_shapefile = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_INLET_AREAS.shp' % (row, col))
    output = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_ACC.tif' % (row, col))

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        out = np.full_like(flow, 25e-6, dtype=np.float32)

        with fiona.open(inlet_shapefile) as fs:
            with click.progressbar(fs) as progress:
                for feature in progress:

                    i, j = ds.index(*feature['geometry']['coordinates'])
                    out[i, j] += feature['properties']['AREAKM2']

        flow_accumulation(flow, out)

        click.secho('Save to %s' % output, fg='green')

        profile = ds.profile.copy()
        profile.update(compress='deflate', nodata=0, dtype=np.float32)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def StreamToFeature(row, col, min_drainage):
    """
    DOCME
    """

    flow_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))
    acc_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_ACC.tif' % (row, col))
    output = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_RHT.shp' % (row, col))

    driver='ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('GID', 'int'),
            ('HEAD', 'int')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver=driver, crs=crs, schema=schema)

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)

        with rio.open(acc_raster) as ds2:
            streams = np.int16(ds2.read(1) > min_drainage)

        with fiona.open(output, 'w', **options ) as dst:

            for current, (segment, head) in enumerate(ta.stream_to_feature(streams, flow, ta.ConsoleFeedback())):

                coords = ta.pixeltoworld(np.fliplr(np.int32(segment)), ds.transform, gdal=False)
                dst.write({
                    'type': 'Feature',
                    'geometry': {'type': 'LineString', 'coordinates': coords},
                    'properties': {'GID': current, 'HEAD': 1 if head else 0}
                })

def Starred(args):
    FlowDirection(*args)
    # Outlets(*args)

@click.group()
def cli():
    pass

@cli.command()
def outlets():
    """
    DOCME
    """

    read_tile_index()
    coverage = 0
    
    with click.progressbar(tile_index) as progress:
        for row, col in progress:
            area = Outlets(row, col)
            coverage += (area / (tile_height*tile_width))

    click.secho('Total coverage : %.1f %%' % (coverage/len(tile_index)*100))


@cli.command()
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
@click.option('--quiet/--no-quiet', '-q', default=True, help='Suppress message output ?')
def batch(overwrite, processes, quiet):
    """
    Calcule toutes les tuiles définies dans `TILES.shp`
    """

    from multiprocessing import Pool

    # read_tile_index()

    click.secho('Running %d processes ...' % processes, fg='yellow')
    arguments = (tuple(key) for key in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(Starred, arguments)
        with click.progressbar(pooled, length=len(tile_index)) as progress:
            for _ in progress:
                click.echo('\r')

if __name__ == '__main__':
    cli()
