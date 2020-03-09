#!/usr/bin/env python
# coding: utf-8

"""
Sequence :

1. FlowDirection (*)
2. Outlets (*)
3. AggregateOutlets

(*) Possibly Parallel Steps

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

import richdem as rd
import speedup
import terrain_analysis as ta

from config import tileindex, filename

tile_height = 7150
tile_width = 9800

def PadElevations(row, col):
    """
    Assemble a 1-pixel padded elevation raster,
    with borders from neighboring tiles.
    """

    tile_index = tileindex()
    elevation_raster = filename('filled', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        height, width = ds.shape
        extended = np.zeros((height+2, width+2), dtype=np.float32)

        extended[1:-1, 1:-1] = ds.read(1)

        # top
        i = row-1
        j = col

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[0, 1:-1] = ds2.read(1, window=Window(0, height-1, width, 1)).reshape(width)

        else:

            extended[0, 1:-1] = ds.nodata

        # bottom
        i = row+1
        j = col

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[-1, 1:-1] = ds2.read(1, window=Window(0, 0, width, 1)).reshape(width)

        else:

            extended[-1, 1:-1] = ds.nodata

        # left
        i = row
        j = col-1

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[1:-1, 0] = ds2.read(1, window=Window(width-1, 0, 1, height)).reshape(height)

        else:

            extended[1:-1, 0] = ds.nodata

        # right
        i = row
        j = col+1

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[1:-1, -1] = ds2.read(1, window=Window(0, 0, 1, height)).reshape(height)

        else:

            extended[1:-1, -1] = ds.nodata

        # top-left
        i = row-1
        j = col-1

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[0, 0] = ds2.read(1, window=Window(width-1, height-1, 1, 1)).item()

        else:

            extended[0, 0] = ds.nodata

        # bottom-left
        i = row+1
        j = col-1

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[-1, 0] = ds2.read(1, window=Window(width-1, 0, 1, 1)).item()

        else:

            extended[-1, 0] = ds.nodata

        # top-right
        i = row-1
        j = col+1

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[0, -1] = ds2.read(1, window=Window(0, height-1, 1, 1)).item()

        else:

            extended[0, -1] = ds.nodata

        # bottom-right
        i = row+1
        j = col+1

        if (i, j) in tile_index:
        
            other_raster = filename('filled', row=i, col=j)
            with rio.open(other_raster) as ds2:
                extended[-1, -1] = ds2.read(1, window=Window(0, 0, 1, 1)).item()

        else:

            extended[-1, -1] = ds.nodata

    return extended

# def FlowDirection(row, col):

#     read_tile_index()

#     output = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))

#     def filename(row, col):
#         return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED2.tif' % (row, col))

#     with rio.open(filename(row, col)) as ds:

#         extended = PadElevations(row, col, filename)

#         extended = rd.rdarray(extended, no_data=ds.nodata)
#         rd.BreachDepressions(extended, True, 'D8')
#         rd.FillDepressions(extended, True, True, 'D8')
#         flow = ta.flowdir(extended, ds.nodata)

#         profile = ds.profile.copy()
#         profile.update(compress='deflate', dtype=np.int16, nodata=-1)

#         with rio.open(output, 'w', **profile) as dst:
#             dst.write(flow[1:-1, 1:-1], 1)

def WallFlats(padded, nodata):

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]

    height, width = padded.shape
    zwall = np.max(padded)
    fixed = 0

    flow = ta.flowdir(padded, nodata)

    # top and bottom
    for ik in [0, height-1]:
        for jk in range(width):
            direction = flow[ik, jk]
            if direction != -1 and direction != 0:
                n = int(np.log2(direction))
                i = ik + ci[n]
                j = jk + cj[n]
                if all([i >= 0, i < height, j >= 0, j < width]):
                    if flow[i, j] == 0 and padded[ik, jk] > padded[i, j]:
                        padded[ik, jk] = zwall
                        fixed += 1

    # left and right
    for jk in [0, width-1]:
        for ik in range(height):
            direction = flow[ik, jk]
            if direction != -1 and direction != 0:
                n = int(np.log2(direction))
                i = ik + ci[n]
                j = jk + cj[n]
                if all([i >= 0, i < height, j >= 0, j < width]):
                    if flow[i, j] == 0 and padded[ik, jk] > padded[i, j]:
                        padded[ik, jk] = zwall
                        fixed += 1

    return fixed


def FlowDirection(row, col):
    """
    DOCME
    """

    # TODO Burn mapped stream network

    elevation_raster = filename('filled', row=row, col=col)
    output = filename('flow', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        padded = PadElevations(row, col)

        WallFlats(padded, ds.nodata)

        # ***********************************************************************
        # Wall flowing border flats

        # flow = ta.flowdir(padded, ds.nodata)
        # labels, outlets = speedup.flat_labels(flow, padded, ds.nodata)
        # notflowing = {k+1 for k, (i, j) in enumerate(outlets) if i == -1 and j == -1}

        # height, width = labels.shape
        # boxes = speedup.flat_boxes(labels)

        # borders = set()
        # for w, (mini, minj, maxi, maxj, count) in boxes.items():
        #     if mini == 0 or minj == 0 or maxi == (height-1) or maxj == (width-1):
        #         if w not in notflowing:
        #             borders.add(w)

        # @np.vectorize
        # def bordermask(x):
        #     return x in borders

        # mask = bordermask(labels)
        # mask[1:-1, 1:-1] = False
        # padded[mask] = np.max(padded)

        # ***********************************************************************

        flow = ta.flowdir(padded, ds.nodata)
        flat_mask, flat_labels = ta.resolve_flat(padded, flow, ta.ConsoleFeedback())
        ta.flat_mask_flowdir(flat_mask, flow, flat_labels)

        # extended = rd.rdarray(flat_mask, no_data=0)
        # rd.FillDepressions(extended, True, True, 'D8')
        
        # extended = rd.rdarray(padded, no_data=ds.nodata)
        # # rd.BreachDepressions(extended, True, 'D8')
        # # rd.ResolveFlats(extended, True)
        # rd.FillDepressions(extended, True, True, 'D8')
        # flow = ta.flowdir(padded, ds.nodata)

        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype=np.int16, nodata=-1)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(flow[1:-1, 1:-1], 1)

def Outlets(row, col):
    """
    DOCME
    """

    tile_index = tileindex()

    crs = fiona.crs.from_epsg(2154)
    driver = 'GeoJSON'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('LCA', 'int'),
            ('TO', 'int'),
            ('TOX', 'float'),
            ('TOY', 'float'),
            ('Z', 'float'),
            ('TOZ', 'float')
        ]
    }
    options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

    # read_tile_index()

    flow_raster = filename('flow', row=row, col=col)

    gid = tile_index[(row, col)].gid
    tiles = defaultdict(list)

    def readz(trow, tcol, x, y):

        with rio.open(filename('filled', row=trow, col=tcol)) as src:
            return float(next(src.sample([(x, y)], 1)))

    with rio.open(flow_raster) as ds:

        height, width = ds.shape
        flow = ds.read(1)
        mask = np.ones_like(flow, dtype=np.uint8)
        outlets, targets = ta.tile_outlets(flow, mask)

        output = filename('outlets', row=row, col=col)

        # with fiona.open(output, 'w', **options) as dst:
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

                # target = tile_index[(row+di, col+dj)].gid
                # (i, j), area = outlets[current]
                # x, y = ds.xy(i, j)
                # tx, ty = ds.xy(ti, tj)
                # outlet_z = readz(row, col, x, y)
                # target_z = readz(row+di, col+dj, tx, ty)

                # geom = {'type': 'Point', 'coordinates': [x, y]}
                # props = {
                #     'TILE': gid,
                #     'LCA': area,
                #     'TO': target,
                #     'TOX': tx,
                #     'TOY': ty,
                #     'Z': outlet_z,
                #     'TOZ': target_z
                # }

                # dst.write({'geometry': geom, 'properties': props})

        cum_area = 0
        skipped = 0

        schema = {
            'geometry': 'Point',
            'properties': [
                ('TILE', 'int'),
                ('LCA', 'int'),
                ('FROM', 'int'),
                ('FROMX', 'float'),
                ('FROMY', 'float')
            ]
        }
        options = dict(driver=driver, crs=crs, schema=schema)

        for trow, tcol in tiles:

            if (trow, tcol) not in tile_index:
                skipped += len(tiles[(trow, tcol)])
                continue

            target = tile_index[(trow, tcol)].gid
            output = filename('inlets', row=trow, col=tcol, gid=gid)

            # if os.path.exists(output):
            #     mode = 'a'
            # else:
            #     mode = 'w'

            with fiona.open(output, 'w', **options) as dst:
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

def AggregateOutlets():
    """
    Aggregate ROW_COL_INLETS_ORIGIN.geojson files
    into one ROW_COL_INLETS.shp shapefile
    """

    tile_index = tileindex()

    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'Point',
        'properties': [
            ('TILE', 'int'),
            ('LCA', 'int'),
            ('FROM', 'int'),
            ('FROMX', 'float'),
            ('FROMY', 'float')
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    with click.progressbar(tile_index) as progress:
        for row, col in progress:

            output = filename('inlets-agg', row=row, col=col)

            with fiona.open(output, 'w', **options) as dst:

                for name in glob.glob(filename('inlets-pat', row=row, col=col)):
                    with fiona.open(name) as fs:
                        
                        for feature in fs:
                            dst.write(feature)

def Starred(args):
    FlowDirection(*args)
    Outlets(*args)

@click.group()
def cli():
    pass

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
    arguments = (tuple(key) for key in tile_index)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(Starred, arguments)
        with click.progressbar(pooled, length=len(tile_index)) as progress:
            for _ in progress:
                click.echo('\r')

@cli.command()
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def aggregate(overwrite):
    AggregateOutlets()

if __name__ == '__main__':
    cli()
