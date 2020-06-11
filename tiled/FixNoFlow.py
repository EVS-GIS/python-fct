# coding: utf-8

"""
Fix Flow Direction for No-Flow Cells

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np
import rasterio as rio
import fiona
from config import tileindex, filename, parameter
import click

origin_x = float('inf')
origin_y = float('-inf')
size_x = 5.0*int(parameter('input.width'))
size_y = 5.0*int(parameter('input.height'))

for tile in tileindex().values():
    origin_x = min(origin_x, tile.x0)
    origin_y = max(origin_y, tile.y0)

def xy2tile(x, y):
    """
    DOCME
    """

    row = (origin_y - y) // size_y
    col = (x - origin_x) // size_x
    return int(row), int(col)

def FixNoFlow(x0, y0, min_drainage=5.0, fix=False):
    """
    DOCME
    """

    ci = [ -1, -1,  0,  1,  1,  1,  0, -1 ]
    cj = [  0,  1,  1,  1,  0, -1, -1, -1 ]
    upward = [ 16,  32,  64,  128,  1,  2,  4,  8 ]

    flow1 = flow2 = None
    streams1 = streams2 = None
    height = width = 0
    ds = None
    profile = None

    flow_raster2 = '/var/local/fct/RMC/FLOW_RGE5M_TILES2.vrt'
    acc_raster2 = '/var/local/fct/RMC/ACC_RGE5M_TILES2.vrt'
    flow2ds = rio.open(flow_raster2)
    acc2ds = rio.open(acc_raster2)

    def read_tile(row, col):
        """
        DOCME
        """

        nonlocal ds
        nonlocal profile
        nonlocal flow1
        nonlocal flow2
        nonlocal streams1
        nonlocal streams2
        nonlocal height
        nonlocal width

        # click.echo('Reading tile (%d, %d)' % (row, col))

        flow_raster1 = filename('flow', row=row, col=col)
        acc_raster1 = filename('acc', row=row, col=col)

        with rio.open(acc_raster1) as ds:
            streams1 = ds.read(1) > min_drainage

        with rio.open(flow_raster1) as ds:
            profile = ds.profile.copy()
            flow1 = ds.read(1)
            height, width = flow1.shape

    def intile(i, j):
         return all([i >= 0, i < height, j >= 0, j < width])

    def flow2(i, j):
        x, y = ds.xy(i, j)
        return int(next(flow2ds.sample([(x, y)], 1)))

    def streams2(i, j):
        x, y = ds.xy(i, j)
        return bool(next(acc2ds.sample([(x, y)], 1)) > min_drainage)

    def write_tile(row, col):

        if fix:

            flow_raster1 = filename('flow', row=row, col=col)

            with rio.open(flow_raster1, 'w', **profile) as dst:
                dst.write(flow1, 1)

    row, col = xy2tile(x0, y0)
    read_tile(row, col)
    i, j = ds.index(x0, y0)

    while not streams2(i, j):

        for k in range(8):

            ix, jx = i + ci[k], j + cj[k]
            if intile(ix, jx) and flow1[ix, jx] == upward[k] and streams1[ix, jx]:
                i, j = ix, jx
                break

        else:

            dsx = ds

            for k in range(8):

                ix, jx = i + ci[k], j + cj[k]
                if not intile(ix, jx):
                    x, y = dsx.xy(ix, jx)
                    rowx, colx = xy2tile(x, y)
                    if not (rowx == row and colx == col):
                        row, col = rowx, colx
                        read_tile(row, col)
                    ix, jx = ds.index(x, y)
                    if flow1[ix, jx] == upward[k] and streams1[ix, jx]:
                        i, j = ix, jx
                        break

            else:

                raise ValueError('No match for (%f, %f)' % (x0, y0))

    while streams1[i, j]:

        direction = flow2(i, j)
        if direction == -1 or direction == 0:
            break

        flow1[i, j] = direction
        k = int(np.log2(direction))
        i, j = i + ci[k], j + cj[k]

        if not intile(i, j):

            write_tile(row, col)
            x, y = ds.xy(i, j)
            row, col = xy2tile(x, y)
            read_tile(row, col)
            i, j = ds.index(x, y)

    while not streams1[i, j]:

        direction = flow2(i, j)
        if direction == -1 or direction == 0:
            break

        flow1[i, j] = direction
        k = int(np.log2(direction))
        i, j = i + ci[k], j + cj[k]

        if not intile(i, j):

            write_tile(row, col)
            x, y = ds.xy(i, j)
            row, col = xy2tile(x, y)
            read_tile(row, col)
            i, j = ds.index(x, y)

    write_tile(row, col)

    flow2ds.close()
    acc2ds.close()

    return ds.xy(i, j)

def test(fix=False):

    noflow = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/NOFLOW_RGE5M_ALL.shp'
    targets = '/media/crousson/Backup/PRODUCTION/RGEALTI/RMC/NOFLOW_TARGETS.shp'

    with fiona.open(noflow) as fs:
        options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)
        with fiona.open(targets, 'w', **options) as dst:
            with click.progressbar(fs) as progress:
                for f in progress:
                    x, y = f['geometry']['coordinates']
                    try:
                        tox, toy = FixNoFlow(x, y, fix=fix)
                        f['geometry']['coordinates'] = [tox, toy]
                        dst.write(f)
                    except ValueError as e:
                        print(f['properties']['GID'], e)
                        continue