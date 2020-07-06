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


    flow_raster1 = '/var/local/fct/RMC/FLOW_RGE5M_TILES.vrt'
    acc_raster1 = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    flow1ds = rio.open(flow_raster1)
    acc1ds = rio.open(acc_raster1)
    
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
            acc1_data = ds.read(1)

        with rio.open(flow_raster1) as ds:
            profile = ds.profile.copy()
            flow1_data = ds.read(1)
            height, width = flow1.shape

    def intile(i, j):
         return all([i >= 0, i < height, j >= 0, j < width])

    def flow1(i, j):

        if intile(i, j):
            return flow1_data[i, j]
        
        x, y = ds.xy(i, j)
        return int(next(flow1ds.sample([(x, y)], 1)))

    def acc1(i, j):

        if intile(i, j):
            return acc1_data[i, j]
        
        x, y = ds.xy(i, j)
        return float(next(acc1ds.sample([(x, y)], 1)))

    def streams1(i, j):
        return acc1(i, j) > min_drainage

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

        # TODO handle confluence, select max upstream acc

        max_acck = min_drainage
        max_ijk = None

        for k in range(8):

            ik, jk = i + ci[k], j + cj[k]
            
            if flow1(ik, jk) == upward[k]:
                acck = acc1(ik, jk)
                if acck > max_acck:
                    max_acck = acck
                    max_ijk = (ik, jk)

        if max_ijk is not None:

            i, j = max_ijk

            if not intile(i, j):
                x, y = ds.xy(i, j)
                row, col = xy2tile(x, y)
                read_tile(row, col)
                i, j = ds.index(x, y)

        else:

            raise ValueError('No match for (%f, %f)' % (x0, y0))

    while streams1(i, j):

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

    while not streams1(i, j):

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

    flow1ds.close()
    acc1ds.close()
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