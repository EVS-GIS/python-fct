# coding: utf-8

"""
Watershed Analysis

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
import itertools
from operator import itemgetter
from multiprocessing import Pool

import numpy as np
import click

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from config import tileindex, filename, parameter
from Command import starcall
import terrain_analysis as ta
import speedup
from tileio import PadRaster

origin_x = float('inf')
origin_y = float('-inf')
size_x = 5.0*int(parameter('input.width'))
size_y = 5.0*int(parameter('input.height'))

def initialize():
    """
    DOCME
    """

    global origin_x
    global origin_y

    for tile in tileindex().values():
        origin_x = min(origin_x, tile.x0)
        origin_y = max(origin_y, tile.y0)

initialize()

def xy2tile(x, y):
    """
    DOCME
    """

    row = (origin_y - y) // size_y
    col = (x - origin_x) // size_x
    return int(row), int(col)

def border(height, width):
    """
    Generate a sequence of pixel (row, col)
    over the border of an array of shape (height, width)
    """

    offset = 0
    for i in (0, height-1):
        for j in range(offset, width+offset-1):
            yield i, j
        offset = 1

    offset = 1
    for j in (0, width-1):
        for i in range(offset, height+offset-1):
            yield i, j
        offset = 0

def Watershed(row, col, seeds, wid=1, tmp=''):
    """
    seeds: (x, y, value, distance)
    """

    flow, profile = PadRaster(row, col, 'flow', padding=1)
    transform = profile['transform']
    destination = filename('watershed-u', row=row, col=col, wid=wid)
    height, width = flow.shape

    if os.path.exists(destination):
        # with rio.open(destination) as ds:
        #     ds.read(1, out=out[1:-1, 1:-1])
        data, _ = PadRaster(row, col, 'watershed-u', padding=1, wid=wid)
    else:
        data = np.zeros_like(flow, dtype='float32')

    out = np.copy(data)
    coord = itemgetter(0, 1)
    value = itemgetter(2)
    pixels = ta.worldtopixel(np.array([coord(seed) for seed in seeds], dtype='float32'), transform, gdal=False)
    values = np.array([value(seed) for seed in seeds], dtype='float32')
    out[pixels[:, 0], pixels[:, 1]] = values
    # ta.watershed(flow, out, 0)
    speedup.watershed(flow, out, 0)

    spillover = list()
    for i, j in border(height, width):
        if data[i, j] == 0 and out[i, j] != 0:
            spillover.append((i, j))

    def attribute(p):
        """
        Returns value, dest. row, dest. col for pixel (i, j)
        """

        i, j = p

        if i == 0:
            prow = row - 1
        elif i == height-1:
            prow = row + 1
        else:
            prow = row

        if j == 0:
            pcol = col - 1
        elif j == width-1:
            pcol = col + 1
        else:
            pcol = col

        return out[p], prow, pcol

    if spillover:
        points = ta.pixeltoworld(np.array(spillover, dtype='int32'), transform, gdal=False)
        spillover = [tuple(points[k]) + attribute(p) for k, p in enumerate(spillover)]

    out = out[1:-1, 1:-1]
    height, width = out.shape
    transform = transform * transform.translation(1, 1)
    profile.update(dtype='float32', height=height, width=width, transform=transform, nodata=0)

    with rio.open(destination + tmp, 'w', **profile) as dst:
        dst.write(out, 1)

    return spillover, destination + tmp

def VectorizeTile(wid, row, col):
    """
    DOCME
    """

    rasterfile = filename('watershed-u', row=row, col=col, wid=wid)

    if os.path.exists(rasterfile):

        with rio.open(rasterfile) as ds:
            watershed = ds.read(1)
            transform = ds.transform

        polygons = features.shapes(
            watershed,
            connectivity=4,
            transform=transform)

        return [polygon for polygon, value in polygons if value == 1], row, col

    else:

        return [], row, col

def Vectorize(wid, processes=1):
    """
    DOCME
    """

    output = filename('watershed', wid=wid)

    epsg = 2154
    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('WATID', 'int:4'),
            ('ROW', 'int:3'),
            ('COL', 'int:3')
        ]
    }
    crs = fiona.crs.from_epsg(epsg)

    options = dict(
            driver='ESRI Shapefile',
            crs=crs,
            schema=schema
        )

    if processes == 1:

        with fiona.open(output, 'w', **options) as dst:
            with click.progressbar(tileindex()) as bar:
                for row, col in bar:
                    polygons, _, _ = VectorizeTile(wid, row, col)
                    for polygon in polygons:
                        geom = asShape(polygon).buffer(0.0)
                        feature = {
                            'geometry': geom.__geo_interface__,
                            'properties': {
                                'WATID': wid,
                                'ROW': row,
                                'COL': col
                            }
                        }
                        dst.write(feature)

    else:

        kwargs = dict()
        arguments = [(VectorizeTile, wid, row, col, kwargs) for row, col in tileindex()]

        with fiona.open(output, 'w', **options) as dst:
            with Pool(processes=processes) as pool:

                pooled = pool.imap_unordered(starcall, arguments)
                with click.progressbar(pooled, length=len(arguments)) as bar:

                    for polygons, row, col in bar:
                        for polygon in polygons:
                            geom = asShape(polygon).buffer(0.0)
                            feature = {
                                'geometry': geom.__geo_interface__,
                                'properties': {
                                    'WATID': wid,
                                    'ROW': row,
                                    'COL': col
                                }
                            }
                            dst.write(feature)

def print_spillover_tiles(spillover):

    tile = itemgetter(3, 4)
    tiles = set([tile(s) for s in spillover])
    print(tiles)

def step(spillover, wid=1, processes=1):

    xy = itemgetter(0, 1)
    value = itemgetter(2)
    tile = itemgetter(3, 4)

    spillover = sorted(spillover, key=tile)
    tiles = itertools.groupby(spillover, key=tile)

    g_spillover = list()
    tmpfiles = list()

    if processes == 1:

        for (row, col), seeds in tiles:
            seeds = [xy(seed) + (value(seed),) for seed in seeds]
            t_spillover, tmpfile = Watershed(row, col, seeds, wid=wid, tmp='.tmp')
            g_spillover.extend(t_spillover)
            tmpfiles.append(tmpfile)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    else:

        kwargs = {'tmp': '.tmp', 'wid': wid}
        arguments = list()

        for (row, col), seeds in tiles:
            seeds = [xy(seed) + (value(seed),) for seed in seeds]
            arguments.append((Watershed, row, col, seeds, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)
            for t_spillover, tmpfile in pooled:
                g_spillover.extend(t_spillover)
                tmpfiles.append(tmpfile)

            # with click.progressbar(pooled, length=len(arguments)) as bar:
            #     for t_spillover in bar:
            #         g_spillover.extend(t_spillover)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    return g_spillover

@click.command()
@click.option('--processes', '-j', default=1)
@click.option('--wid', '-w', default=1)
def test(processes=1, wid=1):

    # River Ain
    # x = 869165.0
    # y = 6523885.0

    # River Rhone
    x = 849265.0
    y = 6250190.0
    row, col = xy2tile(x, y)

    seeds = [(x, y, 1, row, col)]
    count = 0
    tile = itemgetter(3, 4)

    click.secho('Watershed ID = %d' % wid, fg='cyan')
    click.secho('Run %d processes' % processes, fg='yellow')

    while seeds:

        seeds = step(seeds, wid, processes)

        count += 1
        tiles = set([tile(s) for s in seeds])
        click.echo('Step %02d -- %d spillovers, %d tiles' % (count, len(seeds), len(tiles)))

    click.secho('Ok', fg='green')

if __name__ == '__main__':
    test()
