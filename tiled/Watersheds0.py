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
import numpy as np
import rasterio as rio
import fiona
from config import tileindex, filename, parameter
import click

import itertools
from operator import itemgetter

import terrain_analysis as ta

from tileio import PadRaster

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

def Watershed(row, col, wid, seeds):
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
    ta.watershed(flow, out, 0)

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

    with rio.open(destination, 'w', **profile) as dst:
        dst.write(out, 1)

    return spillover


def print_spillover_tiles(spillover):

    tile = itemgetter(3, 4)
    tiles = set([tile(s) for s in spillover])
    print(tiles)

def test():

    # River Ain
    x = 869165.0
    y = 6523885.0
    row, col = xy2tile(x, y)
    print(row, col)
    wid = 1

    seeds = [(x, y, 1)]
    spillover = Watershed(row, col, wid, seeds)
    return spillover

def step(spillover, wid=1):

    xy = itemgetter(0, 1)
    value = itemgetter(2)
    tile = itemgetter(3, 4)

    spillover = sorted(spillover, key=tile)
    tiles = itertools.groupby(spillover, key=tile)

    g_spillover = list()

    for (row, col), seeds in tiles:
        seeds = [xy(seed) + (value(seed),) for seed in seeds]
        t_spillover = Watershed(row, col, wid, seeds)
        g_spillover.extend(t_spillover)

    return g_spillover
