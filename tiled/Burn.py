#!/usr/bin/env python
# coding: utf-8

from collections import defaultdict, Counter
from functools import wraps
import numpy as np
import click

import rasterio as rio

import fiona
import terrain_analysis as ta

from config import tileindex, filename

def rasterize_linestring(a, b):
    """
    Returns projected segment
    as a sequence of (px, py) coordinates.

    See https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm

    Parameters
    ----------

    a, b: vector of coordinate pair
        end points of segment [AB]

    Returns
    -------

    Generator of (x, y) coordinates
    corresponding to the intersection of raster cells with segment [AB],
    yielding one data point per intersected cell.
    """

    dx = abs(b[0] - a[0])
    dy = abs(b[1] - a[1])

    if dx > 0 or dy > 0:

        if dx > dy:
            count = dx
            dx = 1.0
            dy = dy / count
        else:
            count = dy
            dy = 1.0
            dx = dx / count

        if a[0] > b[0]:
            dx = -dx
        if a[1] > b[1]:
            dy = -dy

        x = float(a[0])
        y = float(a[1])
        i = 0

        while i < count+1:

            yield int(round(x)), int(round(y))

            x = x + dx
            y = y + dy
            i += 1

    else:

        yield a[0], a[1]

def AdjustStreamElevations(elevations, shapefile, transform, nodata, delta=0):
    """
    Ensure monotonous decreasing elevation profile
    along stream network.
    """

    geometries = []
    graph = defaultdict(list)
    indegree = Counter()

    height, width = elevations.shape

    def isdata(px, py):
            """
            True if (py, px) is a valid pixel coordinate
            """

            return px >= 0 and py >= 0 and px < width and py < height

    with fiona.open(shapefile) as fs:
        for feature in fs:

            geom = feature['geometry']['coordinates']
            linestring = np.fliplr(ta.worldtopixel(np.float32(geom), transform, gdal=False))
            a = feature['properties']['NODEA']
            b = feature['properties']['NODEB']

            idx = len(geometries)
            geometries.append(linestring)

            graph[a].append((b, idx))
            indegree[b] += 1

    queue = [node for node in graph if indegree[node] == 0]
    count = 0

    while queue:

        source = queue.pop(0)

        for node, idx in graph[source]:

            geom = geometries[idx]
            zmin = float('inf')

            for a, b in zip(geom[:-1], geom[1:]):
                for px, py in rasterize_linestring(a, b):
                    if isdata(px, py):

                        z = elevations[py, px]
                        
                        if z != nodata:

                            if z > zmin:
                                count += 1
                            else:
                                zmin = z
                            
                            elevations[py, px] = zmin + delta

            indegree[node] -= 1

            if indegree[node] == 0:
                queue.append(node)

def BurnTile(row, col, delta=-1.0, **kwargs):
    """
    DOCME
    """

    elevation_raster = filename('patched', row=row, col=col)
    stream_network = filename('stream_network', row=row, col=col)
    output = filename('burnt', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        elevations = ds.read(1)
        AdjustStreamElevations(elevations, stream_network, ds.transform, ds.nodata, delta)

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(elevations, 1)

def starcall(args):
    """
    Invoke first arg function with all other arguments.
    """

    fun = args[0]
    return fun(*args[1:-1], **args[-1])

def batch(group, tilefun):
    """
    Define a new Click command within `group` as a Multiprocessing wrapper.
    This command will process tiles in parallel
    using function `tilefun`.
    """

    def decorate(fun):
        """
        Decorator function
        """

        @group.command()
        @click.option('--processes', '-j', default=1, help="Execute j parallel processes")
        @click.option('--progress', '-p', default=False, help="Display progress bar", is_flag=True)
        @wraps(fun)
        def decorated(processes, progress, **kwargs):
            """
            Multiprocessing wrapper
            See https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
            """

            from multiprocessing import Pool

            click.secho('Running %d processes ...' % processes, fg='yellow')

            tile_index = fun()
            arguments = ([tilefun, row, col, kwargs] for row, col in tile_index)

            with Pool(processes=processes) as pool:

                pooled = pool.imap_unordered(starcall, arguments)

                if progress:

                    with click.progressbar(pooled, length=len(tile_index)) as bar:
                        for _ in bar:
                            click.echo('\n\r')

                else:

                    for _ in pooled:
                        pass

        return decorated

    return decorate

@click.group()
def cli():
    pass

# def TestTile(row, col, delta, **kwargs):
#     import os
#     import time
#     print(os.getpid(), row, col, delta)
#     time.sleep(1)

# @batch(cli, TestTile)
# @click.option('--delta', default=-1.0)
# def test():
#     """
#     Print arguments and exit
#     """
#     tiles = dict()
#     for i in range(10):
#         tiles[0, i] = i
#     return tiles

@batch(cli, BurnTile)
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--delta', default=-1.0)
def burn():
    """
    Burn Stream Network
    """
    return tileindex()


if __name__ == '__main__':
    cli()