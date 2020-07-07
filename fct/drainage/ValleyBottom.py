# coding: utf-8

"""
Valley Bottom

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
import time

import numpy as np
import click

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..config import config
from ..cli import starcall, pretty_time_delta
from .. import terrain_analysis as ta
from .. import speedup
from ..tileio import PadRaster

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

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

def ReadSeeds(axis):

    shapefile = os.path.join(workdir, 'GLOBAL', 'RHTS_TILED.shp')

    def accept(feature):
        properties = feature['properties']
        return properties['AXIS'] == axis

    with fiona.open(shapefile) as fs:
        for feature in fs:
            if accept(feature):

                row = feature['properties']['ROW']
                col = feature['properties']['COL']

                for x, y, z in feature['geometry']['coordinates']:
                    yield x, y, None, 0.0, row, col


def TileValleyBottom(axis, row, col, seeds):

    output_flow_relativez = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'TILES', 'FLOW_RELZ_%02d_%02d.tif' % (row, col))
    output_flow_distance = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'TILES', 'FLOW_DIST_%02d_%02d.tif' % (row, col))

    elevations, profile = PadRaster(row, col, 'tiled', 1)
    transform = profile['transform']
    nodata = profile['nodata']
    flow, _ = PadRaster(row, col, 'flow', 1)

    height, width = elevations.shape

    def intile(i, j):
        return all([i >= 0, i < height, j >= 0, j < width])

    reference = distance = None

    if os.path.exists(output_flow_relativez):
        with rio.open(output_flow_relativez) as ds:
            if ds.height == height and ds.width == width:
                relative = ds.read(1)
                reference = elevations - relative
                reference[relative == ds.nodata] = nodata
                del relative

    if reference is None:
        reference = np.full_like(elevations, nodata)

    if os.path.exists(output_flow_distance):
        with rio.open(output_flow_distance) as ds:
            if ds.height == height and ds.width == width:
                distance = ds.read(1)
                distance[distance == ds.nodata] = 0.0

    if distance is None:
        distance = np.zeros_like(elevations)

    for x, y, z, dist in seeds:

        i, j = ta.index(x, y, transform)
        
        if intile(i, j):

            if z is None:
                reference[i, j] = elevations[i, j]
                distance[i, j] = 0.0
            else:
                reference[i, j] = z
                distance[i, j] = dist

    result = np.copy(reference)
    speedup.valley_bottom_flow(flow, result, elevations, nodata, distance, 15.0, 1000.0)
    
    relative = elevations - result
    relative[result == nodata] = nodata
    distance[result == nodata] = nodata

    spillovers = list()

    def attribute(pixel):
        """
        Returns value, dest. row, dest. col for pixel (i, j)
        """

        i, j = pixel
        x, y = ta.xy(i, j, transform)

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

        return x, y, result[pixel], distance[pixel], prow, pcol

    for pixel in border(height, width):
        if reference[pixel] == nodata and result[pixel] != nodata:
            
            spillovers.append(attribute(pixel))

    output_flow_relativez += '.tmp'
    output_flow_distance += '.tmp'

    with rio.open(output_flow_relativez, 'w', **profile) as dst:
        dst.write(relative, 1)

    with rio.open(output_flow_distance, 'w', **profile) as dst:
        dst.write(distance, 1)

    return spillovers, (output_flow_relativez, output_flow_distance)

def ValleyBottomIteration(spillovers, axis, processes=1):

    attributes = itemgetter(0, 1, 2, 3)
    tile = itemgetter(4, 5)

    spillovers = sorted(spillovers, key=tile)
    tiles = itertools.groupby(spillovers, key=tile)

    g_spillover = list()

    if processes == 1:

        for (row, col), seeds in tiles:

            seeds = [attributes(seed) for seed in seeds]
            t_spillover, outputs = TileValleyBottom(axis, row, col, seeds)
            g_spillover.extend(t_spillover)

            for tmpfile in outputs:
                os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    else:

        kwargs = dict()
        arguments = list()
        tmpfiles = list()

        for (row, col), seeds in tiles:
            seeds = [attributes(seed) for seed in seeds]
            arguments.append((TileValleyBottom, axis, row, col, seeds, kwargs))

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)
            for t_spillover, outputs in pooled:
                g_spillover.extend(t_spillover)
                tmpfiles.extend(outputs)

            # with click.progressbar(pooled, length=len(arguments)) as bar:
            #     for t_spillover in bar:
            #         g_spillover.extend(t_spillover)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))

    return g_spillover

def CropAndScaleRasterTile(rasterfile, scale=1.0, padding=1):

    with rio.open(rasterfile) as ds:

        profile = ds.profile.copy()
        transform = ds.transform * ds.transform.translation(padding, padding)
        height = ds.height - 2*padding
        width = ds.width - 2*padding
        data = ds.read(1)
        mask = (data == ds.nodata)
        data = scale * data
        data[mask] = ds.nodata

    profile.update(transform=transform, height=height, width=width)
    with rio.open(rasterfile, 'w', **profile) as dst:
        dst.write(data[padding:-padding, padding:-padding], 1)

def CropAndScale(axis, tiles, processes=1):

    def rasters(row, col):

        yield os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'TILES', 'FLOW_RELZ_%02d_%02d.tif' % (row, col)), 1.0
        yield os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'TILES', 'FLOW_DIST_%02d_%02d.tif' % (row, col)), 5.0

    if processes == 1:

        with click.progressbar(tiles) as processing:
            for row, col in processing:
                for rasterfile, scale in rasters(row, col):
                    CropAndScaleRasterTile(rasterfile, scale)

    else:

        kwargs = dict(padding=1)
        arguments = [
            (CropAndScaleRasterTile, rasterfile, scale, kwargs)
            for row, col in tiles
            for rasterfile, scale in rasters(row, col)
        ]

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)
            with click.progressbar(pooled, length=len(arguments)) as processing:
                for _ in processing:
                    pass

def ValleyBottom(axis, processes=1):

    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'TILES.csv')

    g_tiles = set()
    tile = itemgetter(4, 5)
    count = 0

    seeds = list(ReadSeeds(axis))

    while seeds:

        tiles = set([tile(s) for s in seeds])
        g_tiles.update(tiles)
        count += 1
        click.echo('Step %02d -- %d tiles, %d spillovers' % (count, len(tiles), len(seeds)))

        seeds = ValleyBottomIteration(seeds, axis, processes)

    click.echo('Done after %d iterations' % count)

    click.secho('Crop and scale output tiles', fg='cyan')
    CropAndScale(axis, g_tiles, processes)

    click.secho('Save axis tiles list', fg='cyan')
    with open(output, 'w') as fp:
        for row, col in sorted(g_tiles):
            fp.write('%d,%d,%d\n' % (axis, row, col))
