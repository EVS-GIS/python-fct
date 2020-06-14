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

import time
from Command import pretty_time_delta

origin_x = float('inf')
origin_y = float('-inf')
size_x = 5.0*int(parameter('input.width'))
size_y = 5.0*int(parameter('input.height'))

def iniitalize():
    """
    DOCME
    """

    global origin_x
    global origin_y

    for tile in tileindex().values():
        origin_x = min(origin_x, tile.x0)
        origin_y = max(origin_y, tile.y0)

iniitalize()

def xy2tile(x, y):
    """
    DOCME
    """

    row = (origin_y - y) // size_y
    col = (x - origin_x) // size_x
    return int(row), int(col)

def test():

    row = 5
    col = 7
    network = '/media/crousson/Backup/WORK/TestAin/RHT_AIN_05_07.shp'
    elevation_raster = filename('tiled', row=row, col=col)
    flow_raster = filename('flow', row=row, col=col)
    output_relative_flow = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_FLOW_05_07.tif'
    output_relative_shortest = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_SHORTEST_05_07.tif'
    output_flow_distance = '/media/crousson/Backup/WORK/TestAin/AIN_FLOWDIST_05_07.tif'
    output_shortest_distance = '/media/crousson/Backup/WORK/TestAin/AIN_SHORTESTDIST_05_07.tif'
    
    with rio.open(elevation_raster) as ds:

        click.secho('Read elevations', fg='cyan')
        
        elevations = ds.read(1)
        profile = ds.profile.copy()
        height, width = elevations.shape

        def intile(i, j):
            return all([i >= 0, i < height, j >= 0, j < width])

        def mk_reference():

            reference = np.full_like(elevations, ds.nodata)

            with fiona.open(network) as fs:
                for feature in fs:
                
                    coords = np.array([
                        (i, j)
                        for i, j in ta.worldtopixel(
                            np.array(feature['geometry']['coordinates'], dtype='float32'),
                            ds.transform,
                            gdal=False)
                        if intile(i, j)
                    ])
                
                    reference[coords[:, 0], coords[:, 1]] = elevations[coords[:, 0], coords[:, 1]]

            return reference

        click.secho('Flow Valley Bottom', fg='cyan')

        start_time = time.process_time()

        with rio.open(flow_raster) as flowds:
            flow = flowds.read(1)

        reference = mk_reference()
        distance = np.zeros_like(elevations)
        speedup.valley_bottom_flow(flow, reference, elevations, ds.nodata, distance, 15.0, 1000.0)
        
        relative = elevations - reference
        relative[reference == ds.nodata] = ds.nodata
        distance[reference == ds.nodata] = ds.nodata

        with rio.open(output_relative_flow, 'w', **profile) as dst:
            dst.write(relative, 1)

        with rio.open(output_flow_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        click.echo('Process time: %s' % pretty_time_delta(time.process_time() - start_time))

        click.secho('Shortest Valley Bottom', fg='cyan')

        start_time = time.process_time()

        reference = mk_reference()
        distance = np.zeros_like(elevations)
        speedup.valley_bottom_shortest(reference, elevations, ds.nodata, distance, 1000.0)
        
        relative = elevations - reference
        distance[(reference == ds.nodata) | (relative > 15.0)] = ds.nodata
        relative[(reference == ds.nodata) | (relative > 15.0)] = ds.nodata

        with rio.open(output_relative_shortest, 'w', **profile) as dst:
            dst.write(relative, 1)

        with rio.open(output_shortest_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        click.echo('Process time: %s' % pretty_time_delta(time.process_time() - start_time))

if __name__ == '__main__':
    test()
