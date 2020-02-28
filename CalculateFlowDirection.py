#!/usr/bin/env python
# coding: utf-8

"""
Re-délimitation des zones hydrographiques BDC à la résolution du MNT

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
from collections import defaultdict, Counter
import itertools
import click

import numpy as np

import rasterio as rio
from rasterio.features import (
    shapes,
    sieve
)

from shapely.geometry import asShape
import fiona
import fiona.crs

import terrain_analysis as ta
from rasterize import rasterize_linestring

import logging
logger = logging.getLogger(__name__)

from console import (
    success,
    info,
    important,
    warning
)

ci = [-1, -1,  0,  1,  1,  1,  0, -1]
cj = [ 0,  1,  1,  1,  0, -1, -1, -1]

def AdjustStreamElevations(elevations, shapefile, transform, nodata):
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
                                elevations[py, px] = zmin
                                count += 1
                            else:
                                zmin = z

            indegree[node] -= 1

            if indegree[node] == 0:
                queue.append(node)

    info('Adjusted %d pixels' % count)

def RasterizeStream(raster, transform, nodata, shapefile, fill, cdzones, junctions=None):
    """
    Rastérisation du réseau hydrographique cartographié.
    """

    height, width = raster.shape
    streams = np.zeros((height, width), dtype=np.uint32)
    priorities = np.zeros((height, width), dtype=np.uint32)
    out = np.full((height, width), fill, dtype=np.float32)

    def isdata(px, py):
        """
        True if (py, px) is a valid pixel coordinate
        """

        return px >= 0 and py >= 0 and px < width and py < height

    def set_data(row, col, gid, value, priority):
        """
        Set Pixel Value to Line Primary Field
        """

        current_priority = priorities[row, col]

        if current_priority == 0 or priority < current_priority:
            # Override with the smallest ID
            streams[row, col] = gid
            priorities[row, col] = priority
            out[row, col] = value

    with fiona.open(shapefile) as fs:

        for feature in fs:

            gid = feature['properties']['GID']
            hack = feature['properties']['HACK']
            cdzone = feature['properties']['CDZONEHYDR']
            cdzoneidx = cdzones[cdzone]
            geom = feature['geometry']['coordinates']
            linestring = np.fliplr(ta.worldtopixel(np.float32(geom), transform, gdal=False))

            burn_on = False

            for a, b in zip(linestring[:-1], linestring[1:]):
                for col, row in rasterize_linestring(a, b):
                    if isdata(col, row) and raster[row, col] != nodata:
                        for x in range(8):
                            ix = row + ci[x]
                            jx = col + cj[x]
                            if not isdata(jx, ix) or raster[ix, jx] == nodata:
                                # on arrive au bord
                                # on laisse burn_on à sa valeur précédente
                                # si on était à l'extérieur, on n'écrit pas le pixel
                                # si on était à l'intérieur, on écrit le pixel
                                # qui est un potentiel exutoire
                                break
                        else:
                            # on n'est pas au bord
                            burn_on = True
                        if burn_on:
                            set_data(row, col, gid, cdzoneidx, hack)
                    else:
                        burn_on = False

            # a = linestring[-2]
            # b = linestring[-1]
            # for col, row in rasterize_linestring(a, b):
            #     if isdata(col, row):
            #         set_data(row, col, gid, cdzoneidx, hack)


        if junctions is not None:

            with fiona.open(shapefile) as fs:
                for feature in fs:

                    gid = feature['properties']['GID']
                    hack = feature['properties']['HACK']
                    geom = feature['geometry']['coordinates']
                    linestring = np.fliplr(ta.worldtopixel(np.float32(geom), transform, gdal=False))

                    pixels = list()

                    for a, b in zip(linestring[:-1], linestring[1:]):
                        for col, row in rasterize_linestring(a, b):
                            if isdata(col, row) and streams[row, col] == gid:
                                pixels.append((row, col))

                    if pixels:
                        py, px = pixels[-1]
                        # junctions.append((py, px, gid))
                        junctions[py, px] = 1

        return out

def CaclulateFlowDirection(bassin, zone, **options):
    """
    Calcule le plan de drainage,
    en tenant compte du réseau hydrographique cartographié.
    """

    root = options.get('workdir', '.')
    overwrite = options.get('overwrite', False)
    basename = options.get('output', 'FLOW.tif')
    output_streams = options.get('streams', False)
    output_junctions = options.get('junctions', False)

    info('Processing zone %s' % zone)
    info('Working Directory = %s' % root)

    raster_template = os.path.join(root, bassin, zone, 'DEM5M.tif')
    stream_network = os.path.join(root, bassin, zone, 'StreamNetwork.shp')
    output = os.path.join(root, bassin, zone, basename)

    if os.path.exists(output) and not overwrite:
        important('Output already exists : %s' % output)
        return

    with rio.open(raster_template) as ds:

        info('Rasterize Stream Network')

        elevations = ds.read(1)

        info('Adjust Stream Elevations')
        AdjustStreamElevations(elevations, stream_network, ds.transform, ds.nodata)

        cdzonecnt = itertools.count(1)
        cdzones = defaultdict(lambda: next(cdzonecnt))
        fill_value = 0
        # burn_value = 1
        junctions = np.zeros((ds.height, ds.width), dtype=np.uint8)
        streams = RasterizeStream(elevations, ds.transform, ds.nodata, stream_network, fill_value, cdzones, junctions)

        if output_streams:

            filename = os.path.join(root, bassin, zone, 'STREAMS.tif')
            important('Write %s' % filename)

            profile = ds.profile.copy()
            profile.update(compress='deflate')

            with rio.open(filename, 'w', **profile) as dst:
                dst.write(streams, 1)

        if output_junctions:

            filename = os.path.join(root, bassin, zone, 'JUNCTIONS.tif')
            important('Write %s' % filename)

            profile = ds.profile.copy()
            profile.update(dtype=np.uint8, nodata=255, compress='deflate')

            with rio.open(filename, 'w', **profile) as dst:
                dst.write(junctions, 1)

        info('Calculate Flow Direction')

        feedback = ta.ConsoleFeedback()
        
        zdelta = 0.0001
        flow = ta.burnfill(
            elevations,
            streams,
            junctions,
            ds.nodata,
            zdelta,
            feedback=feedback)
        feedback.setProgress(100)

        logger.info('Save to %s' % output)

        profile = ds.profile.copy()
        profile.update(dtype=np.int16, nodata=-1, compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(flow, 1)

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--output', '-o', default='FLOW.tif', help='Output filename')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--streams/--no-streams', default=False, help='Output stream raster ?')
def zone(basin, zone, output, workdir, overwrite, streams):

    CaclulateFlowDirection(basin, zone, output=output, workdir=workdir, overwrite=overwrite, streams=streams)

@cli.command()
@click.argument('zonelist')
@click.option('--output', '-o', default='FLOW.tif', help='Output filename')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def batch(zonelist, output, workdir, overwrite):

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    with click.progressbar(zones) as progress:
        for basin, zone in progress:
            
            click.echo('\r')
            CaclulateFlowDirection(basin, zone, output=output, workdir=workdir, overwrite=overwrite)

if __name__ == '__main__':
    cli()
