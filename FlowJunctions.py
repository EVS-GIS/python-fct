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
from collections import defaultdict
import itertools
import click
import click_log

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

def RasterizeStream(template, nodata, shapefile, fill, cdzones):
    """
    Rastérisation du réseau hydrographique cartographié.
    """

    with rio.open(template) as ds:

        raster = ds.read(1)
        streams = np.zeros((ds.height, ds.width), dtype=np.uint32)
        priorities = np.zeros((ds.height, ds.width), dtype=np.uint32)
        out = np.full((ds.height, ds.width), fill, dtype=np.float32)

        def isdata(px, py):
            """
            True if (py, px) is a valid pixel coordinate
            """

            return px >= 0 and py >= 0 and px < ds.width and py < ds.height

        def set_data(row, col, gid, value, priority):
            """
            Set Pixel Value to Line Primary Field
            """

            current_priority = priorities[row, col]

            if current_priority == 0 or priority < current_priority:
                # Override with the smallest ID
                streams[row, col] = gid
                priorities[row, col] = priority
                out[row, col] = gid

        with fiona.open(shapefile) as fs:

            for feature in fs:

                gid = feature['properties']['GID']
                hack = feature['properties']['HACK']
                cdzone = feature['properties']['CDZONEHYDR']
                cdzoneidx = cdzones[cdzone]
                geom = feature['geometry']['coordinates']
                linestring = np.fliplr(ta.worldtopixel(np.float32(geom), ds.transform, gdal=False))

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

        junctions = list()

        with fiona.open(shapefile) as fs:
            for feature in fs:

                gid = feature['properties']['GID']
                hack = feature['properties']['HACK']
                geom = feature['geometry']['coordinates']
                linestring = np.fliplr(ta.worldtopixel(np.float32(geom), ds.transform, gdal=False))

                pixels = list()

                for a, b in zip(linestring[:-1], linestring[1:]):
                    for col, row in rasterize_linestring(a, b):
                        if isdata(col, row) and streams[row, col] == gid:
                            pixels.append((row, col))

                if pixels:
                    py, px = pixels[-1]
                    junctions.append((py, px, gid))

                # for i in range(linestring.shape[0]-1, -1, -1):

                #     px, py = linestring[i]

                #     if not isdata(px, py):
                #         continue

                #     if streams[py, px] == gid:
                #         junctions.append((py, px, gid))
                #         break

        return out, junctions

def FlowJunctions(bassin, zone, **options):
    """
    Calcule le plan de drainage,
    en tenant compte du réseau hydrographique cartographié.
    """

    root = options.get('workdir', '.')
    overwrite = options.get('overwrite', False)
    basename = options.get('output', 'JUNCTIONS.shp')
    epsg = options.get('epsg', 2154)

    info('Processing zone %s' % zone)
    info('Working Directory = %s' % root)

    raster_template = os.path.join(root, bassin, zone, 'DEM5M.tif')
    stream_network = os.path.join(root, bassin, zone, 'StreamNetwork.shp')
    output = os.path.join(root, bassin, zone, basename)

    if os.path.exists(output) and not overwrite:
        important('Output already exists : %s' % output)
        return

    with rio.open(raster_template) as ds:

        click.secho('Rasterize Stream Network', fg='cyan')

        cdzonecnt = itertools.count(1)
        cdzones = defaultdict(lambda: next(cdzonecnt))
        fill_value = 0
        # burn_value = 1
        streams, junctions = RasterizeStream(raster_template, ds.nodata, stream_network, fill_value, cdzones)

        crs = fiona.crs.from_epsg(epsg)
        schema = {
            'geometry': 'Point',
            'properties': [
                ('CDZONEHYDR', 'str:4'),
                ('STREAMID', 'int')
            ]
        }
        options = dict(
            driver='ESRI Shapefile',
            crs=crs,
            schema=schema
        )

        with fiona.open(output, 'w', **options) as dst:
            for i, j, gid in junctions:
                geom = {
                    'type': 'Point',
                    'coordinates': ds.xy(i, j)
                }
                props = {
                    'CDZONEHYDR': zone,
                    'STREAMID': gid
                }
                dst.write({'geometry': geom, 'properties': props})

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--output', '-o', default='JUNCTIONS.shp', help='Output filename')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, output, workdir, overwrite):

    FlowJunctions(basin, zone, output=output, workdir=workdir, overwrite=overwrite)

@cli.command()
@click.argument('zonelist')
@click.option('--output', '-o', default='JUNCTIONS.shp', help='Output filename')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def batch(zonelist, output, workdir, overwrite):

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    with click.progressbar(zones) as progress:
        for basin, zone in progress:
            
            click.echo('\r')
            FlowJunctions(basin, zone, output=output, workdir=workdir, overwrite=overwrite)

if __name__ == '__main__':
    cli()