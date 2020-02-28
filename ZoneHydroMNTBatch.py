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

SUCCESS = 'green'
INFO = 'cyan'
WARNING = 'yellow'
ERROR = 'red'

ci = [-1, -1,  0,  1,  1,  1,  0, -1]
cj = [ 0,  1,  1,  1,  0, -1, -1, -1]

def StreamToRaster(template, nodata, shapefile, fill, cdzones, junctions=None):
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
                out[row, col] = value

        with fiona.open(shapefile) as fs:
            for feature in fs:

                gid = feature['properties']['GID']
                hack = feature['properties']['HACK']
                cdzone = feature['properties']['CDZONEHYDR']
                cdzoneidx = cdzones[cdzone]
                geom = feature['geometry']['coordinates']
                linestring = np.fliplr(ta.worldtopixel(np.float32(geom), ds.transform, gdal=False))

                for a, b in zip(linestring[:-2], linestring[1:-1]):
                    for col, row in rasterize_linestring(a, b):
                        if isdata(col, row):
                            for x in range(8):
                                ix = row + ci[x]
                                jx = col + cj[x]
                                if not isdata(jx, ix) or raster[ix, jx] == nodata:
                                    break
                            else:
                                set_data(row, col, gid, cdzoneidx, hack)

                a = linestring[-2]
                b = linestring[-1]
                for col, row in rasterize_linestring(a, b):
                    if isdata(col, row):
                        set_data(row, col, gid, cdzoneidx, hack)


        if junctions is not None:

            with fiona.open(shapefile) as fs:
                for feature in fs:

                    gid = feature['properties']['GID']
                    hack = feature['properties']['HACK']
                    geom = feature['geometry']['coordinates']
                    linestring = np.fliplr(ta.worldtopixel(np.float32(geom), ds.transform, gdal=False))

                    for i in range(linestring.shape[0]-1, -1, -1):

                        px, py = linestring[i]

                        if not isdata(px, py):
                            continue

                        if streams[py, px] == gid:
                            junctions[py, px] = 1
                            break

        return out


def ExtractZoneHydro(bassin, zone, root, output, flowdir, epsg, overwrite, overwrite_flow, debug):
    """
    Re-délimitation des zones hydrographiques BDC à la résolution du MNT

    1. Rastérise le réseau hydro cartographié en utilisant le même algorithme que celui utilisé dans l'algorithme `StreamToRaster` de la FCT
    2. Calcule le plan de drainage en utilisant la variante de l'algorithme Priority Flood de Lindsay
    3. Réalise une analyse de bassin versant (Watershed Analysis)
    4. Vectorize le polygone correspondant à la zone indiquée
    """

    logger.info('Processing zone %s' % zone)
    logger.info('Working Directory = %s' % root)

    raster_template = os.path.join(root, bassin, zone, 'DEM5M.tif')
    stream_network = os.path.join(root, bassin, zone, 'StreamNetwork.shp')
    outfilename = os.path.join(root, bassin, zone, output)

    if os.path.exists(outfilename) and not overwrite:
        # logger.warning('Output already exists : %s' % outfilename)
        return

    feedback = ta.SilentFeedback()
    ds = rio.open(raster_template)

    logger.info('Rasterize Stream Network')

    cdzonecnt = itertools.count(1)
    cdzones = defaultdict(lambda: next(cdzonecnt))
    fill_value = 0
    # burn_value = 1
    junctions = np.zeros((ds.height, ds.width), dtype=np.uint8)
    streams = StreamToRaster(raster_template, ds.nodata, stream_network, fill_value, cdzones, junctions)

    if debug:

        filename = os.path.join(root, bassin, zone, 'STREAMS.tif')
        logger.debug('Write %s' % filename)

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(filename, 'w', **profile) as dst:
            dst.write(streams, 1)

        filename = os.path.join(root, bassin, zone, 'JUNCTIONS.tif')
        logger.debug(' Write %s' % filename)

        profile.update(dtype=np.uint8, nodata=255, compress='deflate')

        with rio.open(filename, 'w', **profile) as dst:
            dst.write(junctions, 1)

    flow_raster = os.path.join(root, bassin, zone, flowdir)

    if os.path.exists(flow_raster) and not overwrite_flow:

        logger.info('Read Flow Direction from %s' % flow_raster)

        with rio.open(flow_raster) as src:
            flow = src.read(1)

    else:

        logger.info('Calculate Flow Direction')

        elevations = ds.read(1)
        zdelta = 0.0001
        flow = ta.burnfill(
            elevations,
            streams,
            junctions,
            ds.nodata,
            zdelta,
            feedback=feedback)
        feedback.setProgress(100)

        logger.info('Save to %s' % flow_raster)

        profile = ds.profile.copy()
        profile.update(dtype=np.int16, nodata=-1, compress='deflate')

        with rio.open(flow_raster, 'w', **profile) as dst:
            dst.write(flow, 1)

    logger.info('Calculate Watersheds')

    watersheds = np.copy(streams)
    ta.watershed(flow, watersheds, fill_value, feedback)
    feedback.setProgress(100)

    if debug:

        filename = os.path.join(root, bassin, zone, 'WATERSHEDS.tif')
        logger.debug('Write %s' % filename)

        profile = ds.profile.copy()
        profile.update(dtype=np.int32, nodata=0, compress='deflate')

        with rio.open(filename, 'w', **profile) as dst:
            dst.write(np.int32(watersheds), 1)

    logger.info('Vectorize Polygons')

    watersheds = sieve(np.int32(watersheds), 400)

    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('CdZoneHydr', 'str:4')
        ]
    }
    crs = fiona.crs.from_epsg(epsg)

    CdToZones = {v: k for k, v in cdzones.items()}

    polygons = shapes(watersheds, (watersheds == cdzones[zone]), connectivity=8, transform=ds.transform)
    options = dict(
        driver='ESRI Shapefile',
        crs=crs,
        schema=schema
    )

    with fiona.open(outfilename, 'w', **options) as dst:
        for polygon, value in polygons:
            if value > 0:
                geom = asShape(polygon).buffer(0.0)
                feature = {
                    'geometry': geom.__geo_interface__,
                    'properties': {
                        'CdZoneHydr': CdToZones[value]
                    }
                }
                dst.write(feature)

    logger.info('Everything Ok')

@click.command()
@click.argument('zonelist')
@click.option('--root', type=click.Path(True, False, True), default='/media/crousson/Backup/PRODUCTION/ZONEHYDR', help='Working Directory')
@click.option('--output', default='ZONEHYDRO5M.shp', help='Output base filename')
@click.option('--flowdir', default='FLOW.tif', help='Flow Direction filename')
@click.option('--epsg', default=2154, help='Output Coordinate Reference System')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--overwrite-flow', '-wf', 'overwrite_flow', default=False, help='Overwrite existing Flow Dir raster ?', is_flag=True)
@click.option('--debug', '-g', default=False, help='Debug mode : output intermediate results', is_flag=True)
def ExtractZoneHydroBatch(zonelist, root, output, flowdir, epsg, overwrite, overwrite_flow, debug):
    """
    Re-délimitation des zones hydrographiques BDC à la résolution du MNT
    (Batch version)
    """

    with open(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    def show_item(item):

        if item is not None:
            return item[1]

        return ''

    with click.progressbar(zones, label='Extracting Zone Hydro', item_show_func=show_item) as progress:
        for basin, zone in progress:
            ExtractZoneHydro(basin, zone, root, output, flowdir, epsg, overwrite, overwrite_flow, debug)


if __name__ == '__main__':
    ExtractZoneHydroBatch()
