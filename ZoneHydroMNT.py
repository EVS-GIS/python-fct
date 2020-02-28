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

from console import (
    success,
    info,
    important,
    warning
)

from CalculateFlowDirection import RasterizeStream

def ExtractZoneHydro(bassin, zone, **options):
    """
    Re-délimitation des zones hydrographiques BDC à la résolution du MNT

    1. Rastérise le réseau hydro cartographié en utilisant le même algorithme que celui utilisé dans l'algorithme `StreamToRaster` de la FCT
    2. Calcule le plan de drainage en utilisant la variante de l'algorithme Priority Flood de Lindsay
    3. Réalise une analyse de bassin versant (Watershed Analysis)
    4. Vectorize le polygone correspondant à la zone indiquée
    """

    root = options.get('workdir', '.')
    overwrite = options.get('overwrite', False)
    basename = options.get('output', 'ZONEHYDRO5M.shp')
    flowdir = options.get('flowdir', 'FLOW.tif')
    epsg = options.get('epsg', 2154)

    info('Processing zone %s' % zone)
    info('Working Directory = %s' % root)

    raster_template = os.path.join(root, bassin, zone, 'DEM5M.tif')
    stream_network = os.path.join(root, bassin, zone, 'StreamNetwork.shp')
    output = os.path.join(root, bassin, zone, basename)

    if os.path.exists(output) and not overwrite:
        important('Output already exists : %s' % output)
        return

    feedback = ta.ConsoleFeedback()
    
    with rio.open(raster_template) as ds:

        info('Rasterize Stream Network')

        elevations = ds.read(1)
        transform = ds.transform

        cdzonecnt = itertools.count(1)
        cdzones = defaultdict(lambda: next(cdzonecnt))
        fill_value = 0
        # burn_value = 1
        junctions = np.zeros((ds.height, ds.width), dtype=np.uint8)
        streams = RasterizeStream(elevations, ds.transform, ds.nodata, stream_network, fill_value, cdzones, junctions)

    # filename = os.path.join(root, bassin, zone, 'STREAMS.tif')
    # info('Write %s' % filename)

    # profile = ds.profile.copy()
    # profile.update(compress='deflate')

    # with rio.open(filename, 'w', **profile) as dst:
    #     dst.write(streams, 1)

    # filename = os.path.join(root, bassin, zone, 'JUNCTIONS.tif')
    # info(' Write %s' % filename)

    # profile.update(dtype=np.uint8, nodata=255, compress='deflate')

    # with rio.open(filename, 'w', **profile) as dst:
    #     dst.write(junctions, 1)

    flow_raster = os.path.join(root, bassin, zone, flowdir)

    info('Read Flow Direction from %s' % flow_raster)

    with rio.open(flow_raster) as src:
        flow = src.read(1)

    info('Calculate Watersheds')

    watersheds = np.copy(streams)
    ta.watershed(flow, watersheds, fill_value, feedback)
    feedback.setProgress(100)


    # filename = os.path.join(root, bassin, zone, 'WATERSHEDS.tif')
    # info('Write %s' % filename)

    # profile = ds.profile.copy()
    # profile.update(dtype=np.int32, nodata=0, compress='deflate')

    # with rio.open(filename, 'w', **profile) as dst:
    #     dst.write(np.int32(watersheds), 1)

    info('Vectorize Polygons')

    watersheds = sieve(np.int32(watersheds), 400)

    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('CdZoneHydr', 'str:4')
        ]
    }
    crs = fiona.crs.from_epsg(epsg)

    CdToZones = {v: k for k, v in cdzones.items()}

    polygons = shapes(watersheds, (watersheds == cdzones[zone]), connectivity=8, transform=transform)
    options = dict(
        driver='ESRI Shapefile',
        crs=crs,
        schema=schema
    )

    with fiona.open(output, 'w', **options) as dst:
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

    success('Everything Ok')

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--output', '-o', default='ZONEHYDRO5M.shp', help='Output filename')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, output, workdir, overwrite):
    """
    DOCME
    """

    ExtractZoneHydro(basin, zone, output=output, workdir=workdir, overwrite=overwrite)

@cli.command()
@click.argument('zonelist')
@click.option('--output', '-o', default='ZONEHYDRO5M.shp', help='Output filename')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def batch(zonelist, output, workdir, overwrite):
    """
    DOCME
    """

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    with click.progressbar(zones) as progress:
        for basin, zone in progress:
            
            click.echo('\r')
            ExtractZoneHydro(basin, zone, output=output, workdir=workdir, overwrite=overwrite)

if __name__ == '__main__':
    cli()
