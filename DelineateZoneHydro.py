#!/usr/bin/env python
# coding: utf-8

"""
Dérive un réseau sous-maille
à partir d'un carroyage et d'un plan de drainage
de résolution différente

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
import click
import numpy as np
import itertools
from collections import defaultdict

import terrain_analysis as ta
import rasterio as rio
from rasterio.features import (
    shapes,
    sieve
)
import fiona
from shapely.geometry import asShape

def DelineateZoneHydro(basin, zone, outlets_shapefile, root, overwrite):
    """
    """

    output = os.path.join(root, basin, zone, 'ZONEHYDRO_MNT.shp')
    flow_raster = os.path.join(root, basin, zone, 'FLOW.tif')

    if os.path.exists(output) and not overwrite:
        # click.secho('Output already exists : %s' % output, fg='yellow')
        return

    cdzonecnt = itertools.count(1)
    cdzones = defaultdict(lambda: next(cdzonecnt))

    with rio.open(flow_raster) as ds:

        flow = ds.read(1)
        watersheds = np.zeros_like(flow, dtype=np.float32)

        def isdata(i, j):
            return i >= 0 and i < ds.height and j >=0 and j < ds.width

        with fiona.open(outlets_shapefile) as fs:

            crs = fs.crs
            driver = fs.driver

            for feature in fs:
                
                cdzone = feature['properties']['CDZONEHYDR']
                drainage = feature['properties']['DRAINAGE']

                # if drainage > 0 and drainage < 200000:
                #     continue

                x, y = feature['geometry']['coordinates']
                i, j = ds.index(x, y)

                if isdata(i, j):
                    
                    # print(cdzone)
                    idzone = cdzones[cdzone]
                    watersheds[i, j] = idzone

        fill_value = 0
        feedback = ta.SilentFeedback()
        ta.watershed(flow, watersheds, fill_value, feedback)

        profile = ds.profile.copy()
        profile.update(compress='deflate', nodata=0, dtype=np.float32)
        with rio.open(os.path.join(root, basin, zone, 'WATERSHEDS.tif'), 'w', **profile) as dst:
            dst.write(watersheds, 1)

        watersheds = sieve(np.int32(watersheds), 40)

        CdToZones = {v: k for k, v in cdzones.items()}

        schema = {
            'geometry': 'Polygon',
            'properties': [
                ('CdZoneHydr', 'str:4')
            ]
        }
        polygons = shapes(watersheds, (watersheds == cdzones[zone]), connectivity=8, transform=ds.transform)
        options = dict(driver=driver, crs=crs, schema=schema)

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

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--outlets', default='ZONEHYDRO_OUTLETS.shp')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, outlets, workdir, overwrite):
    """
    DOCME
    """
    
    outlets_shapefile = os.path.join(workdir, outlets)
    DelineateZoneHydro(basin, zone, outlets_shapefile, workdir, overwrite)

def Starred(args):
    """
    Starred version of `function` for use with pool.imap_unordered()
    """

    return DelineateZoneHydro(*args)

@cli.command()
@click.argument('zonelist')
@click.option('--outlets', default='ZONEHYDRO_OUTLETS.shp')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
@click.option('--processes', '-j', default=1, help="Execute j parallel processes")
def batch(zonelist, outlets, workdir, overwrite, processes):
    """
    DOCME
    """

    outlets_shapefile = os.path.join(workdir, outlets)

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    # def display_item(item):
    #     if item:
    #         return item[1]
    #     return '...'

    # with click.progressbar(zones, item_show_func=display_item) as progress:
    #     for basin, zone in progress:

    #         DelineateZoneHydro(basin, zone, workdir, overwrite)

    from multiprocessing import Pool

    click.secho('Running %d processes ...' % processes, fg='yellow')
    arguments = (tuple(z) + (outlets_shapefile, workdir, overwrite) for z in zones)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(Starred, arguments)
        with click.progressbar(pooled, length=len(zones)) as progress:
            for _ in progress:
                pass

if __name__ == '__main__':
    cli()
