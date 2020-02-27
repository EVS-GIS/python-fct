#!/usr/bin/env python
# coding: utf-8

"""
Reclassification des pentes en 4 classes
et continuité de la classe de pente par rapport au réseau hydrographique

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
from console import (
    success,
    info,
    important,
    warning
)

def SlopeReclass(bassin, zone, workdir, overwrite):
    """
    Reclassification des pentes en 4 classes
    """

    from qgis_helper import execute
    from processing.tools.system import getTempDirInTempFolder

    dem = os.path.join(workdir, bassin, zone, 'DEM5M.tif')
    # slope = os.path.join(workdir, bassin, zone, 'SLOPE.tif')
    output = os.path.join(workdir, bassin, zone, 'SLOPE_CLS.tif')

    if os.path.exists(output) and not overwrite:
        important('Output already exists : %s' % output)
        return

    info('Smooth DEM using 5x5 mean filter')

    parameters = dict(
        input=dem,
        bands=[1],
        filter_type=0,
        size=5,
        output=os.path.join(getTempDirInTempFolder(), 'SMOOTHED.tif')
    )

    result = execute("fct:simplerasterfilter", **parameters)

    if 'OUTPUT' not in result:
        warning('Error :(')
        return

    info('Calculate slope')

    parameters = dict(
        input=result['OUTPUT'],
        z_factor=1,
        output=os.path.join(getTempDirInTempFolder(), 'SLOPE.tif')
    )

    result = execute("qgis:slope", **parameters)

    if 'OUTPUT' not in result:
        warning('Error :(')
        return

    info('Reclass slopes')

    parameters = dict(
        input_raster=result['OUTPUT'],
        raster_band=1,
        table=[
            0, 2, 1,
            2, 6, 2,
            6, 12, 3,
            12, None, 4
        ],
        no_data=0,
        range_boundaries=1,
        nodata_for_missing=True,
        data_type=0,
        output=output
    )

    result = execute('native:reclassifybytable', **parameters)

    if 'OUTPUT' in result:
        success('Saved to %s' % result['OUTPUT'])

@click.group()
def cli():
    pass

@cli.command()
@click.argument('basin')
@click.argument('zone')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def zone(basin, zone, workdir, overwrite):

    from qgis_helper import (
        app,
        providers
    )

    SlopeReclass(basin, zone, workdir, overwrite)

@cli.command()
@click.argument('zonelist')
@click.option('--workdir', '-d', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def batch(zonelist, workdir, overwrite):

    from qgis_helper import (
        app,
        providers
    )

    with click.open_file(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    def display_item(item):
        if item:
            return item[1]
        return '...'

    with click.progressbar(zones, item_show_func=display_item) as progress:
        for basin, zone in progress:
            
            click.echo('\r')
            SlopeReclass(basin, zone, workdir, overwrite)

if __name__ == '__main__':
    cli()
