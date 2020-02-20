#!/usr/bin/env python
# coding: utf-8

"""
Calcule un raster de pente

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import click
import os
from tqdm import tqdm

@click.command()
@click.argument('zonelist')
@click.option('--overwrite', default=False, help='Overwrite existing output ?', is_flag=True)
def SlopeBatch(zonelist, overwrite):
    """
    Calcule un raster de pente en utilisant le traitement QGis `qgis:slope`
    pour chaque (bassin, zone) dans la liste en entrée.
    """

    from qgis_helper import (
        app,
        providers,
        execute
    )

    with open(zonelist) as fp:
        zones = [info.strip().split(' ') for info in fp]

    with tqdm(zones) as progress:
        for bassin, zone in progress:

            elevations = os.path.join(bassin, zone, 'DEM5M.tif')
            slope = os.path.join(bassin, zone, 'SLOPE.tif')

            if os.path.exists(slope) and not overwrite:
                # click.secho('Output already exists : %s' % slope, fg='yellow')
                progress.set_description('Output already exists : %s' % slope)
                continue

            result = execute('qgis:slope', input=elevations, output=slope)

            if 'OUTPUT' in result:
                # click.secho('Saved to %s' % result['OUTPUT'], fg='green')
                progress.set_description('Saved to %s' % result['OUTPUT'])

if __name__ == '__main__':
    SlopeBatch()
