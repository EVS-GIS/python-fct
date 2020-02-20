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

@click.command()
@click.argument('bassin')
@click.argument('zone')
@click.option('--overwrite', default=False, help='Overwrite existing output ?', is_flag=True)
def Slope(bassin, zone, overwrite):
    """
    Calcule un raster de pente en utilisant le traitement QGis `qgis:slope`.
    """

    from qgis_helper import (
        app,
        providers,
        execute
    )

    elevations = os.path.join(bassin, zone, 'DEM5M.tif')
    slope = os.path.join(bassin, zone, 'SLOPE.tif')

    if os.path.exists(slope) and not overwrite:
        click.secho('Output already exists : %s' % slope, fg='yellow')
        return

    result = execute('qgis:slope', input=elevations, output=slope)

    if 'OUTPUT' in result:
        click.secho('Saved to %s' % result['OUTPUT'], fg='green')

if __name__ == '__main__':
    Slope()
