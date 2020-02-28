#!/usr/bin/env python
# coding: utf-8

"""
Calcule le plan de drainage en utilisant RichDEM

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

import numpy as np
import rasterio as rio
import richdem as rd
import terrain_analysis as ta

SUCCESS = 'green'
INFO = 'cyan'
WARNING = 'yellow'
ERROR = 'red'

@click.command()
@click.argument('bassin')
@click.argument('zone')
@click.option('--root', type=click.Path(True, False, True), default='/media/crousson/Backup/PRODUCTION/ZONEHYDR', help='Working Directory')
@click.option('--flowdir', default='FLOWRD.tif', help='Flow Direction filename')
@click.option('--overwrite', default=False, help='Overwrite existing output ?', is_flag=True)
def RichFlow(bassin, zone, root, flowdir, overwrite):
    """
    Calcule le plan de drainage en utilisant RichDEM :
    FillDepressions + ResolveFlats
    """

    raster_template = os.path.join(root, bassin, zone, 'DEM5M.tif')
    flow_raster = os.path.join(root, bassin, zone, flowdir)

    if os.path.exists(flow_raster) and not overwrite:
        click.secho('Output already exists : %s' % flow_raster, fg=WARNING)
        return

    dem = rd.LoadGDAL(raster_template)
    filled = rd.FillDepressions(dem)
    rd.ResolveFlats(filled, True)
    flow = ta.flowdir(filled, filled.no_data)

    ds = rio.open(raster_template)
    profile = ds.profile.copy()
    profile.update(dtype=np.int16, nodata=-1, compress='deflate')

    with rio.open(flow_raster, 'w', **profile) as dst:
        dst.write(flow, 1)

    click.secho('Saved to %s' % flow_raster, fg=SUCCESS)

if __name__ == '__main__':
    RichFlow()
