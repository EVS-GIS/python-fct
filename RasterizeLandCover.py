#!/usr/bin/env python
# coding: utf-8

"""
Rastérization de la carte d'occupation du sol
à la résolution du MNT

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
from rasterio.features import rasterize
import fiona

SUCCESS = 'green'
INFO = 'cyan'
WARNING = 'yellow'
ERROR = 'red'

@click.command()
@click.argument('bassin')
@click.argument('zone')
@click.option('--root', type=click.Path(True, False, True, resolve_path=True), default='.', help='Working Directory')
@click.option('--overwrite', '-w', default=False, help='Overwrite existing output ?', is_flag=True)
def RasterizeLandCover(bassin, zone, root, overwrite):
    """
    Rastérization de la carte d'occupation du sol
    à la résolution du MNT
    """

    click.secho('Processing zone %s' % zone, fg=SUCCESS)
    click.secho('Working Directory = %s' % root, fg=INFO)

    shapefile = os.path.join(root, bassin, zone, 'LandCover.shp')
    template_raster = os.path.join(root, bassin, zone, 'DEM5M.tif')
    output = os.path.join(root, bassin, zone, 'LANDCOVER5M.tif')

    if os.path.exists(output) and not overwrite:
        click.secho('Output already exists : %s' % output, fg=WARNING)
        return

    with rio.open(template_raster) as ds:
        
        out = np.zeros((ds.height, ds.width), dtype=np.float32)

        def shapes():
            """
            Generator of (geometry, value) pairs
            """
            with fiona.open(shapefile) as fs:
                for feature in fs:
                    value = feature['properties']['CODE']
                    geom = feature['geometry']
                    yield geom, value

        nodata = -99999
        rasterize(shapes(), out=out, transform=ds.transform, fill=nodata)

        profile = ds.profile.copy()
        profile.update(nodata=nodata, compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

if __name__ == '__main__':
    RasterizeLandCover()
