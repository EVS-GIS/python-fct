# coding: utf-8

"""
DEM Burning
Match mapped stream network and DEM by adjusting stream's elevation

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
import numpy as np
import click

import rasterio as rio
import fiona
from shapely.geometry import (
    asShape,
    box
)

from fct  import terrain_analysis as ta
from fct.config import config
from fct.rasterize import rasterize_linestringz

# def DispatchHydrographyToTiles():

#     src = '/var/local/fct/RMC/TILES2/HYDROGRAPHY_TILED.shp'
#     tileindex = config.tileset().tileindex

#     def rowcol(feature):
#         return feature['properties']['ROW'], feature['properties']['COL']

#     with fiona.open(src) as fs:
#         options = dict(driver=fs.driver, crs=fs.crs, schema=fs.schema)
#         features = sorted(list(fs), key=rowcol)

#     groups = itertools.groupby(features, key=rowcol)

#     with click.progressbar(groups, length=len(tileindex)) as progress:
#         for (row, col), features in progress:
#             with fiona.open(config.tileset().filename('hydrography', row=row, col=col), 'w', **options) as fst:
#                 for feature in features:
#                     fst.write(feature)

def BurnTile(params, row, col, elevations=None, tileset='default'):
    """
    DOCME
    """

    elevation_raster = params.elevations.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename(dataset, row=row, col=col)
    hydrography = params.hydrography.tilename(row=row, col=col, tileset=tileset)
    # config.tileset().tilename('stream-network-draped', row=row, col=col)

    with rio.open(elevation_raster) as ds:

        if elevations is None:
            elevations = ds.read(1)

        height, width = elevations.shape

        if os.path.exists(hydrography):

            with fiona.open(hydrography) as fs:
                for feature in fs:

                    geom = np.array(feature['geometry']['coordinates'], dtype=np.float32)
                    geom[:, :2] = np.fliplr(ta.worldtopixel(geom, ds.transform, gdal=False))

                    for a, b in zip(geom[:-1], geom[1:]):
                        for px, py, z in rasterize_linestringz(a, b):
                            if all([py >= 0, py < height, px >= 0, px < width, not np.isinf(z)]):
                                elevations[py, px] = z - params.offset
        else:

            click.secho('File not found: %s' % hydrography, fg='yellow')

    return elevations
