# coding: utf-8

"""
Extract Landover Raster within Valley Bottom

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
from multiprocessing import Pool
import click
import rasterio as rio
import fiona
from ..config import config
from ..cli import starcall
from ..tileio import buildvrt

def ValleyBottomLandcoverTile(row, col):

    tileset = config.tileset()

    mask_tile = tileset.tilename(
        # 'backup_valley_mask',
        'nearest_height',
        row=row,
        col=col
    )

    raster_tile = tileset.tilename(
        'landcover-bdt',
        row=row,
        col=col
    )

    output = tileset.tilename(
        'landcover_valley_bottom',
        row=row,
        col=col
    )

    if not (os.path.exists(raster_tile) and os.path.exists(mask_tile)):
        return

    with rio.open(raster_tile) as ds:

        data = ds.read(1)
        nodata = ds.nodata
        profile = ds.profile.copy()

    with rio.open(mask_tile) as ds:

        mask = ds.read(1)
        data[mask == ds.nodata] = nodata

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def ValleyBottomLandcover(processes=1, **kwargs):

    tileset = config.tileset()
    tilefile = tileset.filename('shortest_tiles')

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    ValleyBottomLandcoverTile,
                    row,
                    col,
                    {}
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

    buildvrt('default', 'landcover_valley_bottom')
