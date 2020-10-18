# coding: utf-8

"""
Create Valley Bottom Mask from Height Raster

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
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
import click
import rasterio as rio

from ..tileio import as_window
from ..config import config
from ..cli import starcall

def MaskRasterTile(axis, tile):

    mask_file = '/media/crousson/Backup/Tech/FCT/BACKUP/AX0001/VALLEY_MASK.tif'
    # raster_file = config.tileset().tilename('ax_nearest_height', axis=axis, row=tile.row, col=tile.col)
    # output = config.tileset().tilename('ax_valley_mask', axis=axis, row=tile.row, col=tile.col)

    raster_file = config.tileset().tilename('ax_continuity', axis=axis, row=tile.row, col=tile.col)
    output = config.tileset().tilename('ax_continuity', axis=axis, row=tile.row, col=tile.col)

    if not os.path.exists(raster_file):
        return

    with rio.open(raster_file) as ds:

        data = ds.read(1)
        nodata = ds.nodata
        profile = ds.profile.copy()

    with rio.open(mask_file) as ds:

        window = as_window(tile.bounds, ds.transform)
        mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        data[mask == ds.nodata] = nodata

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def MaskRaster(axis, processes=1, **kwargs):

    tileset = config.tileset()

    def arguments():

        for tile in tileset.tiles():
            yield (
                MaskRasterTile,
                axis,
                tile,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass
