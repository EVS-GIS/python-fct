# coding: utf-8

"""
Valley Bottom Delineation based on Valley Height Raster

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

import numpy as np

import click
import rasterio as rio
from rasterio.features import sieve
import fiona
import xarray as xr

from .. import speedup
from ..tileio import as_window
from ..config import config
from ..cli import starcall

def ValleyHeightMaskTile(axis, row, col):
    """
    Extract bottom mask from valley height raster
    """

    tileset = config.tileset()

    def _tilename(name):
        return tileset.tilename(name, axis=axis, row=row, col=col)

    height_raster = _tilename('ax_valley_height')
    continuity_raster = _tilename('ax_continuity')
    output = _tilename('ax_valley_height_mask')

    if not os.path.exists(height_raster):
        # click.echo('missing tile (%d, %d)' % (row, col))
        return

    with rio.open(continuity_raster) as ds:
        continuity = ds.read(1)

    with rio.open(height_raster) as ds:

        valley_height = ds.read(1)

        nodata = 255
        out = np.full_like(valley_height, nodata, dtype='uint8')

        valley_mask = (valley_height != ds.nodata)
        bottom_mask = (valley_height >= -10.0) & (valley_height <= 4.0)
        out[valley_mask] = 2
        out[valley_mask & bottom_mask] = 0
        out[continuity < 2] = 0

        sieve(out, 400)

        profile = ds.profile.copy()
        profile.update(
            dtype='uint8',
            nodata=nodata,
            compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def ValleyHeightMask(axis, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):
    """
    Refine valley bottom mask
    0: bottom
    1: margin/hole => separate with exterior region algorithm
    """

    tilefile = config.tileset().filename(ax_tiles, axis=axis, **kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:
                row, col = tuple(int(x) for x in line.split(','))
                yield (
                    ValleyHeightMaskTile,
                    axis,
                    row,
                    col,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass
