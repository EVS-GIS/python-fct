# coding: utf-8

"""
LandCover Tiles Extraction
TODO move outside metrics => data preparation

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

from ..cli import starcall
from ..config import config

def CropLandcoverTile(axis, tile, **kwargs):

    height_raster = config.tileset().tilename(
        'ax_valley_mask',
        axis=axis,
        row=tile.row,
        col=tile.col,
        **kwargs)

    landcover_raster = config.tileset().tilename(
        'landcover-bdt',
        axis=axis,
        row=tile.row,
        col=tile.col,
        **kwargs)

    output = config.tileset().tilename(
        'ax_landcover',
        axis=axis,
        row=tile.row,
        col=tile.col,
        **kwargs)

    if os.path.exists(height_raster) and os.path.exists(landcover_raster):

        with rio.open(height_raster) as ds:

            hand = ds.read(1)
            hand_nodata = ds.nodata

        with rio.open(landcover_raster) as ds:

            profile = ds.profile.copy()

            landcover = ds.read(1)
            landcover[hand == hand_nodata] = ds.nodata

            # profile.update(
            #     height=height,
            #     width=width,
            #     nodata=255,
            #     dtype='uint8',
            #     transform=transform,
            #     compress='deflate'
            # )

            with rio.open(output, 'w', **profile) as dst:
                dst.write(landcover, 1)

def CropLandcover(axis, processes=1, **kwargs):

    # tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')

    tileset = config.tileset()

    def arguments():

        for tile in tileset.tiles():
            yield (CropLandcoverTile, axis, tile, kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass
