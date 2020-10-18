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
import numpy as np

import click
import rasterio as rio
import fiona

from ..cli import starcall
from ..config import config

def RemapContinuityTile(axis, tile, **kwargs):

    landcover_raster = config.tileset().tilename(
        'landcover-bdt',
        axis=axis,
        row=tile.row,
        col=tile.col,
        **kwargs)

    if 'variant' in kwargs:

        continuity_raster = config.tileset().tilename(
            'ax_continuity_variant',
            axis=axis,
            row=tile.row,
            col=tile.col,
            **kwargs)

        output = config.tileset().tilename(
            'ax_continuity_variant_remapped',
            axis=axis,
            row=tile.row,
            col=tile.col,
            **kwargs)

    else:

        continuity_raster = config.tileset().tilename(
            'ax_continuity',
            axis=axis,
            row=tile.row,
            col=tile.col,
            **kwargs)

        output = config.tileset().tilename(
            'ax_continuity_remapped',
            axis=axis,
            row=tile.row,
            col=tile.col,
            **kwargs)

    if os.path.exists(continuity_raster) and os.path.exists(landcover_raster):

        with rio.open(landcover_raster) as ds:
            landcover = ds.read(1)

        with rio.open(continuity_raster) as ds:

            profile = ds.profile.copy()

            data = ds.read(1)
            out = np.full_like(data, ds.nodata)

            out[(data == 0) | (data == 1)] = 1
            out[(data == 2) | (data == 3)] = 10
            out[(data == 4)] = 20
            out[(data == 5)] = 30
            out[(data >= 6) & (data <= 8) & (landcover >= 0) & (landcover <= 5)] = 40
            out[(data >= 6) & (data <= 8) & (landcover >= 6) & (landcover <= 8)] = 50

            # profile.update(
            #     height=height,
            #     width=width,
            #     nodata=255,
            #     dtype='uint8',
            #     transform=transform,
            #     compress='deflate'
            # )

            with rio.open(output, 'w', **profile) as dst:
                dst.write(out, 1)

def RemapContinuityRaster(axis, processes=1, **kwargs):

    # tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')

    tileset = config.tileset()

    def arguments():

        for tile in tileset.tiles():
            yield (RemapContinuityTile, axis, tile, kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass
