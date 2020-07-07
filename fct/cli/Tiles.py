# coding: utf-8

"""
DOCME

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from multiprocessing import Pool
import click

import rasterio as rio
from ..config import config
from ..tileio import as_window
from ..cli import starcall

def ExtractTile(datasource, dataset, tile):

    raster = config.datasource(datasource).filename
    output = config.tileset(tile.tileset).tilename(dataset, row=tile.row, col=tile.col)

    with rio.open(raster) as ds:

        window = as_window(tile.bounds, ds.transform)
        data = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            height=window.height,
            width=window.width,
            transform=transform,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            dst.write(data, 1)

def configure(*args):
    config.default()

def DatasourceToTiles(datasource, tileset, dataset, processes=1, **kwargs):

    arguments = list()

    for tile in config.tileset(tileset).tileindex.values():
        arguments.append((ExtractTile, datasource, dataset, tile, kwargs))

    with Pool(processes=processes, initializer=configure) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
