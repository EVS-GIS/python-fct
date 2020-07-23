# coding: utf-8

"""
LandCover Tiles Extraction

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
import fiona

from ..cli import starcall
from ..config import config
from ..tileio import as_window

def MkLandCoverTile(tile):

    template_raster = config.datasource('dem1').filename
    landcover_raster = config.datasource('landcover').filename
    mapping_file = config.datasource('landcover-mapping').filename

    headers = None
    mapping = dict()

    with open(mapping_file) as fp:
        for line in fp:

            x = line.strip().split(',')

            if headers is None:
                headers = x
            else:
                mapping[int(x[1])] = int(x[2])

    def reclass(data, src_nodata, dst_nodata):

        out = np.zeros_like(data, dtype='uint8')

        for key, value in mapping.items():
            out[data == key] = value

        out[data == src_nodata] = dst_nodata

        return out

    output = config.tileset('landcover').tilename(
        'landcover',
        row=tile.row,
        col=tile.col)

    with rio.open(template_raster) as template:

        # resolution_x = template.transform.a
        # resolution_y = template.transform.e

        with rio.open(landcover_raster) as ds:

            profile = ds.profile.copy()

            window = as_window(tile.bounds, ds.transform)
            window_t = as_window(tile.bounds, template.transform)

            it = window_t.row_off
            jt = window_t.col_off
            height = window_t.height
            width = window_t.width

            transform = template.transform * \
                template.transform.translation(jt, it)

            data = ds.read(
                1,
                window=window,
                boundless=True,
                fill_value=ds.nodata,
                out_shape=(height, width))

            data = reclass(data, ds.nodata, 255)

            profile.update(
                height=height,
                width=width,
                nodata=255,
                dtype='uint8',
                transform=transform,
                compress='deflate'
            )

            with rio.open(output, 'w', **profile) as dst:
                dst.write(data, 1)

def MkLandCoverTiles(processes=1, **kwargs):

    # tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')
    tiles = config.tileset('landcover').tileindex

    arguments = [(MkLandCoverTile, tile, kwargs) for tile in tiles.values()]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def SeparateLandCoverClassesTile(
        row,
        col,
        tileset='landcover',
        dataset='landcover',
        destination='landcover-separate',
        bands=1,
        nodata=255,
        **kwargs):
    """
    Split land cover classe into separate contingency bands
    """

    rasterfile = config.tileset(tileset).tilename(
        dataset,
        row=row,
        col=col,
        **kwargs)

    output = config.tileset(tileset).tilename(
        destination,
        row=row,
        col=col,
        **kwargs)

    with rio.open(rasterfile) as ds:

        data = ds.read(1)

        profile = ds.profile.copy()
        profile.update(
            count=bands,
            dtype='uint8',
            nodata=nodata,
            compress='deflate'
        )

        with rio.open(output, 'w', **profile) as dst:
            for k in range(bands):

                band = np.uint8(data == k)
                band[data == ds.nodata] = nodata
                dst.write(band, k+1)

def SeparateLandCoverClasses(k, processes=1, tileset='landcover', **kwargs):
    """
    Split land cover classes into k separate contingency bands

    Parameters
    ----------

    k: int

        Number of landcover classes to separate,
        classes numeric count are expected to be {0, ..., k-1}

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword arguments
    -----------------

    tileset: str

        logical tileset
        defaults to `landcover`

    dataset: str

        logical name of
        landcover dataset to process

    destination: str

        logical name of destination dataset,
        defaults to `landcover-separate`

    nodata: int

        nodata value in output dataset,
        defaults to 255

    Other keywords are passed to dataset filename templates.
    """

    kwargs.update(bands=k, tileset=tileset)
    tileset = config.tileset(tileset)

    def arguments():

        for tile in tileset.tiles():
            yield (
                SeparateLandCoverClassesTile,
                tile.row,
                tile.col,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass
