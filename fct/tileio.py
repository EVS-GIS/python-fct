# coding: utf-8

"""
Input/Output Routines

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
import math
import numpy as np
import rasterio as rio
from rasterio.windows import Window
from rasterio.warp import Resampling
from .config import config
from . import transform as fct

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset('drainage').tileindex

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = fct.index(minx, maxy, transform)
    row_end, col_end = fct.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def ReadRasterTile(row, col, dataset1, dataset2=None, padding=0):

    tile_index = tileindex()
    tile = tile_index[row, col]

    tileset = config.tileset('drainage')
    file1 = config.datasource(dataset1).filename
    tile_height = tileset.height + 2*padding
    tile_width = tileset.width + 2*padding
    xres = tileset.resolution
    yres = tileset.resolution

    with rio.open(file1) as ds:

        row_offset, col_offset = ds.index(tile.x0, tile.y0)

        window1 = Window(col_offset - padding, row_offset - padding, tile_width, tile_height)
        data = ds.read(1, window=window1, boundless=True, fill_value=ds.nodata)
        transform = ds.transform * ds.transform.translation(
            col_offset - padding,
            row_offset - padding)
        profile = ds.profile.copy()
        profile.update(
            transform=transform,
            height=tile_height,
            width=tile_width)

        if dataset2:

            file2 = config.datasource(dataset2).filename

            with rio.open(file2) as ds2:

                i2, j2 = ds2.index(*ds.xy(window1.row_off, window1.col_off))
                window2 = Window(j2, i2, tile_width//xres, tile_height//yres)

                data2 = ds2.read(
                    1,
                    window=window2, boundless=True, fill_value=ds2.nodata,
                    resampling=Resampling.bilinear,
                    out_shape=data.shape)

                mask = (data == ds.nodata) & (data2 != ds2.nodata)
                data[mask] = data2[mask]

    return data, profile

def DownsampleRasterTile(row, col, dataset1, dataset2=None, factor=2):

    tile_index = tileindex()
    tile = tile_index[row, col]

    file1 = config.datasource(dataset1).filename
    # tile_height = int(parameter('input.height'))
    # tile_width = int(parameter('input.width'))
    # xres = float(parameter('input.xres'))
    # yres = float(parameter('input.yres'))

    tile_height = tile_width = 800
    xres = yres = 49.950637774860638

    height = math.ceil(tile_height / factor)
    width = math.ceil(tile_width / factor)

    with rio.open(file1) as ds:

        row_offset, col_offset = ds.index(tile.x0, tile.y0)
        window1 = Window(col_offset, row_offset, tile_width, tile_height)

        data = ds.read(
            1, window=window1,
            boundless=True, fill_value=ds.nodata,
            out_shape=(height, width))

        transform = ds.transform * ds.transform.translation(col_offset, row_offset) * \
            ds.transform.scale(
                tile_width / width,
                tile_height / height)

        profile = ds.profile.copy()
        profile.update(
            transform=transform,
            height=height,
            width=width)

        if dataset2:

            file2 = config.datasource(dataset2).filename

            with rio.open(file2) as ds2:

                i2, j2 = ds2.index(*ds.xy(window1.row_off, window1.col_off))
                window2 = Window(j2, i2, tile_width//xres, tile_height//yres)

                data2 = ds2.read(
                    1,
                    window=window2, boundless=True, fill_value=ds2.nodata,
                    resampling=Resampling.bilinear,
                    out_shape=data.shape)

                mask = (data == ds.nodata) & (data2 != ds2.nodata)
                data[mask] = data2[mask]

    return data, profile

def PadRaster(row, col, dataset='filled', padding=1, **kwargs):
    """
    Assemble a n-pixels padded raster,
    with borders from neighboring tiles.
    """

    tile_index = tileindex()
    rasterfile = config.datasource(dataset).filename(row=row, col=col, **kwargs)

    with rio.open(rasterfile) as ds:

        height, width = ds.shape
        padded = np.zeros((height+2*padding, width+2*padding), dtype=ds.dtypes[0])

        padded[padding:-padding, padding:-padding] = ds.read(1)

        # top
        i = row-1
        j = col

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[0:padding, padding:-padding] = ds2.read(
                        1,
                        window=Window(0, height-padding, width, padding))

        else:

            padded[0:padding, padding:-padding] = ds.nodata

        # bottom
        i = row+1
        j = col

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[height+padding:, padding:-padding] = ds2.read(
                        1,
                        window=Window(0, 0, width, padding))

        else:

            padded[height+padding:, padding:-padding] = ds.nodata

        # left
        i = row
        j = col-1

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[padding:-padding, 0:padding] = ds2.read(
                        1,
                        window=Window(width-padding, 0, padding, height))

        else:

            padded[padding:-padding, 0:padding] = ds.nodata

        # right
        i = row
        j = col+1

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[padding:-padding, width+padding:] = ds2.read(
                        1,
                        window=Window(0, 0, padding, height))

        else:

            padded[padding:-padding, width+padding:] = ds.nodata

        # top-left
        i = row-1
        j = col-1

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[0:padding, 0:padding] = ds2.read(
                        1,
                        window=Window(width-padding, height-padding, padding, padding))

        else:

            padded[0:padding, 0:padding] = ds.nodata

        # bottom-left
        i = row+1
        j = col-1

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[height+padding:, 0:padding] = ds2.read(
                        1,
                        window=Window(width-padding, 0, padding, padding))

        else:

            padded[height+padding:, 0:padding] = ds.nodata

        # top-right
        i = row-1
        j = col+1

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[0:padding, width+padding:] = ds2.read(
                        1,
                        window=Window(0, height-padding, padding, padding))

        else:

            padded[0:padding, width+padding:] = ds.nodata

        # bottom-right
        i = row+1
        j = col+1

        if (i, j) in tile_index:

            other_raster = config.datasource(dataset).filename(row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[height+padding:, width+padding:] = ds2.read(
                        1,
                        window=Window(0, 0, padding, padding))

        else:

            padded[height+padding:, width+padding:] = ds.nodata

        transform = ds.transform * ds.transform.translation(-padding, -padding)
        profile = ds.profile.copy()
        profile.update(
            transform=transform,
            height=height+2*padding,
            width=width+2*padding)

    return padded, profile
