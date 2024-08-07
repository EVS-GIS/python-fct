"""
Input/Output Routines
"""

import os
import math
import subprocess
from typing import Union
from glob import glob

import numpy as np
import rasterio as rio
from rasterio.windows import Window, from_bounds
from rasterio.warp import Resampling

from .config import config
from .config.descriptors import (
    DatasetResolver,
    DatasourceResolver)
from . import transform as fct

def tileindex():
    """
    Return default tileindex
    """
    return config.tileset().tileindex

def border(height, width):
    """
    Generate a sequence of pixel (row, col)
    over the border of an array of shape (height, width)
    """

    offset = 0
    for i in (0, height-1):
        for j in range(offset, width+offset-1):
            yield i, j
        offset = 1

    offset = 1
    for j in (0, width-1):
        for i in range(offset, height+offset-1):
            yield i, j
        offset = 0

# def as_window(bounds, transform):
#     """
#     Convert real world bounds (minx, miny, maxx, maxy)
#     to raster window using defined RasterIO geo-transform 
#     """

#     minx, miny, maxx, maxy = bounds

#     row_offset, col_offset = fct.index(minx, maxy, transform)
#     row_end, col_end = fct.index(maxx, miny, transform)

#     height = row_end - row_offset
#     width = col_end - col_offset

#     return Window(col_offset, row_offset, width, height)

def as_window(bounds, transform):
    """
    Convert real world bounds (minx, miny, maxx, maxy)
    to raster window using defined RasterIO geo-transform 
    """

    return from_bounds(*bounds, transform)

def grow_window(window, padding):
    """
    Enlarge window by `padding` pixels in every direction
    (north, south, east and west)
    """

    return Window(
        window.col_off - padding,
        window.row_off - padding,
        window.width + 2*padding,
        window.height + 2*padding)

def ReadRasterTile(
        row: int,
        col: int,
        dataset1: Union[str, DatasourceResolver],
        dataset2: Union[str, DatasourceResolver] = None,
        padding: int = 0):

    if isinstance(dataset1, DatasourceResolver):
        # kwargs = dataset.arguments(kwargs)
        dataset1 = dataset1.name

    if isinstance(dataset2, DatasourceResolver):
        # kwargs = dataset.arguments(kwargs)
        dataset2 = dataset2.name

    tileset = config.tileset('default')
    tile = tileset.tileindex[row, col]

    file1 = config.datasource(dataset1).filename
    tile_height = tileset.height + 2*padding
    tile_width = tileset.width + 2*padding

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
            xres = config.datasource(dataset2).resolution
            yres = config.datasource(dataset2).resolution

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

def PadRaster(
        row: int,
        col: int,
        dataset: Union[str, DatasetResolver],
        tileset: str = 'default',
        padding: int = 1,
        **kwargs):
    """
    Assemble a n-pixels padded raster,
    with borders from neighboring tiles.
    """

    if isinstance(dataset, DatasetResolver):

        kwargs = dataset.arguments(kwargs)
        dataset = dataset.name

    tile_index = config.tileset(tileset).tileindex
    rasterfile = config.tileset(tileset).tilename(dataset, row=row, col=col, **kwargs)

    with rio.open(rasterfile) as ds:

        height, width = ds.shape
        padded = np.full((height+2*padding, width+2*padding), ds.nodata, dtype=ds.dtypes[0])

        padded[padding:-padding, padding:-padding] = ds.read(1)

        # top
        i = row-1
        j = col

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[0:padding, padding:-padding] = ds2.read(
                        1,
                        window=Window(0, ds2.height-padding, ds2.width, padding))

        else:

            padded[0:padding, padding:-padding] = ds.nodata

        # bottom
        i = row+1
        j = col

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[height+padding:, padding:-padding] = ds2.read(
                        1,
                        window=Window(0, 0, ds2.width, padding))

        else:

            padded[height+padding:, padding:-padding] = ds.nodata

        # left
        i = row
        j = col-1

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[padding:-padding, 0:padding] = ds2.read(
                        1,
                        window=Window(ds2.width-padding, 0, padding, ds2.height))

        else:

            padded[padding:-padding, 0:padding] = ds.nodata

        # right
        i = row
        j = col+1

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[padding:-padding, width+padding:] = ds2.read(
                        1,
                        window=Window(0, 0, padding, ds2.height))

        else:

            padded[padding:-padding, width+padding:] = ds.nodata

        # top-left
        i = row-1
        j = col-1

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[0:padding, 0:padding] = ds2.read(
                        1,
                        window=Window(ds2.width-padding, ds2.height-padding, padding, padding))

        else:

            padded[0:padding, 0:padding] = ds.nodata

        # bottom-left
        i = row+1
        j = col-1

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[height+padding:, 0:padding] = ds2.read(
                        1,
                        window=Window(ds2.width-padding, 0, padding, padding))

        else:

            padded[height+padding:, 0:padding] = ds.nodata

        # top-right
        i = row-1
        j = col+1

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
            if os.path.exists(other_raster):
                with rio.open(other_raster) as ds2:
                    padded[0:padding, width+padding:] = ds2.read(
                        1,
                        window=Window(0, ds2.height-padding, padding, padding))

        else:

            padded[0:padding, width+padding:] = ds.nodata

        # bottom-right
        i = row+1
        j = col+1

        if (i, j) in tile_index:

            other_raster = config.tileset(tileset).tilename(dataset, row=i, col=j, **kwargs)
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

def buildvrt(tileset: str, dataset: Union[str, DatasetResolver], suffix:bool = True, **kwargs):
    """
    Build GDAL Virtual Raster from tile dataset
    """

    if isinstance(dataset, DatasetResolver):

        vrt = dataset.filename(tileset=None, **kwargs)
        # vrt = config.filename(dataset.name, **dataset.arguments(kwargs))

    else:

        vrt = config.filename(dataset, **kwargs)

    tiledir = config.tileset(tileset).tiledir
    workdir = os.path.dirname(vrt)
    output = os.path.basename(vrt)
    prefix, extension = os.path.splitext(output)

    if suffix:
        output = os.path.join(workdir, ''.join([prefix, '_', tiledir, extension]))

    searchpath = os.path.join(workdir, tiledir, prefix)
    tiles = glob(os.path.join(searchpath, f'{prefix}_*.tif'))
    
    command = ['gdalbuildvrt', 
               '-a_srs', f'EPSG:{config.srid}', 
               '-overwrite', 
               output, 
               *tiles]
    
    subprocess.call(command)

def translate(dataset, driver='gtiff', suffix=None, **kwargs):
    """
    Translate virtual raster dataset to GeoTiff or NetCDF 4
    """

    vrt = config.tileset().filename(dataset, **kwargs)

    if not os.path.exists(vrt):
        raise ValueError('file does not exist: %s' % vrt)

    if driver in ('gtiff', 'tif'):

        # output = vrt.replace('.vrt', '.tif')
        basename, _ = os.path.splitext(vrt)

        if suffix:
            output = basename + suffix + '.tif'
        else:
            output = basename + '.tif'

        creation_options = '-co TILED=YES -co COMPRESS=DEFLATE'.split(' ')
        subprocess.run(
            ['gdal_translate', '-of', 'gtiff'] + creation_options + [vrt, output],
            check=True)

    elif driver in ('netcdf', 'nc'):

        output = vrt.replace('.vrt', '.nc')
        creation_options = '-co FORMAT=NC4 -co COMPRESS=DEFLATE -co ZLEVEL=9'.split(' ')
        subprocess.run(
            ['gdal_translate', '-of', 'netcdf'] + creation_options + [vrt, output],
            check=True)

    else:

        raise ValueError('not a supported GDAL driver: %s' % driver)
