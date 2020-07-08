import os
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

from .. import terrain_analysis as ta
from ..cli import starcall
from ..config import config

# workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def TileLateralContinuity(axis, row, col):

    # landcover_raster = os.path.join(workdir, 'GLOBAL', 'LANDCOVER_2018.vrt')
    # distance_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'NEAREST_DISTANCE.vrt')
    # relz_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'NEAREST_RELZ.vrt')
    # output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'TILES', 'CONTINUITY_%02d_%02d.tif' % (row, col))

    tileset = config.tileset('landcover')
    landcover_raster = config.filename('landcover')
    distance_raster = config.filename('ax_nearest_distance', axis=axis)
    relz_raster = config.filename('ax_relative_elevation', axis=axis)
    output = tileset.tilename('ax_continuity', axis=axis, row=row, col=col)

    padding = 200
    height = tileset.height + 2*padding
    width = tileset.width + 2*padding
    tile_index = tileset.tileindex
    tile = tile_index[row, col]

    with rio.open(relz_raster) as ds1:

        i0, j0 = ds1.index(tile.x0, tile.y0)
        window1 = Window(j0 - padding, i0 - padding, width, height)
        relz = ds1.read(1, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(distance_raster) as ds2:

            i, j = ds2.index(tile.x0, tile.y0)
            window2 = Window(j - padding, i - padding, width, height)
            distance = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)

        with rio.open(landcover_raster) as ds3:

            profile = ds3.profile.copy()

            i, j = ds3.index(tile.x0, tile.y0)
            window3 = Window(j - padding, i - padding, width, height)
            landcover = ds3.read(1, window=window3, boundless=True, fill_value=ds3.nodata)

        landcover = np.float32(landcover) + 1
        landcover[distance == 0] = 0

        landcover[relz == ds1.nodata] = ds3.nodata
        landcover[relz > 20] = ds3.nodata
        # landcover[landcover == 7] = 1

        out = np.zeros_like(landcover)
        distance = np.zeros_like(landcover)

        cost = np.ones_like(landcover)
        cost[landcover <= 5] = 1.0
        cost[landcover >= 6] = 10.0
        cost[landcover >= 7] = 100.0

        ta.shortest_max(landcover, ds3.nodata, 0, cost, out, distance)

        out[landcover == 0] = 1
        out = np.uint8(out) - 1
        out[landcover == ds3.nodata] = ds3.nodata
        # out[landcover == 7] = 6

        height = height - 2*padding
        width = width - 2*padding
        transform = ds1.transform * ds1.transform.translation(j0, i0)
        profile.update(
            driver='GTiff',
            height=height,
            width=width,
            transform=transform,
            compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(out[padding:-padding, padding:-padding], 1)

def LateralContinuity(axis, processes=1, **kwargs):

    # tilefile = os.path.join(workdir, 'TILESET', 'TILES.shp')
    tileset = config.tileset('landcover')

    # with fiona.open(tilefile) as fs:
        
    #     arguments = list()
        
    #     for feature in fs:

    #         properties = feature['properties']
    #         row = properties['ROW']
    #         col = properties['COL']
    #         arguments.append((TileLateralContinuity, axis, row, col, kwargs))

    arguments = list()

    for tile in tileset.tileindex.values():
        arguments.append((TileLateralContinuity, axis, tile.row, tile.col, kwargs))

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
