import os
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import asShape
import fiona

from .. import terrain_analysis as ta
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

        for k, v in mapping.items():
            out[data == k] = v

        out[data == src_nodata] = dst_nodata

        return out

    def shape(x0, y0, x1, y1, ds):

        i0, j0 = ds.index(x0, y0)
        i1, j1 = ds.index(x1, y1)
        height = i1 - i0
        width = j1 - j0

        return height, width

    def output(tile):

        return config.tileset('landcover').tilename(
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

            with rio.open(output(tile), 'w', **profile) as dst:
                dst.write(data, 1)

def configure(*args):
    config.default()

def MkLandCoverTiles(processes=1, **kwargs):

    # tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')
    tiles = config.tileset('landcover').tileindex

    arguments = [(MkLandCoverTile, tile, kwargs) for tile in tiles.values()]

    with Pool(processes=processes, initializer=configure) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
