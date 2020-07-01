import os
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
from shapely.geometry import asShape
import fiona

import terrain_analysis as ta

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def starcall(args):
    """
    Invoke first arg function with all other arguments.
    """

    fun = args[0]
    return fun(*args[1:-1], **args[-1])

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def MkLandCoverTile(feature, bounds):

    template_raster = '/media/crousson/Backup/REFERENTIELS/IGN/RGEALTI/2017/RGEALTI.tif'
    cesbio_raster = '/media/crousson/Backup/REFERENTIELS/CESBIO/2018/OCS_2018_CESBIO.tif'
    mapping_file = '/media/crousson/Backup/REFERENTIELS/CESBIO/2018/mapping.csv'
    
    headers = None
    mapping = dict()
    minx, miny, maxx, maxy = bounds

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

    def output(x, y):

        row = int((maxy - y) // 10000)
        col = int((x - minx) // 10000)

        return os.path.join(
            workdir,
            'GLOBAL', 'LANDCOVER',
            'CESBIO_%02d_%02d.tif' % (row, col))

    with rio.open(template_raster) as template:

        resolution_x = template.transform.a
        resolution_y = template.transform.e

        with rio.open(cesbio_raster) as ds:

            profile = ds.profile.copy()

            geometry = asShape(feature['geometry'])

            properties = feature['properties']
            x0 = properties['left']
            y0 = properties['top']
            # x1 = properties['right']
            # y1 = properties['bottom']

            # i0, j0 = ds.index(x0, y0)
            # i1, j1 = ds.index(x1, y1)

            # h0 = i1 - i0
            # w0 = j1 - j0
            # window = Window(j0, i0, w0, h0)

            window = as_window(geometry.bounds, ds.transform)
            window_t = as_window(geometry.bounds, template.transform)

            it = window_t.row_off
            jt = window_t.col_off
            height = window_t.height
            width = window_t.width

            transform = template.transform * \
                template.transform.translation(jt, it)

            # transform = ds.transform * \
            #     ds.transform.translation(j0, i0) * \
            #     ds.transform.scale(w0 / width, h0 / height)

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

            with rio.open(output(x0, y0), 'w', **profile) as dst:
                dst.write(data, 1)

def MkLandCoverTiles(processes=1, **kwargs):

    tile_shapefile = os.path.join(workdir, 'TILESET', 'GRILLE_10K.shp')

    with fiona.open(tile_shapefile) as fs:
        arguments= [(MkLandCoverTile, feature, fs.bounds, kwargs) for feature in fs]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
