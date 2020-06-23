import os
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
from rasterio.windows import Window
import fiona

def starcall(args):
    """
    Invoke first arg function with all other arguments.
    """

    fun = args[0]
    return fun(*args[1:-1], **args[-1])

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
            '/media/crousson/Backup/TESTS/TuilesAin/OCS',
            'CESBIO_%02d_%02d.tif' % (row, col))

    with rio.open(template_raster) as template:
        with rio.open(cesbio_raster) as ds:

            profile = ds.profile.copy()

            properties = feature['properties']
            x0 = properties['left']
            y0 = properties['top']
            x1 = properties['right']
            y1 = properties['bottom']

            i0, j0 = ds.index(x0, y0)
            i1, j1 = ds.index(x1, y1)

            h0 = i1 - i0
            w0 = j1 - j0
            window = Window(j0, i0, w0, h0)

            height, width = shape(x0, y0, x1, y1, template)
            transform = ds.transform * \
                ds.transform.translation(j0, i0) * \
                ds.transform.scale(w0 / width, h0 / height)

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

    tile_shapefile = '/media/crousson/Backup/PRODUCTION/OCSOL/GRILLE_10K_AIN.shp'

    with fiona.open(tile_shapefile) as fs:
        arguments= [(MkLandCoverTile, feature, fs.bounds, kwargs) for feature in fs]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass
