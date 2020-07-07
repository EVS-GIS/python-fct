import rasterio as rio
from rasterio.windows import Window
from rasterio.warp import Resampling
import fiona
import fiona.crs
import numpy as np

from ..config import config

tileset = config.tileset('drainage')

tile_height = tileset.height
tile_width = tileset.width
srs = config.srid

crs = fiona.crs.from_epsg(srs)
schema = {
    'geometry': 'Polygon',
    'properties': [
        ('GID', 'int'),
        ('ROW', 'int'),
        ('COL', 'int'),
        ('X0', 'float'),
        ('Y0', 'float')
    ]}
driver = 'ESRI Shapefile'
options = dict(crs=crs, driver=driver, schema=schema)

BDA = '/media/crousson/Backup/REFERENTIELS/IGN/BDALTI_25M/BDALTI25M.tif'
RGE = '/media/crousson/Backup/REFERENTIELS/IGN/RGEALTI/2017/RGEALTI.tif'

def asPolygon(window, ds):
    x0, y0 = ds.xy(window.row_off, window.col_off)
    x1, y1 = ds.xy(window.row_off + window.height, window.col_off + window.width)
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]

with rio.open(RGE) as ds:

    current = 0

    height, width = ds.shape
    # tile_height = height // 20
    # tile_width = width // 10

    with fiona.open('/media/crousson/Backup/PRODUCTION/RGEALTI/RATILES.shp', 'w', **options) as dst:
        for i in range(0, height, tile_height):
            for j in range(0, width, tile_width):
                w = Window(j, i, tile_width, tile_height)
                # dem = ds.read(1, window=w)
                # if np.count_nonzero(dem == ds.nodata) == tile_width*tile_height:
                #     continue
                box = asPolygon(w, ds)
                geom = {'type': 'Polygon', 'coordinates': [box]}
                props = {'GID': current, 'ROW': i // tile_height, 'COL': j // tile_width, 'X0': box[0][0], 'Y0': box[0][1]}
                dst.write({'geometry': geom, 'properties': props})
                current += 1

# ds2 = rio.open(BDA)
# i2, j2 = ds2.index(*ds.xy(w.row_off, w.col_off))
# w2 = Window(j2, i2, width//10//5, height//10//5)
# dem2 = ds2.read(1, resampling=Resampling.bilinear, out_shape=dem.shape, boundless=True, fill_value=ds2.nodata, window=w2)
# dem2.shape == dem.shape
