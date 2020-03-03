import os
import numpy as np
import rasterio as rio
import terrain_analysis as ta
from Prepare import read_tile_index, tile_index

tile_shapefile = '/Volumes/Backup/PRODUCTION/RGEALTI/TILES.shp'
workdir = '/Volumes/Backup/PRODUCTION/RGEALTI/TILES'

row = 7
col = 5
read_tile_index(tile_shapefile)
gid = tile_index[(row, col)].gid
elev_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_PATCHED.tif' % (row, col))
flat_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLATS.tif' % (row, col))
ds = rio.open(elev_raster)
elev = ds.read(1)
with rio.open(flat_raster) as ds2:
    flats = ds2.read(1)

labels = ta.flat_labels(flats, ds.nodata)

height, width = labels.shape
np.count_nonzero(labels > 0) / (height*width) * 100

boxes = ta.flat_boxes(labels)

borders = set()
for w, (mini, minj, maxi, maxj, count) in boxes.items():
    if mini == 0 or minj == 0 or maxi == (height-1) or maxj == (width-1):
        borders.add(w)


mask = np.uint8(labels > 0)

profile = ds.profile.copy()
profile.update(dtype=np.uint8, nodata=255)
with rio.open(os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLATMASK.tif' % (row, col)), 'w', **profile) as dst:
    dst.write(mask, 1)

@np.vectorize
def bordermask(x):
    return x in borders

mask = np.uint8(bordermask(labels))
with rio.open(os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_BORDERFLATS.tif' % (row, col)), 'w', **profile) as dst:
    dst.write(mask, 1)

borders = set()
zones = dict()

for w, (mini, minj, maxi, maxj, count) in boxes.items():

    top = (mini == 0)
    bottom = (maxi == (height-1))
    left = (minj == 0)
    right = (maxj == (width-1))

    if not any((top, bottom, left, right)):
        continue

    borders.add(w)
    zone = (top, left, bottom, right)

    if zone in zones:
        i, j, ix, jx, c = zones[zone]
        zones[zone] = (min(i, mini), min(j, minj), max(ix, maxi), max(jx, maxj), count+c)
    else:
        zones[zone] = (mini, minj, maxi, maxj, count)