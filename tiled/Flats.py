import os
import numpy as np
import rasterio as rio
import terrain_analysis as ta
import speedup
from Prepare import read_tile_index, tile_index

# tile_shapefile = '/Volumes/Backup/PRODUCTION/RGEALTI/TILES.shp'
# workdir = '/Volumes/Backup/PRODUCTION/RGEALTI/TILES'

tile_shapefile = '/media/crousson/Backup/PRODUCTION/RGEALTI/TILES.shp'
workdir = '/media/crousson/Backup/PRODUCTION/RGEALTI/TILES'
read_tile_index(tile_shapefile)

row = 8
col = 5

gid = tile_index[(row, col)].gid

elevation_raster = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED2.tif' % (row, col))
ds = rio.open(elevation_raster)

def filename(row, col):
    return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED2.tif' % (row, col))

padded = PadElevations(row, col, filename)

flow = ta.flowdir(padded, ds.nodata)
labels, outlets = speedup.flat_labels(flow, padded, ds.nodata)
notflowing = {k+1 for k, (i, j) in enumerate(outlets) if i == -1 and j == -1}

# @np.vectorize
# def noflow_mask(x):
#     return x in notflowing

# mask = np.uint8(noflow_mask(labels))
# mask[elev == ds.nodata] = 255

# profile = ds.profile.copy()
# profile.update(dtype=np.uint8, nodata=255)
# with rio.open(os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_NOFLOW.tif' % (row, col)), 'w', **profile) as dst:
#     dst.write(mask, 1)

# profile.update(dtype=np.uint32, nodata=0)
# with rio.open(os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_PITLABELS.tif' % (row, col)), 'w', **profile) as dst:
#     dst.write(labels, 1)

# labels, graph, pit_regions = speedup.watershed_labels(elev, ds.nodata)
# labels, outlets = speedup.flat_labels(elev, flats, ds2.nodata)
# len([(i, j) for i, j in outlets if i != -1 and j != -1]) 

# height, width = labels.shape
# np.count_nonzero(labels > 0) / (height*width) * 100

# mask = np.uint8(labels > 0)
# mask[elev == ds.nodata] = 255

# profile = ds.profile.copy()
# profile.update(dtype=np.uint8, nodata=255)
# with rio.open(os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLATMASK.tif' % (row, col)), 'w', **profile) as dst:
#     dst.write(mask, 1)

height, width = labels.shape
boxes = speedup.flat_boxes(labels)

borders = set()
for w, (mini, minj, maxi, maxj, count) in boxes.items():
    if mini == 0 or minj == 0 or maxi == (height-1) or maxj == (width-1):
        if w not in notflowing:
            borders.add(w)

# @np.vectorize
# def bordermask(x):
#     return x in borders

# mask = np.uint8(bordermask(labels))
# with rio.open(os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_BORDERFLATS.tif' % (row, col)), 'w', **profile) as dst:
#     dst.write(mask, 1)

# borders = set()
# zones = dict()

# for w, (mini, minj, maxi, maxj, count) in boxes.items():

#     top = (mini == 0)
#     bottom = (maxi == (height-1))
#     left = (minj == 0)
#     right = (maxj == (width-1))

#     if not any((top, bottom, left, right)):
#         continue

#     borders.add(w)
#     zone = (top, left, bottom, right)

#     if zone in zones:
#         i, j, ix, jx, c = zones[zone]
#         zones[zone] = (min(i, mini), min(j, minj), max(ix, maxi), max(jx, maxj), count+c)
#     else:
#         zones[zone] = (mini, minj, maxi, maxj, count)

def FlowDirection(row, col):

    output = os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FLOW.tif' % (row, col))

    def filename(row, col):
        return os.path.join(workdir, 'RGE5M_TILE_%02d_%02d_FILLED2.tif' % (row, col))

    with rio.open(filename(row, col)) as ds:

        padded = PadElevations(row, col, filename)

        flow = ta.flowdir(padded, ds.nodata)
        labels, outlets = speedup.flat_labels(flow, padded, ds.nodata)
        notflowing = {k+1 for k, (i, j) in enumerate(outlets) if i == -1 and j == -1}

        height, width = labels.shape
        boxes = speedup.flat_boxes(labels)

        borders = set()
        for w, (mini, minj, maxi, maxj, count) in boxes.items():
            if mini == 0 or minj == 0 or maxi == (height-1) or maxj == (width-1):
                if w not in notflowing:
                    borders.add(w)

        @np.vectorize
        def bordermask(x):
            return x in borders

        mask = bordermask(labels)
        mask[1:-1, 1:-1] = False
        padded = padded + np.max(padded)*mask

        flat_mask, flat_labels = ta.resolve_flat(padded, flow, ta.ConsoleFeedback())
        # extended = rd.rdarray(flat_mask, no_data=0)
        # rd.FillDepressions(extended, True, True, 'D8')
        ta.flat_mask_flowdir(flat_mask, flow, flat_labels)
        # extended = rd.rdarray(padded, no_data=ds.nodata)
        # rd.BreachDepressions(extended, True, 'D8')
        # rd.ResolveFlats(extended, True)
        # rd.FillDepressions(extended, True, True, 'D8')

        # flow = ta.flowdir(padded, ds.nodata)

        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype=np.int16, nodata=-1)

        with rio.open(output, 'w', **profile) as dst:
            dst.write(flow[1:-1, 1:-1], 1)