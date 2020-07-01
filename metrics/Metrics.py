import os
import math
import itertools
from operator import itemgetter
from collections import defaultdict

import numpy as np
import click

import rasterio as rio
from rasterio.windows import Window
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

import terrain_analysis as ta

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def as_window(bounds, transform):

    minx, miny, maxx, maxy = bounds

    row_offset, col_offset = ta.index(minx, maxy, transform)
    row_end, col_end = ta.index(maxx, miny, transform)

    height = row_end - row_offset
    width = col_end - col_offset

    return Window(col_offset, row_offset, width, height)

def MetricElevation(axis):

    streams_shapefile = os.path.join(workdir, 'GLOBAL', 'RHT.shp')
    dgo_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'DGO.vrt')
    elevation_raster = '/var/local/fct/RMC/RGEALTI.tif'
    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_MINZ.csv')
    
    metrics = defaultdict(lambda: float('inf'))
    # metrics = defaultdict(lambda: (float('inf'), float('-inf')))

    gid = itemgetter(0)
    elevation = itemgetter(1)

    with rio.open(elevation_raster) as ds:
        with rio.open(dgo_raster) as dgods:

            with fiona.open(streams_shapefile) as fs:
                with click.progressbar(fs) as iterator:
                    for feature in iterator:

                        xyz = np.array(feature['geometry']['coordinates'], dtype='float32')
                        elevations = np.array(list(ds.sample(xyz[:, :2], 1)))
                        dgos = np.array(list(dgods.sample(xyz[:, :2], 1)))

                        obs = sorted(np.column_stack([dgos, elevations]), key=gid)

                        for dgo, points in itertools.groupby(obs, key=gid):

                            # zmin, zmax = metrics[dgo]
                            # zs = list(elevation(p) for p in points)
                            # zmin = min(zmin, min(zs))
                            # zmax = max(zmax, max(zs))

                            zmin = min(elevation(p) for p in points)
                            metrics[dgo] = min(zmin, metrics[dgo])

    with open(output, 'w') as fp:
        for dgo in sorted(metrics.keys()):
            fp.write('%d,%d,%f,%f\n' % ((axis, dgo) + metrics[dgo]))

def MetricDrainageArea(axis):

    dgo_shapefile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REF', 'DGO.shp')
    dgo_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'DGO.vrt')
    accumulation_raster = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_DRAINAGE_AREA.csv')
    metrics = dict()

    with rio.open(accumulation_raster) as ds1:

        with fiona.open(dgo_shapefile) as fs:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    gid = feature['properties']['GID']
                    # measure = feature['properties']['M']
                    geometry = asShape(feature['geometry'])

                    window = as_window(geometry.bounds, ds1.transform)
                    accumulation = ds1.read(1, window=window, boundless=True, fill_value=ds1.nodata)

                    with rio.open(dgo_raster) as ds2:
                        window2 = as_window(geometry.bounds, ds2.transform)
                        mask = (ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata) == gid)
                        mask = mask & (accumulation != ds1.nodata)

                    assert(accumulation.shape == mask.shape)

                    metrics[gid] = np.ma.max(np.ma.masked_array(accumulation, ~mask))

    with open(output, 'w') as fp:
        for gid in sorted(metrics.keys()):
            fp.write('%d,%d,%.1f\n' % (axis, gid, metrics[gid]))

def MetricSlope(axis, distance):
    """
    DGO Slope, expressed in percent
    """

    filename = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_MINZ.csv')
    output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_SLOPE.csv')

    def gid(t):
        return int(t[1])

    def elevation(t):
        return float(t[2])
        # return 0.5 * (float(t[2]) + float(t[3]))

    with open(filename) as fp:
        data = [line.strip().split(',') for line in fp]
        elevations = np.array([elevation(t) for t in data])

    slope = np.diff(elevations, prepend=elevations[0]) * 100 / distance

    with open(output, 'w') as fp:
        for k, value in enumerate(slope):
            fp.write('%d,%d,%.3f\n' % (axis, gid(data[k]), value))


# def AggregateMetrics(axis):

#     dgo_shapefile = os.path.join(workdir, 'AX%03d_DGO.shp' % axis)
#     output = os.path.join(workdir, 'METRICS', 'AX%03d_METRICS.csv' % axis)
#     headers = ['AXIS', 'DGO', 'MEASURE']

#     metrics = {
#         'elevation': 'DGO_MINZ.csv',
#         'slope': 'DGO_SLOPE.csv',
#         'drainage': 'DGO_DRAINAGE_AREA.csv'
#     }

#     values = defaultdict(dict)

#     with fiona.open(dgo_shapefile) as fs:
#         for feature in fs:

#             gid = feature['properties']['GID']
#             measure = feature['properties']['M']
#             values[gid]['MEASURE'] = '%.0f' % measure

#     for metric, dataset in metrics.items():

#         filename = os.path.join(workdir, 'METRICS', 'AX%03d_%s' % (axis, dataset))

#         if os.path.exists(filename):

#             headers.append(metric)

#             with open(filename) as fp:
#                 for line in fp:
#                     t = line.strip().split(',')
#                     gid = int(t[1])
#                     value = t[2]
#                     values[gid][metric] = value

#     def format_values(axis, gid, values):

#         str_values = [
#             '%d' % axis,
#             '%d' % gid
#         ]

#         for name in headers[2:]:
#             if name in values:
#                 str_values.append(values[name])
#             else:
#                 str_values.append('nan')

#         return ','.join(str_values)


#     with open(output, 'w') as fp:

#         fp.write(','.join(headers))
#         fp.write('\n')

#         for gid in sorted(values.keys()):

#             if gid == 0:
#                 continue

#             fp.write(format_values(axis, gid, values[gid]))
#             fp.write('\n')
