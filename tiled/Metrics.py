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

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def MetricElevation(axis):

    streams_shapefile = os.path.join(workdir, 'RHT.shp')
    dgo_raster = os.path.join(workdir, 'AX%03d_DGO.vrt' % axis)
    elevation_raster = '/var/local/fct/RMC/RGEALTI.tif'
    output = os.path.join(workdir, 'METRICS', 'AX%03d_DGO_MINZ.csv' % axis)
    metrics = defaultdict(lambda: float('inf'))

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

                            zmin = min(elevation(p) for p in points)
                            metrics[dgo] = min(zmin, metrics[dgo])

    with open(output, 'w') as fp:
        for dgo in sorted(metrics.keys()):
            fp.write('%d,%d,%f\n' % (axis, dgo, metrics[dgo]))
