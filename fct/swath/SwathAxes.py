# coding: utf-8

"""
Swath Axes

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape

from .. import transform as fct
from ..config import config
from ..tileio import as_window
from ..cli import starcall

def UnitSwathAxis(axis, gid, m0, bounds):
    """
    DOCME
    """

    dgo_raster = config.tileset().filename('ax_valley_swaths', axis=axis)
    measure_raster = config.tileset().filename('ax_axis_measure', axis=axis)
    distance_raster = config.tileset().filename('ax_axis_distance', axis=axis)
    measure_weight = 0.8

    with rio.open(measure_raster) as ds:
        window = as_window(bounds, ds.transform)
        measure = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(dgo_raster) as ds:
        window = as_window(bounds, ds.transform)
        mask = (ds.read(1, window=window, boundless=True, fill_value=ds.nodata) == gid)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        assert(all([
            measure.shape == distance.shape,
            mask.shape == distance.shape
        ]))

        if np.count_nonzero(mask) == 0:
            return axis, gid, None, None, None

        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        height, width = distance.shape
        dmin = np.min(distance[mask])
        dmax = np.max(distance[mask])
        pixi, pixj = np.meshgrid(
            np.arange(height, dtype='int32'),
            np.arange(width, dtype='int32'),
            indexing='ij')

        def find(d0):

            cost = measure_weight * np.square(measure[mask] - m0) + (1 - measure_weight) * np.square(distance[mask] - d0)
            idx = np.argmin(cost)
            i = pixi[mask].item(idx)
            j = pixj[mask].item(idx)
            return fct.xy(i, j, transform)

        return axis, gid, find(0), find(dmin), find(dmax)

def SwathAxes(axis, processes=1):

    dgo_shapefile = config.filename('ax_valley_swaths_polygons', axis=axis)
    output = config.filename('ax_swath_axes', axis=axis)

    driver = 'ESRI Shapefile'
    crs = fiona.crs.from_epsg(2154)
    schema = {
        'geometry': 'LineString',
        'properties': [
            ('GID', 'int:4'),
            ('AXIS', 'int:4'),
            ('M', 'float:10.2'),
            ('OX', 'float'),
            ('OY', 'float'),
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    if processes == 1:

        with fiona.open(output, 'w', **options) as dst:
            with fiona.open(dgo_shapefile) as fs:
                with click.progressbar(fs) as iterator:
                    for feature in iterator:

                        gid = feature['properties']['GID']
                        measure = feature['properties']['M']
                        geometry = asShape(feature['geometry'])
                        _, _, pt0, pt_min, pt_max = UnitSwathAxis(axis, gid, measure, geometry.bounds)

                        if pt0 is None:
                            continue

                        dst.write({
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': [pt_min, pt0, pt_max]
                            },
                            'properties': {
                                'GID': gid,
                                'AXIS': axis,
                                'M': measure,
                                'OX': float(pt0[0]),
                                'OY': float(pt0[1])
                            }
                        })

    else:

        kwargs = dict()
        profiles = dict()
        arguments = list()

        with fiona.open(dgo_shapefile) as fs:
            for feature in fs:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                geometry = asShape(feature['geometry'])

                profiles[axis, gid] = [axis, gid, measure]
                arguments.append([UnitSwathAxis, axis, gid, measure, geometry.bounds, kwargs])

        with fiona.open(output, 'w', **options) as dst:
            with Pool(processes=processes) as pool:

                pooled = pool.imap_unordered(starcall, arguments)

                with click.progressbar(pooled, length=len(arguments)) as iterator:

                    for _, gid, pt0, pt_min, pt_max in iterator:

                        if pt0 is None:
                            continue

                        profile = profiles[axis, gid]
                        measure = profile[2]

                        dst.write({
                            'geometry': {
                                'type': 'LineString',
                                'coordinates': [pt_min, pt0, pt_max]
                            },
                            'properties': {
                                'GID': gid,
                                'AXIS': axis,
                                'M': measure,
                                'OX': float(pt0[0]),
                                'OY': float(pt0[1])
                            }
                        })
