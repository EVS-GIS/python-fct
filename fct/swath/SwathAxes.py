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

from operator import itemgetter
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape

# from .. import transform as fct
from ..config import config
from ..tileio import as_window
from ..cli import starcall
from ..corridor.ValleyMedialAxis import SwathMedialAxis

def unproject(axis, measures):

    refaxis_shapefile = config.filename('ax_valley_medialaxis', axis=axis)

    with fiona.open(refaxis_shapefile) as fs:

        assert len(fs) == 1

        for feature in fs:

            coordinates = np.array(
                feature['geometry']['coordinates'],
                dtype='float32')

            # reverse line direction,
            # keep only x and y coords
            coordinates = coordinates[::-1, :2]

            # calculate m coord
            m0 = feature['properties'].get('M0', 0.0)
            m = m0 + np.cumsum(np.linalg.norm(
                coordinates[1:] - coordinates[:-1],
                axis=1))

            m = np.concatenate([[0], m], axis=0)

    size = len(measures)
    points = np.zeros((size, 2), dtype='float32')
    directions = np.zeros((size, 2), dtype='float32')
    i = 0

    for k, mk in enumerate(m[:-1]):

        while i < size and mk <= measures[i] < m[k+1]:

            # p0 between vertices k and k+1

            lenk = m[k+1] - mk
            dirk = (coordinates[k+1] - coordinates[k]) / lenk
            pti = coordinates[k] + (measures[i] - mk) * dirk

            points[i] = pti
            directions[i] = dirk
            i += 1

        if i >= size:
            break

    return points, directions

def UnitSwathAxis(axis, k, gid, m0, bounds):
    """
    DOCME
    """

    dgo_raster = config.tileset().filename('ax_valley_swaths', axis=axis)
    measure_raster = config.tileset().filename('ax_axis_measure', axis=axis)
    distance_raster = config.tileset().filename('ax_axis_distance', axis=axis)
    valleybottom_raster = config.tileset().filename('ax_valley_mask_refined', axis=axis)
    # measure_weight = 0.8

    long_length = 200.0
    resolution = 5.0

    with rio.open(measure_raster) as ds:
        window = as_window(bounds, ds.transform)
        measure = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(dgo_raster) as ds:
        window = as_window(bounds, ds.transform)
        mask = (ds.read(1, window=window, boundless=True, fill_value=ds.nodata) == gid)

    with rio.open(valleybottom_raster) as ds:
        window = as_window(bounds, ds.transform)
        valleybottom = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        mask = mask & (valleybottom < 2)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        assert measure.shape == distance.shape
        assert mask.shape == distance.shape

        if np.count_nonzero(mask) == 0:
            return axis, gid, k, None, None, None, None

        # transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        # height, width = distance.shape
        dmin = np.min(distance[mask])
        dmax = np.max(distance[mask])
        # pixi, pixj = np.meshgrid(
        #     np.arange(height, dtype='int32'),
        #     np.arange(width, dtype='int32'),
        #     indexing='ij')

        # def find(d0):

        #     cost = measure_weight * np.square(measure[mask] - m0) + (1 - measure_weight) * np.square(distance[mask] - d0)
        #     idx = np.argmin(cost)
        #     i = pixi[mask].item(idx)
        #     j = pixj[mask].item(idx)
        #     return fct.xy(i, j, transform)

        # return axis, gid, find(0), find(dmin), find(dmax)

        medialpoint = SwathMedialAxis(axis, gid, m0, bounds, long_length, resolution)

        if medialpoint:
            
            return axis, gid, k, medialpoint[0], m0, dmin, dmax

        return axis, gid, k, (m0, 0.0), m0, dmin, dmax

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
            ('MEDX', 'float'),
            ('MEDY', 'float'),
        ]
    }
    options = dict(driver=driver, crs=crs, schema=schema)

    kwargs = dict()
    # profiles = dict()
    arguments = list()

    with fiona.open(dgo_shapefile) as fs:

        size = len(fs)
        measures = np.zeros(size, dtype='float32')
        valid = np.full(size, True)

        for k, feature in enumerate(fs):

            if feature['properties']['VALUE'] == 0:
                # measure = feature['properties']['M']
                # measures[k] = measure
                valid[k] = False
                continue

            gid = feature['properties']['GID']
            measure = feature['properties']['M']
            geometry = asShape(feature['geometry'])
            measures[k] = measure

            # profiles[axis, gid] = [axis, gid, measure]
            arguments.append([gid, measure, geometry.bounds, kwargs])

    points, directions = unproject(axis, np.sort(measures[valid]))
    arguments = [
        [UnitSwathAxis, axis, k] + args
        for k, args in enumerate(sorted(arguments, key=itemgetter(1)))
    ]

    with fiona.open(output, 'w', **options) as dst:
        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:

                for _, gid, k, medialpoint, coordm, dmin, dmax in iterator:

                    pt0 = points[k]
                    normalk = np.array([-directions[k][1], directions[k][0]])
                    pt_min = pt0 - dmin * normalk
                    pt_max = pt0 - dmax * normalk
                    pt_med = pt0 - medialpoint[1] * normalk

                    # profile = profiles[axis, gid]
                    # measure = profile[2]

                    dst.write({
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': [pt_min, pt_max]
                        },
                        'properties': {
                            'GID': gid,
                            'AXIS': axis,
                            'M': coordm,
                            'OX': float(pt0[0]),
                            'OY': float(pt0[1]),
                            'MEDX': float(pt_med[0]),
                            'MEDY': float(pt_med[1])
                        }
                    })
