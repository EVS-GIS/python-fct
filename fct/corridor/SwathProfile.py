# coding: utf-8

"""
Elevation Swath Profile

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
from multiprocessing import Pool

import numpy as np
import click

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from .. import transform as fct
from ..config import config
from ..tileio import as_window
from .ransac import LinearModel, ransac
from ..cli import starcall

def TileCropInvalidRegions(axis, row, col):
    """
    DOCME
    """

    invalid_shapefile = config.filename('ax_dgo_invalid', axis=axis)
    regions_raster = config.tileset().tilename('ax_dgo', axis=axis, row=row, col=col)

    if not os.path.exists(regions_raster):
        return

    with rio.open(regions_raster) as ds:

        data = ds.read(1)
        transform = ds.transform
        nodata = ds.nodata
        profile = ds.profile.copy()

    with fiona.open(invalid_shapefile) as fs:

        def accept(feature):
            return all([
                feature['properties']['AXIS'] == axis
            ])

        mask = features.rasterize(
            [f['geometry'] for f in fs if accept(f)],
            out_shape=data.shape,
            transform=transform,
            fill=0,
            default_value=1,
            dtype='uint8')

    data[mask == 1] = nodata

    with rio.open(regions_raster, 'w', **profile) as dst:
        dst.write(data, 1)

def CropInvalidRegions(axis, processes=1, **kwargs):

    tileindex = config.tileset().tileindex

    arguments = [
        [TileCropInvalidRegions, axis, row, col, kwargs]
        for row, col in tileindex
    ]

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)
        with click.progressbar(pooled, length=len(arguments)) as iterator:
            for _ in iterator:
                pass

def UnitSwathProfile(axis, gid, bounds):
    """
    Calculate Elevation Swath Profile for Valley Unit (axis, gid)
    """

    dgo_raster = config.filename('ax_dgo', axis=axis)
    measure_raster = config.filename('ax_axis_measure', axis=axis)
    distance_raster = config.filename('ax_axis_distance', axis=axis)
    elevation_raster = config.datasource('dem1').filename
    relz_raster = config.filename('ax_relative_elevation', axis=axis)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(elevation_raster) as ds1:
            window1 = as_window(bounds, ds1.transform)
            elevations = ds1.read(1, window=window1, boundless=True, fill_value=ds1.nodata)

        with rio.open(measure_raster) as ds2:
            measure = ds2.read(1, window=window, boundless=True, fill_value=ds2.nodata)

        with rio.open(relz_raster) as ds3:
            relz = ds3.read(1, window=window, boundless=True, fill_value=ds3.nodata)

        with rio.open(dgo_raster) as ds4:
            mask = (ds4.read(1, window=window, boundless=True, fill_value=ds4.nodata) == gid)
            mask = mask & (elevations != ds1.nodata)

        assert(all([
            distance.shape == elevations.shape,
            measure.shape == elevations.shape,
            relz.shape == elevations.shape,
            mask.shape == elevations.shape
        ]))

        # Valley Bottom pixel area (to calculate areal width)

        heights = np.arange(5.0, 15.5, 0.5)
        valley_area = np.zeros(len(heights), dtype='uint32')
        
        for k, h in enumerate(heights):
            valley_area[k] = np.sum((mask == 1) & (relz <= h))

        # Profile density

        xbins = np.arange(np.min(distance[mask]), np.max(distance[mask]), 10.0)
        binned = np.digitize(distance, xbins)
        x = 0.5*(xbins[1:] + xbins[:-1])

        density = np.zeros_like(x, dtype='int32')

        for i in range(1, len(xbins)):
            
            density[i-1] = np.sum(mask & (binned == i))

        # Absolute elevation swath profile

        swath_absolute = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        for i in range(1, len(xbins)):
            
            swath_elevations = elevations[mask & (binned == i)]
            if swath_elevations.size:
                swath_absolute[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # Relative-to-stream elevation swath profile

        swath_rel_stream = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        for i in range(1, len(xbins)):

            swath_elevations = relz[mask & (relz != ds3.nodata) & (binned == i)]
            if swath_elevations.size:
                swath_rel_stream[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # Relative-to-valley-floor elevation swath profile

        swath_rel_valley = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        def fit_valley_floor(fit_mask=None, error_threshold=1.0, iterations=100):

            if fit_mask is None:
                mask0 = mask
            else:
                mask0 = mask & fit_mask

            size = elevations.shape[0]*elevations.shape[1]
            matrix = np.stack([
                measure.reshape(size),
                np.ones(size, dtype='float32'),
                elevations.reshape(size)
            ]).T
            matrix = matrix[mask0.reshape(size), :]
            samples = matrix.shape[0] // 10
            model = LinearModel([0, 1], [2])

            (slope, z0), _, _ = ransac(matrix, model, samples, iterations, error_threshold, 2*samples)

            relative = elevations - (z0 + slope*measure)

            for i in range(1, len(xbins)):

                swath_elevations = relative[mask & (binned == i)]
                if swath_elevations.size:
                    swath_rel_valley[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        try:

            fit_valley_floor()            

        except RuntimeError:

            try:

                fit_valley_floor(fit_mask=(relz <= 10.0))

            except RuntimeError:

                swath_rel_valley = np.array([])

        values = dict(
            x=x,
            varea=valley_area,
            density=density,
            sz=swath_absolute,
            hand=swath_rel_stream,
            hvf=swath_rel_valley
        )

        return axis, gid, values

def UnitSwathAxis(axis, gid, m0, bounds):
    """
    DOCME
    """

    dgo_raster = config.filename('ax_dgo', axis=axis)
    measure_raster = config.filename('ax_axis_measure', axis=axis)
    distance_raster = config.filename('ax_axis_distance', axis=axis)
    measure_weight = 0.8

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(measure_raster) as ds2:
            measure = ds2.read(1, window=window, boundless=True, fill_value=ds2.nodata)

        with rio.open(dgo_raster) as ds4:
            mask = (ds4.read(1, window=window, boundless=True, fill_value=ds4.nodata) == gid)

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

def SwathProfiles(axis, processes=1):

    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)
    relative_errors = 0
    
    def output(axis, gid):
        return config.filename('ax_swath_elevation', axis=axis, gid=gid)

    if processes == 1:

        with fiona.open(dgo_shapefile) as fs:
            with click.progressbar(fs) as iterator:
                for feature in iterator:

                    gid = feature['properties']['GID']
                    measure = feature['properties']['M']
                    geometry = asShape(feature['geometry'])
                    _, _, values = UnitSwathProfile(axis, gid, geometry.bounds)

                    if values['hvf'].size == 0:
                        relative_errors += 1

                    np.savez(
                        output(axis, gid),
                        profile=(axis, gid, measure),
                        **values)

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
                arguments.append([UnitSwathProfile, axis, gid, geometry.bounds, kwargs])

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments)

            with click.progressbar(pooled, length=len(arguments)) as iterator:

                for _, gid, values in iterator:

                    profile = profiles[axis, gid]

                    if values['hvf'].size == 0:
                        relative_errors += 1

                    np.savez(
                        output(axis, gid),
                        profile=profile,
                        **values)

    if relative_errors:
        click.secho('%d DGO without relative-to-valley-bottom profile' % relative_errors, fg='yellow')

def SwathAxes(axis, processes=1):

    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)
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

