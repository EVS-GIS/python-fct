# coding: utf-8

"""
Arbitrary DGO Polygon Swath Profiles

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import os
from operator import itemgetter
import math

import numpy as np
import click
import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape, Point

from .. import terrain_analysis as ta
from .. import speedup
from ..rasterize import rasterize_linestring, rasterize_linestringz
from ..tileio import as_window
from ..drainage.SpatialReferencing import nearest_value_and_distance

from .ransac import LinearModel, ransac

workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def SwathProfile(axis, gid, geometry):
    """
    Calculate Elevation Swath Profile for Valley Unit (axis, gid)
    """

    refaxis_shapefile = os.path.join(workdir, 'AX%03d_REFAXIS.shp' % axis)
    network_shapefile = os.path.join(workdir, 'RHT_AXIS_TILED.shp')
    elevation_raster = '/var/local/fct/RMC/RGEALTI.tif'

    with rio.open(elevation_raster) as ds:

        # Read elevations

        click.echo('Read elevations')

        window = as_window(geometry.bounds, ds.transform)
        elevations = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)

        height, width = elevations.shape
        buf = geometry.buffer(100.0)

        # Create DGO mask

        mask = features.rasterize(
                [geometry],
                out_shape=elevations.shape,
                transform=transform,
                fill=0,
                default_value=1,
                dtype='uint8')

        # Measure and distance from reference axis

        click.echo('Measure and distance from reference axis')

        refaxis_pixels = list()

        def accept_pixel(i, j):
                x, y = ta.xy(i, j, transform)
                return all([
                    i >= -height, i < 2*height,
                    j >= -width, j < 2*width
                ]) and buf.contains(Point(x, y))

        coord = itemgetter(0, 1)

        # mmin = float('inf')
        # mmax = float('-inf')

        with fiona.open(refaxis_shapefile) as fs:
            for feature in fs:

                m0 = feature['properties']['M0']
                length = feature['properties']['LENGTH']

                # if m0 < mmin:
                #     mmin = m0

                # if m0 + length > mmax:
                #     mmax = m0 + length

                coordinates = np.array([
                    coord(p) + (m0,) for p in reversed(feature['geometry']['coordinates'])
                ], dtype='float32')

                coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                    coordinates[1:, :2] - coordinates[:-1, :2],
                    axis=1))

                coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], transform, gdal=False)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j, m in rasterize_linestringz(a, b):
                        if accept_pixel(i, j):
                            # distance[i, j] = 0
                            # measure[i, j] = m
                            refaxis_pixels.append((i, j, m))

        # mmin = math.floor(mmin / mdelta) * mdelta
        # mmax = math.ceil(mmax / mdelta) * mdelta

        mask1 = np.float32(mask)
        mask1[mask1 == 0] = ds.nodata

        measure, distance = nearest_value_and_distance(
            np.flip(np.array(refaxis_pixels), axis=0),
            mask1,
            ds.nodata)

        distance = 5.0 * distance
        distance[(elevations == ds.nodata) | (mask == 0)] = ds.nodata
        measure[(elevations == ds.nodata) | (mask == 0)] = ds.nodata

        # Relative elevations (Height above nearest drainage)

        click.echo('Relative elevations (Height above nearest drainage)')

        stream_pixels = list()

        def accept_feature(feature):

            properties = feature['properties']
            return properties['AXH'] == axis

        coord = itemgetter(0, 1, 2)
        unique = set()

        with fiona.open(network_shapefile) as fs:
            for feature in fs:

                if accept_feature(feature):

                    coordinates = np.array([
                        coord(p) for p in feature['geometry']['coordinates']
                    ], dtype='float32')

                    coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], transform, gdal=False)

                    for a, b in zip(coordinates[:-1], coordinates[1:]):
                        for i, j, z in rasterize_linestringz(a, b):
                            if accept_pixel(i, j) and (i, j) not in unique:
                                # distance[i, j] = 0
                                # measure[i, j] = m
                                # z = elevations[i, j]
                                stream_pixels.append((i, j, z))
                                unique.add((i, j))

        reference, _ = nearest_value_and_distance(
            np.array(stream_pixels),
            mask1,
            ds.nodata)

        relz = elevations - reference
        relz[(elevations == ds.nodata) | (mask == 0)] = ds.nodata

        # Swath profiles

        click.echo('Swath profiles')

        xbins = np.arange(np.min(distance[mask == 1]), np.max(distance[mask == 1]), 10.0)
        binned = np.digitize(distance, xbins)
        x = 0.5*(xbins[1:] + xbins[:-1])
        density = np.zeros_like(x, dtype='int32')

        # Profile density

        for i in range(1, len(xbins)):

            density[i-1] = np.sum((mask == 1) & (binned == i))

        # Absolute elevation swath profile

        swath_absolute = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        for i in range(1, len(xbins)):
            
            swath_elevations = elevations[(mask == 1) & (binned == i)]
            if swath_elevations.size:
                swath_absolute[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # Relative-to-stream elevation swath profile

        swath_rel_stream = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        for i in range(1, len(xbins)):

            swath_elevations = relz[(mask == 1) & (binned == i)]
            if swath_elevations.size:
                swath_rel_stream[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # Relative-to-valley-floor elevation swath profile

        click.echo('Fit valley floor')

        swath_rel_valley = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        def fit_valley_floor(fit_mask=None, error_threshold=1.0, iterations=100):

            if fit_mask is None:
                mask0 = (mask == 1)
            else:
                mask0 = (mask == 1) & fit_mask

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

                swath_elevations = relative[(mask == 1) & (binned == i)]
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
            density=density,
            swath_abs=swath_absolute,
            swath_rel=swath_rel_stream,
            swath_vb=swath_rel_valley
        )

        return axis, gid, values

def test():

    shapes = os.path.join(workdir, 'DGOEXT/AX1044_DGO.shp')

    def output(axis, gid):
        return os.path.join(workdir, 'DGOEXT', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    with fiona.open(shapes) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                properties = feature['properties']
                axis = properties['AXIS']
                gid = properties['GID']
                measure = properties['M']
                geometry = asShape(feature['geometry'])

                _, _, values = SwathProfile(axis, gid, geometry)

                np.savez(
                    output(axis, gid),
                    profile=(axis, gid, measure),
                    **values)

def PlotSwath(axis, gid, kind='absolute', output=None):

    from PlotSwath import plot_swath

    filename = os.path.join(workdir, 'DGOEXT', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    if os.path.exists(filename):

        data = np.load(filename, allow_pickle=True)

        x = data['x']
        _, _,  measure = data['profile']

        if kind == 'absolute':
            swath = data['swath_abs']
        elif kind == 'relative':
            swath = data['swath_rel']
        elif kind == 'valley bottom':
            swath = data['swath_vb']
            if swath.size == 0:
                click.secho('No relative-to-valley-bottom swath profile for DGO (%d, %d)' % (axis, gid), fg='yellow')
                click.secho('Using relative-to-nearest-drainage profile', fg='yellow')
                swath = data['swath_rel']
        else:
            click.secho('Unknown swath kind: %s' % kind)
            return

        if swath.shape[0] == x.shape[0]:
            title = 'Swath Profile #%d, PK %.1f km' % (gid, measure / 1000.0)
            if output is True:
                output = os.path.join(workdir, 'SWATH', 'AX%03d_SWATH_%04d.pdf' % (axis, gid))
            plot_swath(-x, swath, kind in ('relative', 'valley bottom'), title, output)
        else:
            click.secho('Invalid swath data')

def PlotSwathDensity(axis, gid):

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from Plotting import MapFigureSizer

    filename = os.path.join(workdir, 'DGOEXT', 'AX%03d_SWATH_%04d.npz' % (axis, gid))

    if os.path.exists(filename):

        data = np.load(filename, allow_pickle=True)

        x = data['x']
        density = data['density']
        
        bins = np.linspace(np.min(x), np.max(x), int((np.max(x) - np.min(x)) // 100.0) + 1)
        binned = np.digitize(x, bins)
        
        _x = 0.5 * (bins[1:] + bins[:-1])
        _width = bins[1:] - bins[:-1]
        _density = np.zeros_like(_x, dtype='int32')

        for i in range(1, len(bins)):
            _density[i-1] = np.sum(density[binned == i])

        fig = plt.figure(1, facecolor='white',figsize=(6.25,3.5))
        gs = plt.GridSpec(100,100,bottom=0.15,left=0.1,right=1.0,top=1.0)
        ax = fig.add_subplot(gs[25:100,10:95])

        ax.spines['top'].set_linewidth(1)
        ax.spines['left'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['bottom'].set_linewidth(1)
        ax.set_ylabel("Profile Density")
        ax.set_xlabel("Distance from reference axis (m)")
        ax.tick_params(axis='both', width=1, pad = 2)
        for tick in ax.xaxis.get_major_ticks():
            tick.set_pad(2)
        ax.grid(which='both', axis='both', alpha=0.5)

        ax.bar(_x, _density, width=_width, align='center', color='lightgray', edgecolor='k')

        fig.show()
