# coding: utf-8

"""
Spatial Referencing :
Measure along Reference Axis, Space Discretization

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
import math
from operator import itemgetter
from collections import namedtuple
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

import click
import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..config import config
# from ..tileio import ReadRasterTile
from ..rasterize import rasterize_linestring, rasterize_linestringz
from .. import transform as fct
from .. import terrain_analysis as ta
from ..cli import starcall

def nearest_value_and_distance(refpixels, domain, nodata):
    """
    Returns distance in pixels !
    """

    height, width = domain.shape
    midpoints = 0.5*(refpixels + np.roll(refpixels, 1, axis=0))
    midpoints_index = cKDTree(midpoints[1:, :2], balanced_tree=True)
    distance = np.zeros_like(domain)
    values = np.copy(distance)

    # semi-vectorized code, easier to understand

    # for i in range(height):

    #     js = np.arange(width)
    #     row = np.column_stack([np.full_like(js, i), js])
    #     valid = domain[row[:, 0], row[:, 1]] != nodata
    #     query_pixels = row[valid]
    #     nearest_dist, nearest_idx = midpoints_index.query(query_pixels, k=1, jobs=4)
    #     nearest_a = np.take(refpixels, nearest_idx, axis=0, mode='wrap')
    #     nearest_b = np.take(refpixels, nearest_idx+1, axis=0, mode='wrap')
    #     nearest_m = np.take(midpoints[:, 2], nearest_idx+1, axis=0, mode='wrap')
    #     # same as
    #     # nearest_value = 0.5*(nearest_a[:, 2] + nearest_b[:, 2])
    #     dist, signed_dist, pos = ta.signed_distance(
    #         np.float32(nearest_a),
    #         np.float32(nearest_b),
    #         np.float32(query_pixels))

    # faster fully-vectorized code

    pixi, pixj = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    valid = domain != nodata
    query_pixels = np.column_stack([pixi[valid], pixj[valid]])

    del pixi
    del pixj
    del valid

    _, nearest_idx = midpoints_index.query(query_pixels, k=1)

    # nearest_a = np.take(refpixels, nearest_idx, axis=0, mode='wrap')
    # nearest_b = np.take(refpixels, nearest_idx+1, axis=0, mode='wrap')

    nearest_p = np.take(refpixels, np.column_stack([nearest_idx, nearest_idx+1]), axis=0, mode='wrap')
    nearest_a = nearest_p[:, 0, :]
    nearest_b = nearest_p[:, 1, :]

    dist, signed_dist, pos = ta.signed_distance(
        np.float32(nearest_a),
        np.float32(nearest_b),
        np.float32(query_pixels))

    # interpolate between points A and B
    nearest_value = nearest_a[:, 2] + pos*(nearest_b[:, 2] - nearest_a[:, 2])

    # almost same as
    # nearest_m = 0.5*(nearest_a[:, 2] + nearest_b[:, 2])
    # same as
    # nearest_m = np.take(midpoints[:, 2], nearest_idx+1, axis=0, mode='wrap')

    distance[query_pixels[:, 0], query_pixels[:, 1]] = dist * np.sign(signed_dist)
    values[query_pixels[:, 0], query_pixels[:, 1]] = nearest_value

    return values, distance

SpatialReferenceParams = namedtuple('SpatialReferenceParams', [
    'ax_mask',
    'ax_reference',
    'output_distance',
    'output_measure',
    'output_units_raster',
    'output_units_shapefile',
    'mdelta'
])

def DefaultParameters():
    """
    Default parameters
    """

    return dict(
        ax_mask='ax_nearest_height',
        ax_reference='ax_refaxis',
        output_distance='ax_axis_distance',
        output_measure='ax_axis_measure',
        output_units_raster='ax_dgo',
        output_units_shapefile='ax_dgo_parts',
        mdelta=200.0
    )

def NaturalCorridorParameters():
    """
    Parameter set for natural corridor longitudinal disaggregation
    """

    return dict(
        ax_mask='ax_natural_corridor',
        ax_reference='ax_talweg',
        output_distance='ax_talweg_distance',
        output_measure='ax_talweg_measure',
        output_units_raster='ax_natural_corridor_units_raster',
        output_units_shapefile='ax_natural_corridor_units',
        mdelta=200.0
    )

def SpatialReferenceTile(axis, row, col, params, **kwargs):
    """
    see SpatialReference
    """

    tileset = config.tileset()
    # height_raster = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)

    def _tilename(dataset):
        return tileset.tilename(
            dataset,
            axis=axis,
            row=row,
            col=col)

    refaxis_shapefile = config.filename(params.ax_reference, axis=axis)
    mask_raster = _tilename(params.ax_mask)

    output_distance = _tilename(params.output_distance)
    output_measure = _tilename(params.output_measure)
    output_units_raster = _tilename(params.output_units_raster)
    output_units_shapefile = _tilename(params.output_units_shapefile)

    mdelta = params.mdelta

    with rio.open(mask_raster) as ds:

        # click.echo('Read Valley Bottom')

        # valley_bottom = speedup.raster_buffer(ds.read(1), ds.nodata, 6.0)
        mask = ds.read(1)
        height, width = mask.shape

        # distance = np.full_like(valley_bottom, ds.nodata)
        # measure = np.copy(distance)
        refaxis_pixels = list()

        # click.echo('Map Stream Network')

        def accept(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        coord = itemgetter(0, 1)

        mmin = float('inf')
        mmax = float('-inf')

        with fiona.open(refaxis_shapefile) as fs:
            for feature in fs:

                m0 = feature['properties'].get('M0', 0.0)
                length = asShape(feature['geometry']).length

                if m0 < mmin:
                    mmin = m0

                if m0 + length > mmax:
                    mmax = m0 + length

                coordinates = np.array([
                    coord(p) + (m0,) for p in reversed(feature['geometry']['coordinates'])
                ], dtype='float32')

                coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                    coordinates[1:, :2] - coordinates[:-1, :2],
                    axis=1))

                coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], ds.transform, gdal=False)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j, m in rasterize_linestringz(a, b):
                        if accept(i, j):
                            # distance[i, j] = 0
                            # measure[i, j] = m
                            refaxis_pixels.append((i, j, m))

        # ta.shortest_distance(axr, ds.nodata, startval=1, distance=distance, feedback=ta.ConsoleFeedback())
        # ta.shortest_ref(axr, ds.nodata, startval=1, fillval=0, out=measure, feedback=ta.ConsoleFeedback())

        if not refaxis_pixels:
            return []

        mmin = math.floor(mmin / mdelta) * mdelta
        mmax = math.ceil(mmax / mdelta) * mdelta

        # click.echo('Calculate Measure & Distance Raster')

        # Option 1, shortest distance

        # speedup.shortest_value(valley_bottom, measure, ds.nodata, distance, 1000.0)
        # distance = 5.0 * distance
        # distance[valley_bottom == ds.nodata] = ds.nodata
        # measure[valley_bottom == ds.nodata] = ds.nodata
        # Add 5.0 m x 10 pixels = 50.0 m buffer
        # distance2 = np.zeros_like(valley_bottom)
        # speedup.shortest_value(domain, measure, ds.nodata, distance2, 10.0)

        # Option 2, nearest using KD Tree

        measure, distance = nearest_value_and_distance(
            np.flip(np.array(refaxis_pixels), axis=0),
            np.float32(mask),
            ds.nodata)

        nodata = -99999.0
        distance = 5.0 * distance
        distance[mask == ds.nodata] = nodata
        measure[mask == ds.nodata] = nodata

        # click.echo('Write output')

        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype='float32', nodata=nodata)

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        with rio.open(output_measure, 'w', **profile) as dst:
            dst.write(measure, 1)

        # click.echo('Create DGOs')

        breaks = np.arange(mmin, mmax + mdelta, mdelta)
        dgo = np.int32(np.digitize(measure, breaks))
        dgo_measures = np.round(0.5 * (breaks + np.roll(breaks, 1)), 1)

        profile.update(nodata=0, dtype='int32')
        with rio.open(output_units_raster, 'w', **profile) as dst:
            dst.write(dgo, 1)

        # click.echo('Vectorize DGOs')

        polygons = features.shapes(
            dgo,
            dgo > 0,
            connectivity=8,
            transform=ds.transform)

        schema = {
            'geometry': 'Polygon',
            'properties': [
                ('GID', 'int'),
                ('AXIS', 'int:4'),
                ('ROW', 'int:3'),
                ('COL', 'int:3'),
                ('M', 'float:10.2')
            ]
        }
        crs = fiona.crs.from_epsg(2154)
        options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)
        count = 0
        dgos = list()

        with fiona.open(output_units_shapefile, 'w', **options) as fst:
            for polygon, gid in polygons:
                
                geom = asShape(polygon).buffer(0.0)
                measure = float(dgo_measures[int(gid)])
                feature = {
                    'geometry': geom.__geo_interface__,
                    'properties': {
                        'GID': int(gid),
                        'AXIS': axis,
                        'ROW': row,
                        'COL': col,
                        'M': measure
                    }
                }
                
                fst.write(feature)
                count += 1

                dgos.append((measure, int(gid), axis, row, col))

        # click.echo('Created %d DGOs' % count)
        # click.echo('DGO Range = %d - %d' % (1, len(breaks)))
        # click.echo('Measure Range =  %.0f - %.0f' % (np.min(breaks), np.max(breaks)))

        return dgos

def SpatialReference(axis, ax_tiles='ax_tiles', processes=1, **kwargs):
    """
    Calculate measurement support rasters and
    create discrete longitudinal units along the reference axis
    """

    parameters = DefaultParameters()
    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = SpatialReferenceParams(**parameters)

    tilefile = config.tileset().filename(ax_tiles, axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                SpatialReferenceTile,
                axis,
                row,
                col,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

def AggregateSpatialUnits(axis, ax_tiles='ax_tiles', **kwargs):
    """
    Aggregate units tiles together
    """

    parameters = DefaultParameters()
    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = SpatialReferenceParams(**parameters)

    output = config.filename(params.output_units_shapefile, axis=axis)

    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('GID', 'int'),
            ('AXIS', 'int:4'),
            ('ROW', 'int:3'),
            ('COL', 'int:3'),
            ('M', 'float:10.2')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

    tilefile = config.tileset().filename(ax_tiles, axis=axis)

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    with fiona.open(output, 'w', **options) as dst:
        with click.progressbar(tiles) as iterator:

            for row, col in iterator:

                shapefile = config.tileset().tilename(
                    params.output_units_shapefile,
                    axis=axis,
                    row=row,
                    col=col)

                if os.path.exists(shapefile):

                    with fiona.open(shapefile) as fs:

                        for feature in fs:
                            feature['properties'].update(AXIS=axis)
                            dst.write(feature)

def MapReferencePoints(axis, row, col, points, referenceset='streams-tiled'):
    """
    Project points (x, y) on reference axis.
    Only for diagnostic purpose.

    Parameters
    ----------

    axis: int

        Axis identifier

    row, col: int

        Tile coordinates

    points: list of (x, y) map coordinates

    referenceset:

      - `ax_refaxis`: Reference Axis (Measure Axis)
      - `ax_stream_network`: Theoretical stream network (axis subset)
      - `streams-tiled`: Theoretical stream network (all segments)
    """

    tileset = config.tileset()

    network_shapefile = config.filename(referenceset)
    elevation_raster = tileset.tilename('dem', row=row, col=col)
    valley_bottom_rasterfile = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)

    # output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REFPOINTS_%02d_%02d.shp' % (row, col))
    output = tileset.tilename('ax_refpoints', axis=axis, row=row, col=col)

    with rio.open(valley_bottom_rasterfile) as ds:

        click.echo('Read Valley Bottom')

        valley_bottom = ds.read(1)
        height, width = valley_bottom.shape
        refaxis_pixels = list()

        click.echo('Map Stream Network')

        def intile(i, j):
            return all([i >= 0, i < height, j >= 0, j < width])

        coord = itemgetter(0, 1)
        unique = set()

        with rio.open(elevation_raster) as ds2:
            elevations = ds2.read(1)

        with fiona.open(network_shapefile) as fs:
            for feature in fs:

                coordinates = np.array([
                    coord(p) for p in feature['geometry']['coordinates']
                ], dtype='float32')

                coordinates[:] = fct.worldtopixel(coordinates, ds.transform)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j in rasterize_linestring(a, b):
                        if intile(i, j) and (i, j) not in unique:
                            # distance[i, j] = 0
                            # measure[i, j] = m
                            z = elevations[i, j]
                            refaxis_pixels.append((i, j, z))
                            unique.add((i, j))


        refpixels = np.array(refaxis_pixels)
        midpoints = 0.5*(refpixels + np.roll(refpixels, 1, axis=0))
        midpoints_index = cKDTree(midpoints[1:, :2], balanced_tree=True)

        query_points = np.array([ds.index(x, y) for x, y in points])

        _, nearest_idx = midpoints_index.query(query_points, k=1)
        nearest_a = np.take(refpixels, nearest_idx, axis=0, mode='wrap')
        nearest_b = np.take(refpixels, nearest_idx+1, axis=0, mode='wrap')

        # nearest_m = np.take(midpoints[:, 2], nearest_idx+1, axis=0, mode='wrap')
        # same as
        # nearest_m = 0.5*(nearest_a[:, 2] + nearest_b[:, 2])

        dist, signed_dist, pos = ta.signed_distance(
            np.float32(nearest_a),
            np.float32(nearest_b),
            np.float32(query_points))

        # interpolation between points A and B
        nearest_coordinates = nearest_a + pos[:, np.newaxis] * (nearest_b - nearest_a)
        distance = 5.0 * dist * np.sign(signed_dist)

        # elevations, _ = ReadRasterTile(row, col, 'dem')

        schema = {
            'geometry': 'Point',
            'properties': [
                ('GID', 'int'),
                ('OX', 'float'),
                ('OY', 'float'),
                ('REFZ', 'float'),
                ('DZ', 'float'),
                ('DISTANCE', 'float')
            ]
        }
        crs = fiona.crs.from_epsg(2154)
        options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

        with fiona.open(output, 'w', **options) as fst:
            for k, (x, y) in enumerate(points):
                i, j = ds.index(x, y)
                refi, refj, refz = nearest_coordinates[k]
                refx, refy = ds.xy(refi, refj)
                geom = {'type': 'Point', 'coordinates': [refx, refy]}
                feature = {
                    'geometry': geom,
                    'properties': {
                        'GID': k,
                        'OX': x,
                        'OY': y,
                        'REFZ': refz,
                        'DZ': elevations[i, j] - refz,
                        'DISTANCE': float(distance[k])
                    }
                }
                fst.write(feature)
