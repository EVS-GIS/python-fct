# coding: utf-8

"""
Spatial Referencing :
Measure along Reference Axis, Space Discretization

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import math
import click
import numpy as np
from operator import itemgetter

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape
from scipy.spatial import cKDTree

from config import tileindex, filename, parameter
from tileio import ReadRasterTile
from rasterize import rasterize_linestring, rasterize_linestringz
import terrain_analysis as ta
import speedup

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

def test():

    row = 5
    col = 7
    valley_bottom_rasterfile = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_FLOW_05_07.tif'
    refaxis_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_AXREF_05_07.shp'
    output_distance = '/media/crousson/Backup/WORK/TestAin/AIN_AXIS_DISTANCE_05_07.tif'
    output_measure = '/media/crousson/Backup/WORK/TestAin/AIN_AXIS_MEASURE_05_07.tif'
    output_dgo = '/media/crousson/Backup/WORK/TestAin/AIN_DGO_05_07.tif'
    output_dgo_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_DGO_05_07.shp'

    with rio.open(valley_bottom_rasterfile) as ds:

        click.echo('Read Valley Bottom')

        valley_bottom = speedup.raster_buffer(ds.read(1), ds.nodata, 6.0)
        height, width = valley_bottom.shape

        # distance = np.full_like(valley_bottom, ds.nodata)
        # measure = np.copy(distance)
        refaxis_pixels = list()

        click.echo('Map Stream Network')

        def intile(i, j):
            return all([i >= 0, i < height, j >= 0, j < width])

        coord = itemgetter(0, 1)

        with fiona.open(refaxis_shapefile) as fs:
            for feature in fs:

                m0 = feature['properties']['M0']

                coordinates = np.array([
                    coord(p) + (m0,) for p in reversed(feature['geometry']['coordinates'])
                ], dtype='float32')

                coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                    coordinates[1:, :2] - coordinates[:-1, :2],
                    axis=1))

                coordinates[:, :2] = ta.worldtopixel(coordinates[:, :2], ds.transform, gdal=False)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j, m in rasterize_linestringz(a, b):
                        if intile(i, j):
                            # distance[i, j] = 0
                            # measure[i, j] = m
                            refaxis_pixels.append((i, j, m))

                # valid_pixels = np.array([intile(i, j) for i, j in pixels])
                # distance[pixels[valid_pixels, 0], pixels[valid_pixels, 1]] = 0.0
                # measure[pixels[valid_pixels, 0], pixels[valid_pixels, 1]] = coordinates[valid_pixels, 2]
                # axis[pixels[valid_pixels, 0], pixels[valid_pixels, 1]] = 1

        # ta.shortest_distance(axr, ds.nodata, startval=1, distance=distance, feedback=ta.ConsoleFeedback())
        # ta.shortest_ref(axr, ds.nodata, startval=1, fillval=0, out=measure, feedback=ta.ConsoleFeedback())

        click.echo('Calculate Measure & Distance Raster')

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
            np.array(refaxis_pixels),
            valley_bottom,
            ds.nodata)

        distance = 5.0 * distance
        distance[valley_bottom == ds.nodata] = ds.nodata
        measure[valley_bottom == ds.nodata] = ds.nodata

        click.echo('Create DGOs')

        breaks = np.arange(444e3, 499e3, 200)
        dgo = np.int32(np.digitize(measure, breaks))
        dgo_measures = np.round(0.5 * (breaks + np.roll(breaks, 1)), 1)

        profile = ds.profile.copy()

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        with rio.open(output_measure, 'w', **profile) as dst:
            dst.write(measure, 1)

        profile.update(nodata=0, dtype='int32')
        with rio.open(output_dgo, 'w', **profile) as dst:
            dst.write(dgo, 1)

        click.echo('Vectorize DGOs')

        polygons = features.shapes(
            dgo,
            dgo > 0,
            connectivity=8,
            transform=ds.transform)

        schema = {
            'geometry': 'Polygon',
            'properties': [
                ('GID', 'int'),
                ('ROW', 'int:3'),
                ('COL', 'int:3'),
                ('M', 'float:10.2')
            ]
        }
        crs = fiona.crs.from_epsg(2154)
        options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)
        count = 0

        with fiona.open(output_dgo_shapefile, 'w', **options) as fst:
            for polygon, gid in polygons:
                geom = asShape(polygon).buffer(0.0)
                feature = {
                    'geometry': geom.__geo_interface__,
                    'properties': {
                        'GID': int(gid),
                        'ROW': row,
                        'COL': col,
                        'M': float(dgo_measures[int(gid)])
                    }
                }
                fst.write(feature)
                count += 1

        click.echo('Created %d DGOs' % count)
        click.echo('DGO Range = %d - %d' % (1, len(breaks)))
        click.echo('Measure Range =  %.0f - %.0f' % (np.min(breaks), np.max(breaks)))

def test_relz():

    row = 5
    col = 7
    elevation_raster = filename('tiled', row=row, col=col)
    valley_bottom_rasterfile = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_FLOW_05_07.tif'
    network_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_RHT_05_07.shp'
    # network_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_RHT_SMOOTH_05_07.shp'
    output_relative_z = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_NEAREST_05_07.tif'
    output_stream_distance = '/media/crousson/Backup/WORK/TestAin/AIN_NEARESTDIST_05_07.tif'

    with rio.open(valley_bottom_rasterfile) as ds:

        click.echo('Read Valley Bottom')

        valley_bottom = speedup.raster_buffer(ds.read(1), ds.nodata, 6.0)
        height, width = valley_bottom.shape
        profile = ds.profile.copy()

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

                pixels = ta.worldtopixel(coordinates, ds.transform, gdal=False)

                for a, b in zip(pixels[:-1], pixels[1:]):
                    for i, j in rasterize_linestring(a, b):
                        if intile(i, j) and (i, j) not in unique:
                            # distance[i, j] = 0
                            # measure[i, j] = m
                            z = elevations[i, j]
                            refaxis_pixels.append((i, j, z))
                            unique.add((i, j))

        # output_rasterized_ref = '/media/crousson/Backup/WORK/TestAin/AIN_RASTERIZEDREF_05_07.shp'
        # schema = {
        #     'geometry': 'Point',
        #     'properties': [
        #         ('GID', 'int'),
        #         ('I', 'float'),
        #         ('J', 'float'),
        #         ('Z', 'float')
        #     ]
        # }
        # crs = fiona.crs.from_epsg(2154)
        # options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)
        # with fiona.open(output_rasterized_ref, 'w', **options) as fst:
        #     for k, (i, j, z) in enumerate(refaxis_pixels):
        #         geom = {'type': 'Point', 'coordinates': ds.xy(i, j)}
        #         properties = {'GID': k, 'I': i, 'J': j, 'Z': float(z)}
        #         fst.write({'geometry': geom, 'properties': properties})


        click.echo('Calculate Reference & Distance Raster')

        reference, distance = nearest_value_and_distance(
            np.array(refaxis_pixels),
            valley_bottom,
            ds.nodata)

        distance = 5.0 * distance

        click.echo('Write output')

        distance[valley_bottom == ds.nodata] = ds.nodata

        with rio.open(output_stream_distance, 'w', **profile) as dst:
            dst.write(distance, 1)

        del distance

        elevations, _ = ReadRasterTile(row, col, 'dem')
        relative = elevations - reference
        relative[valley_bottom == ds.nodata] = ds.nodata

        with rio.open(output_relative_z, 'w', **profile) as dst:
            dst.write(relative, 1)

def map_ref_points(points):

    row = 5
    col = 7
    elevation_raster = filename('tiled', row=row, col=col)
    valley_bottom_rasterfile = '/media/crousson/Backup/WORK/TestAin/AIN_RELZ_FLOW_05_07.tif'
    # refaxis_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_AXREF_05_07.shp'
    network_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_RHT_05_07.shp'
    # network_shapefile = '/media/crousson/Backup/WORK/TestAin/AIN_RHT_SMOOTH_05_07.shp'

    output = '/media/crousson/Backup/WORK/TestAin/AIN_TESTPOINTS_05_07.shp'

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

                # m0 = feature['properties']['M0']

                coordinates = np.array([
                    coord(p) for p in feature['geometry']['coordinates']
                ], dtype='float32')

                # coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                #     coordinates[1:, :2] - coordinates[:-1, :2],
                #     axis=1))

                coordinates[:] = ta.worldtopixel(coordinates, ds.transform, gdal=False)

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

        elevations, _ = ReadRasterTile(row, col, 'dem')

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

if __name__ == '__main__':
    test()
    test_relz()
    map_ref_points([
        (888067, 6561917),
        (903433, 6590772),
        (905057, 6593619),
        (903460, 6590613),
        (901024, 6583392)
    ])
