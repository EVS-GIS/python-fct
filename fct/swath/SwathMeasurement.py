# coding: utf-8

"""
Longitudinal swath generation :
discretize space along reference axis

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
import xarray as xr

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..config import config
# from ..tileio import ReadRasterTile
from ..tileio import as_window
from ..rasterize import rasterize_linestring, rasterize_linestringz
from .. import transform as fct
from .. import speedup
from .. import terrain_analysis as ta
from ..cli import starcall
from ..metadata import set_metadata

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

SwathMeasurementParams = namedtuple('SwathMeasurementParams', [
    'ax_mask',
    'ax_reference',
    'ax_talweg_distance',
    'output_distance',
    'output_measure',
    'output_swaths_raster',
    'output_swaths_shapefile',
    'output_swaths_bounds',
    'mdelta'
])

# def DefaultParameters():
#     """
#     Default parameters (valley swath units)
#     """

#     return dict(
#         ax_mask='ax_nearest_height',
#         ax_reference='ax_refaxis',
#         output_distance='ax_axis_distance',
#         output_measure='ax_axis_measure',
#         output_swaths_raster='ax_dgo',
#         output_swaths_shapefile='ax_dgo_parts',
#         output_swaths_bounds='ax_dgo_defs',
#         mdelta=200.0
#     )

def ValleyBottomParameters():
    """
    Default parameters (valley swath units)
    """

    return dict(
        ax_mask='ax_valley_mask',
        ax_reference='ax_refaxis',
        ax_talweg_distance='ax_nearest_distance',
        output_distance='ax_axis_distance',
        output_measure='ax_axis_measure',
        output_swaths_raster='ax_valley_swaths',
        output_swaths_shapefile='ax_valley_swaths_polygons',
        output_swaths_bounds='ax_valley_swaths_bounds',
        mdelta=200.0
    )

def ValleyMedialAxisParameters():
    """
    Default parameters (valley swath units)
    """

    return dict(
        ax_mask='ax_valley_mask',
        ax_reference='ax_valley_medialaxis',
        ax_talweg_distance='ax_nearest_distance',
        output_distance='ax_axis_distance',
        output_measure='ax_axis_measure',
        output_swaths_raster='ax_valley_swaths',
        output_swaths_shapefile='ax_valley_swaths_polygons',
        output_swaths_bounds='ax_valley_swaths_bounds',
        mdelta=200.0
    )

def ExtendedCorridorParameters():
    """
    Parameter set for extended corridor longitudinal disaggregation
    """

    return dict(
        ax_mask='ax_corridor_mask',
        ax_reference='ax_talweg',
        ax_talweg_distance='ax_nearest_distance',
        output_distance='ax_talweg_distance',
        output_measure='ax_talweg_measure',
        output_swaths_raster='ax_corridor_swaths',
        output_swaths_shapefile='ax_corridor_swaths_polygons',
        output_swaths_bounds='ax_corridor_swaths_bounds',
        mdelta=200.0
    )

def NaturalCorridorParameters():
    """
    Parameter set for natural corridor longitudinal disaggregation
    """

    return dict(
        ax_mask='ax_natural_corridor',
        ax_reference='ax_talweg',
        ax_talweg_distance='ax_nearest_distance',
        output_distance='ax_talweg_distance',
        output_measure='ax_talweg_measure',
        output_swaths_raster='ax_natural_corridor_swaths',
        output_swaths_shapefile='ax_natural_corridor_swaths_polygons',
        output_swaths_bounds='ax_natural_corridor_units_bounds',
        mdelta=200.0
    )

def DisaggregateTileIntoSwaths(axis, row, col, params, **kwargs):
    """
    see CarveLongitudinalSwaths
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
    output_swaths_raster = _tilename(params.output_swaths_raster)

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
        dgo = np.uint32(np.digitize(measure, breaks))

        def calculate_attrs():

            measures = np.round(0.5 * (breaks + np.roll(breaks, 1)), 1)
            boxes = speedup.flat_boxes(dgo)

            if not boxes:
                return dict()

            maximinj = itemgetter(2, 1)
            minimaxj = itemgetter(0, 3)

            lowerleft = fct.pixeltoworld(np.array([
                maximinj(box) for box in boxes.values()
            ], dtype='int32'), ds.transform)

            upperright = fct.pixeltoworld(np.array([
                minimaxj(box) for box in boxes.values()
            ], dtype='int32'), ds.transform)

            bounds = np.column_stack([lowerleft, upperright])

            return {
                swath: (measures[swath], bounds[k])
                for k, swath in enumerate(boxes)
                if swath > 0
            }

        profile.update(nodata=0, dtype='uint32')

        with rio.open(output_swaths_raster, 'w', **profile) as dst:
            dst.write(dgo, 1)

        return calculate_attrs()

def DisaggregateIntoSwaths(axis, ax_tiles='ax_tiles', processes=1, **kwargs):
    """
    Calculate measurement support rasters and
    create discrete longitudinal swath units along the reference axis

    @api    fct-swath:discretize

    @input  reference_axis: ax_refaxis
    @input  mask: ax_valley_mask
    @input  tiles: ax_shortest_tiles
    @param  mdelta: 200.0

    @output measure: ax_axis_measure
    @output distance: ax_axis_distance
    @output swath_raster: ax_valley_swaths
    @output swath_polygons: ax_valley_swaths_polygons
    @output swath_bounds: ax_valley_swaths_bounds
    """

    parameters = ValleyBottomParameters()
    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = SwathMeasurementParams(**parameters)

    tilefile = config.tileset().filename(ax_tiles, axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                DisaggregateTileIntoSwaths,
                axis,
                row,
                col,
                params,
                kwargs
            )

    g_attrs = dict()

    def merge_bounds(bounds1, bounds2):

        return (
            min(bounds1[0], bounds2[0]),
            min(bounds1[1], bounds2[1]),
            max(bounds1[2], bounds2[2]),
            max(bounds1[3], bounds2[3]),
        )

    def merge(attrs):

        if not attrs:
            # multiprocessing unpickles empty dict as list
            return

        g_attrs.update({
            key: (
                g_attrs[key][0],
                merge_bounds(g_attrs[key][1], attrs[key][1])
            )
            for key in attrs.keys() & g_attrs.keys()
        })

        g_attrs.update({
            key: (attrs[key][0], tuple(attrs[key][1]))
            for key in attrs.keys() - g_attrs.keys()
        })

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for attrs in iterator:
                merge(attrs)

    return g_attrs

def WriteSwathsBounds(axis, attrs, **kwargs):
    """
    Write swath coordinates (id, location on ref axis)
    and bounds (minx, miny, maxx, maxy)
    to netcdf file
    """

    parameters = ValleyBottomParameters()
    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = SwathMeasurementParams(**parameters)

    swaths = np.array(list(attrs.keys()), dtype='uint32')
    measures = np.array([value[0] for value in attrs.values()], dtype='float32')
    bounds = np.array([value[1] for value in attrs.values()], dtype='float32')

    dataset = xr.Dataset(
        {
            'measure': (('swath',), measures),
            'bounds': (('swath', 'coord'), bounds),
            'delta_measure': params.mdelta
        },
        coords={
            'axis': axis,
            'swath': swaths,
            'coord': ['minx', 'miny', 'maxx', 'maxy']
        })

    set_metadata(dataset, 'swath_bounds')

    dataset.attrs['geographic_object'] = params.ax_mask
    dataset.attrs['reference_axis'] = params.ax_reference

    output = config.filename(params.output_swaths_bounds, axis=axis)

    dataset.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'bounds': dict(zlib=True, complevel=9, least_significant_digit=2),
            'swath': dict(zlib=True, complevel=9)
        })

    return dataset

def ReadSwathsBounds(axis, params):

    filename = config.filename(params.output_swaths_bounds, axis=axis)
    dataset = xr.open_dataset(filename)
    dataset.load()

    dataset = dataset.sortby('measure')

    return {
        dataset['swath'].values[k]: (
            dataset['measure'].values[k],
            tuple(dataset['bounds'].values[k, :]))
        for k in range(dataset['swath'].shape[0])
    }

def VectorizeOneSwathPolygon(axis, gid, measure, bounds, params, **kwargs):
    """
    Vectorize swath polygon connected to talweg
    """

    tileset = config.tileset()

    swath_raster = tileset.filename(params.output_swaths_raster, axis=axis)
    # mask_raster = tileset.filename(params.ax_mask, axis=axis)
    # distance_raster = tileset.filename(params.output_distance, axis=axis)
    distance_raster = tileset.filename(params.ax_talweg_distance, axis=axis)

    with rio.open(swath_raster) as ds:

        window = as_window(bounds, ds.transform)
        swath = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        state = np.full_like(swath, 255, dtype='uint8')
        state[swath == gid] = 0
        state[(swath == gid) & (distance == 0)] = 1

        height, width = state.shape

        if height == 0 or width == 0:
            click.secho('Invalid swath %d with height = %d and width = %d' % (gid, height, width), fg='red')
            return gid, measure, list()

        # out = np.full_like(state, 255, dtype='uint32')
        distance = np.zeros_like(state, dtype='float32')

        speedup.continuity_mask(
            state,
            # out,
            distance,
            jitter=0.4)

        # vectorize state => 0: outer space, 2: inner unit

        transform = ds.transform * ds.transform.translation(
            window.col_off,
            window.row_off)

        polygons = features.shapes(
            state,
            state != 255,
            connectivity=8,
            transform=transform)

        return gid, measure, list(polygons)

def VectorizeSwathPolygons(axis, processes=1, **kwargs):
    """
    Vectorize spatial units' polygons
    """

    parameters = ValleyBottomParameters()
    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = SwathMeasurementParams(**parameters)

    defs = ReadSwathsBounds(axis, params)

    def arguments():

        for gid, (measure, bounds) in defs.items():
            yield (
                VectorizeOneSwathPolygon,
                axis,
                gid,
                measure,
                bounds,
                params,
                kwargs
            )

    output = config.filename(params.output_swaths_shapefile, mod=False, axis=axis)

    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('GID', 'int'),
            ('AXIS', 'int:4'),
            ('VALUE', 'int:4'),
            # ('ROW', 'int:3'),
            # ('COL', 'int:3'),
            ('M', 'float:10.2')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as dst:

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments())

            with click.progressbar(pooled, length=len(defs)) as iterator:
                for gid, measure, polygons in iterator:
                    for (polygon, value) in polygons:

                        geom = asShape(polygon).buffer(0.0)

                        feature = {
                            'geometry': geom.__geo_interface__,
                            'properties': {
                                'GID': int(gid),
                                'AXIS': axis,
                                'VALUE': int(value),
                                # 'ROW': row,
                                # 'COL': col,
                                'M': float(measure)
                            }
                        }

                        dst.write(feature)

def UpdateSwathTile(axis, tile, params):

    tileset = config.tileset()

    def _tilename(name):
        return tileset.tilename(name, axis=axis, row=tile.row, col=tile.col)

    swath_shapefile = config.filename(params.output_swaths_shapefile, axis=axis)
    swath_raster = _tilename(params.output_swaths_raster)

    if not os.path.exists(swath_raster):
        return

    with rio.open(swath_raster) as ds:

        swaths = ds.read(1)
        shape = ds.shape
        nodata = ds.nodata
        transform = ds.transform
        profile = ds.profile.copy()
        profile.update(compress='deflate', dtype='uint32')

    with fiona.open(swath_shapefile) as fs:

        def accept(feature):
            return all([
                feature['properties']['AXIS'] == axis,
                feature['properties']['VALUE'] != 2
            ])

        geometries = [
            (f['geometry'], 1) for f in fs.filter(bbox=tile.bounds)
            if accept(f)
        ]

        if geometries:

            mask_invalid = features.rasterize(
                geometries,
                out_shape=shape,
                transform=transform,
                fill=0,
                dtype='int32')

            mask_invalid = features.sieve(mask_invalid, 40) # TODO externalize parameter
            swaths[mask_invalid == 1] = nodata

    with rio.open(swath_raster, 'w', **profile) as dst:
        dst.write(swaths, 1)

def UpdateSwathRaster(axis, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):
    """
    Commit manual swath edits to swaths raster

    @api    fct-swath:update

    @input  tiles: ax_shortest_tiles
    @input  swath_raster: ax_valley_swaths

    @param  sieve_threshold: 40

    @output swath_raster: ax_valley_swaths
    """

    parameters = ValleyBottomParameters()
    parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
    kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
    params = SwathMeasurementParams(**parameters)

    tileset = config.tileset()

    def arguments():

        for tile in tileset.tiles():
            yield (
                UpdateSwathTile,
                axis,
                tile,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass

def VectorizeSpatialUnit(axis, gid, measure, bounds, params):

    tileset = config.tileset()

    units_raster = tileset.filename(params.output_swaths_raster, axis=axis)

    with rio.open(units_raster) as ds:

        window = as_window(bounds, ds.transform)
        swath = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        polygons = features.shapes(
            swath,
            swath > 0,
            connectivity=8,
            transform=ds.transform)

        return gid, measure, [polygon for polygon, gid in polygons]

def Vectorize(axis, params, attrs, processes=1, **kwargs):
    """
    Vectorize spatial units' polygons
    """

    def arguments():

        for gid, (measure, bounds) in attrs.items():
            yield (
                VectorizeSpatialUnit,
                axis,
                gid,
                measure,
                bounds,
                kwargs
            )

    output = config.filename(params.output_swaths_shapefile, axis=axis)

    schema = {
        'geometry': 'Polygon',
        'properties': [
            ('GID', 'int'),
            ('AXIS', 'int:4'),
            # ('ROW', 'int:3'),
            # ('COL', 'int:3'),
            ('M', 'float:10.2')
        ]
    }
    crs = fiona.crs.from_epsg(2154)
    options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as dst:

        with Pool(processes=processes) as pool:

            pooled = pool.imap_unordered(starcall, arguments())

            with click.progressbar(pooled, length=len(attrs)) as iterator:
                for gid, measure, polygons in iterator:
                    for polygon in polygons:

                        geom = asShape(polygon).buffer(0.0)

                        feature = {
                            'geometry': geom.__geo_interface__,
                            'properties': {
                                'GID': int(gid),
                                'AXIS': axis,
                                # 'ROW': row,
                                # 'COL': col,
                                'M': measure
                            }
                        }

                        dst.write(feature)

# ==========================================================================
# Code beneath only for diagnostic purpose.
# ==========================================================================

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

# ==========================================================================
# TODO Clean up dead code after this line
# ==========================================================================

# def VectorizeSpatialUnitsByTile(axis, row, col, params, **kwargs):
#     """
#     DOCME
#     """

#     # click.echo('Vectorize DGOs')

#     tileset = config.tileset()

#     def _tilename(dataset):
#         return tileset.tilename(
#             dataset,
#             axis=axis,
#             row=row,
#             col=col)

#     output_swaths_raster = _tilename(params.output_swaths_raster)
#     output_swaths_shapefile = _tilename(params.output_swaths_shapefile)

#     with rio.open(output_swaths_raster) as ds:

#         dgo = ds.read(1)

#         polygons = features.shapes(
#             dgo,
#             dgo > 0,
#             connectivity=8,
#             transform=ds.transform)

#         schema = {
#             'geometry': 'Polygon',
#             'properties': [
#                 ('GID', 'int'),
#                 ('AXIS', 'int:4'),
#                 ('ROW', 'int:3'),
#                 ('COL', 'int:3'),
#                 ('M', 'float:10.2')
#             ]
#         }
#         crs = fiona.crs.from_epsg(2154)
#         options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)
#         count = 0
#         dgos = list()

#         with fiona.open(output_swaths_shapefile, 'w', **options) as fst:
#             for polygon, gid in polygons:
                
#                 geom = asShape(polygon).buffer(0.0)
#                 measure = float(measures[int(gid)])
#                 feature = {
#                     'geometry': geom.__geo_interface__,
#                     'properties': {
#                         'GID': int(gid),
#                         'AXIS': axis,
#                         'ROW': row,
#                         'COL': col,
#                         'M': measure
#                     }
#                 }
                
#                 fst.write(feature)
#                 count += 1

#                 dgos.append((measure, int(gid), axis, row, col))

#         # click.echo('Created %d DGOs' % count)
#         # click.echo('DGO Range = %d - %d' % (1, len(breaks)))
#         # click.echo('Measure Range =  %.0f - %.0f' % (np.min(breaks), np.max(breaks)))

#         return dgos

# def AggregateSpatialUnits(axis, ax_tiles='ax_tiles', **kwargs):
#     """
#     Aggregate units tiles together
#     """

#     parameters = ValleyBottomParameters()
#     parameters.update({key: kwargs[key] for key in kwargs.keys() & parameters.keys()})
#     kwargs = {key: kwargs[key] for key in kwargs.keys() - parameters.keys()}
#     params = SwathMeasurementParams(**parameters)

#     output = config.filename(params.output_swaths_shapefile, axis=axis)

#     schema = {
#         'geometry': 'Polygon',
#         'properties': [
#             ('GID', 'int'),
#             ('AXIS', 'int:4'),
#             ('ROW', 'int:3'),
#             ('COL', 'int:3'),
#             ('M', 'float:10.2')
#         ]
#     }
#     crs = fiona.crs.from_epsg(2154)
#     options = dict(driver='ESRI Shapefile', crs=crs, schema=schema)

#     tilefile = config.tileset().filename(ax_tiles, axis=axis)

#     with open(tilefile) as fp:
#         tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

#     with fiona.open(output, 'w', **options) as dst:
#         with click.progressbar(tiles) as iterator:

#             for row, col in iterator:

#                 shapefile = config.tileset().tilename(
#                     params.output_swaths_shapefile,
#                     axis=axis,
#                     row=row,
#                     col=col)

#                 if os.path.exists(shapefile):

#                     with fiona.open(shapefile) as fs:

#                         for feature in fs:
#                             feature['properties'].update(AXIS=axis)
#                             dst.write(feature)
