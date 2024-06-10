# coding: utf-8

"""
Corridor medial axis (Voronoi method)
"""

import os
from collections import defaultdict
import logging

import numpy as np
from scipy.spatial import Voronoi

import click
import rasterio as rio
import fiona
import fiona.crs
import xarray as xr

from shapely.geometry import (
    LineString,
    MultiLineString,
    shape
)
from shapely.ops import linemerge

from fct.config import (
    config,
    LiteralParameter,
    DatasetParameter
)
from fct.corridor.ValleyBottomFeatures import (
    MASK_EXTERIOR,
    MASK_FLOOPLAIN_RELIEF,
    MASK_VALLEY_BOTTOM,
    MASK_TERRACE,
    MASK_SLOPE,
    MASK_HOLE
)
from fct.simplify import simplify
from fct import (
    transform,
    speedup
)

class Parameters:
    """
    Medial axis parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')
    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')
    nearest = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')
    distance = DatasetParameter(
        'nearest distance to talweg (raster)',
        type='input')
    measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')
    # swaths_polygons = DatasetParameter(
    #     'swaths polygons',
    #     type='input')
    medialaxis = DatasetParameter(
        'valley medial axis',
        type='output')
    simplified = DatasetParameter(
        'simplified valley medial axis',
        type='output')
    vbw = DatasetParameter(
        'valley bottom width metric (netcdf)',
        type='output')
    swath_length = LiteralParameter('swath length')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.valley_bottom = 'valley_bottom_final'
            self.nearest = 'nearest_drainage_axis'
            self.distance = 'nearest_distance'
            self.measure = 'axis_measure'
            # self.swaths_polygons = 'ax_swaths_refaxis_polygons'
            self.medialaxis = 'medialaxis'
            self.simplified = 'medialaxis_simplified'
            # self.vbw = dict(key='width_valley_bottom_ma', tiled=False)
            self.vbw = dict(key='metrics_width_valley_bottom_tagged', axis=None, tiled=False, tag='MA_TMP')

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.distance = dict(key='ax_nearest_distance', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            # self.swaths_polygons = 'ax_swaths_refaxis_polygons'
            self.medialaxis = dict(key='ax_medialaxis', axis=axis)
            self.simplified = dict(key='ax_medialaxis_simplified', axis=axis)
            self.vbw = dict(key='metrics_width_valley_bottom_tagged', axis=axis, tiled=False, tag='MA_TMP')

        self.swath_length = 100.0

def medial_segments(voronoi, groups):
    """
    Extract Voronoi vertices equidistant of two points
    from two different groups.
    """

    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):

        u, v = sorted(rv)

        if u == -1 or groups[p] == groups[q]:
            continue

        # vertex1 = voronoi.vertices[u]
        # vertex2 = voronoi.vertices[v]

        midpoint = 0.5 * (voronoi.points[p] + voronoi.points[q])
        distance = np.linalg.norm(voronoi.points[p] - voronoi.points[q])

        yield LineString(voronoi.vertices[[u, v]]), midpoint, distance


def BoundaryPoints(params, **kwargs):
    """
    Extract boundary points from corridor swath units.

    Returns the coordinates (x, y) of the points,
    and the side of corridor of each point.
    """

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def tiles():

        with open(tilefile) as fp:
            for line in fp:
                row, col = tuple(int(x) for x in line.split(','))
                yield row, col

    coordinates_all = defaultdict(lambda: np.zeros((0, 2), dtype='float32'))
    groups_all = defaultdict(lambda: np.zeros(0, dtype='bool'))

    with click.progressbar(tiles(), length=length()) as iterator:
        for row, col in iterator:

            # swath_raster = params.height.tilename(row=row, col=col, **kwargs)
            nearest_raster = params.nearest.tilename(row=row, col=col, **kwargs)
            valley_bottom_raster = params.valley_bottom.tilename(row=row, col=col, **kwargs)
            distance_raster = params.distance.tilename(row=row, col=col, **kwargs)

            if not os.path.exists(valley_bottom_raster):
                continue

            with rio.open(nearest_raster) as ds:
                
                nearest = ds.read(1)
                axes = np.unique(nearest[nearest != ds.nodata])

            with rio.open(distance_raster) as ds:
                
                distance = ds.read(1)
                distance_nodata = ds.nodata

            with rio.open(valley_bottom_raster) as ds:

                valley_bottom = ds.read(1)
                # height, width = valley_bottom.shape

                # valley_bottom[
                #     (valley_bottom == MASK_SLOPE) |
                #     (valley_bottom == MASK_TERRACE)
                # ] = MASK_HOLE
                # # valley_bottom[np.array(list(border(height, width)))] = MASK_EXTERIOR
                # speedup.reclass_margin(valley_bottom, MASK_HOLE, MASK_EXTERIOR, MASK_EXTERIOR)

                mask = np.uint8(
                    (valley_bottom == MASK_VALLEY_BOTTOM) |
                    (valley_bottom == MASK_FLOOPLAIN_RELIEF) |
                    (valley_bottom == MASK_HOLE)
                )

                for axis in axes:

                    if axis == 0:
                        continue

                    mask_axis = np.copy(mask)
                    mask_axis[nearest != axis] = 0

                    points = list(speedup.boundary(mask_axis, 1, 0))

                    if not points:
                        continue

                    points = np.array(points, dtype='int32')
                    coordinates = transform.pixeltoworld(points, ds.transform)

                    # groups = distance[points[:, 0], points[:, 1]] >= 0
                    distance_points = distance[points[:, 0], points[:, 1]]
                    valid = (distance_points != distance_nodata)

                    groups = distance_points[valid] >= 0
                    coordinates = coordinates[valid]

                    coordinates_all[axis] = np.concatenate([coordinates_all[axis], coordinates], axis=0)
                    groups_all[axis] = np.concatenate([groups_all[axis], groups], axis=0)

    return coordinates_all, groups_all

def ClipLinestringToMask(axis, linestring, params):

    nearest_raster = params.nearest.filename()
    valley_bottom_raster = params.valley_bottom.filename()

    coordinates = list(linestring.coords)

    with rio.open(valley_bottom_raster) as ds:

        valley_bottom_values = np.array(list(map(int, ds.sample(coordinates, 1))))

    with rio.open(nearest_raster) as ds:

        nearest_values = np.array(list(map(int, ds.sample(coordinates, 1))))

    coordinates = np.array(coordinates)
    coordinates = coordinates[nearest_values == axis]

    kmin = 0
    kmax = len(coordinates)-1

    while kmin < kmax:

        if valley_bottom_values[kmin] not in (MASK_VALLEY_BOTTOM, MASK_FLOOPLAIN_RELIEF):
            kmin += 1
        else:
            break

    while kmax > kmin:

        if valley_bottom_values[kmax] not in (MASK_VALLEY_BOTTOM, MASK_FLOOPLAIN_RELIEF):
            kmax -= 1
        else:
            break

    if kmin == kmax:
        return None

    return LineString(coordinates[kmin:kmax+1])

def FixOrientation(medialaxis, params):

    measure_raster = params.measure.filename()

    with rio.open(measure_raster) as ds:

        points = [
            medialaxis.interpolate((k+1)/201, normalized=True)
            for k in range(200)
        ]

        measures = list(map(float, ds.sample([(p.x, p.y) for p in points], 1)))

    x = np.column_stack([
        np.arange(200),
        np.ones(200)
    ])

    coefs, _, _, _ = np.linalg.lstsq(x, measures, rcond=None)

    if coefs.item(0) > 0:

        return LineString(reversed(medialaxis.coords))

    return medialaxis

def MedialAxis(params):
    """
    Calculate corridor medial axis
    using boundary points from first iteration swath units
    """

    logger = logging.getLogger(__name__)
    coordinates, groups = BoundaryPoints(params)

    def open_medialaxis_sink(output, crs, driver='ESRI Shapefile'):

        schema = {
            'geometry': 'LineString',
            'properties': [('AXIS', 'int')]}
        options = dict(driver=driver, crs=crs, schema=schema)

        with fiona.open(output, 'w', **options) as fst:
            while True:

                axis, medialaxis = (yield)

                fst.write({
                    'geometry': medialaxis.__geo_interface__,
                    'properties': {'AXIS': int(axis)}
                })

    crs = fiona.crs.from_epsg(config.workspace.srid)
    sink = open_medialaxis_sink(params.medialaxis.filename(), crs)
    next(sink)

    vbw_list = list()

    with click.progressbar(coordinates.keys()) as iterator:
        for axis in iterator:

            try:

                voronoi = Voronoi(coordinates[axis], qhull_options='Qbb Qc')

            except:

                logger.error('Cannot calculate medial axis for axis %d', axis)
                continue

            # lines = list(medial_segments(voronoi, groups[axis]))
            
            try:
                tmp_lines, tmp_midpoints, tmp_distances = zip(*medial_segments(voronoi, groups[axis]))
            except:
                click.echo(f'Error with axis {axis}. skipping')
                continue
            
            lines = list()
            midpoints = list()
            distances = list()

            # 0. remove lines between two axes
            with rio.open(params.nearest.filename()) as ds:
                for line, midpoint, distance in zip(tmp_lines, tmp_midpoints, tmp_distances):
                    line_coords = list(line.coords)
                    nearest_values = np.array(list(map(int, ds.sample(line_coords, 1))))

                    if np.all(nearest_values == axis) or line.length < (2*params.swath_length):
                        lines.append(line)
                        midpoints.append(midpoint)
                        distances.append(distance)

            lines = tuple(lines)
            midpoints = tuple(midpoints)
            distances = tuple(distances)
                
            # 1. compute medial axis

            medialaxis = linemerge(lines)

            if hasattr(medialaxis, 'geoms'):

                lines = [
                    line for line in [
                        ClipLinestringToMask(axis, geom, params)
                        for geom in medialaxis.geoms
                    ]
                    if line is not None
                ]

                if not lines:

                    logger.error('Cannot calculate medial axis for axis %d', axis)
                    continue

                lines = sorted(lines, key=lambda line: -line.length)
                medialaxis = lines[0]

            else:

                medialaxis = ClipLinestringToMask(axis, medialaxis, params)

                if medialaxis is None:

                    logger.error('Cannot calculate medial axis for axis %d', axis)
                    continue

            if hasattr(medialaxis, 'geoms'):

                geoms = list()

                for geom in medialaxis.geoms:

                    geoms.append(FixOrientation(geom, params))

                medialaxis = MultiLineString(geoms)

            else:

                medialaxis = FixOrientation(medialaxis, params)

            sink.send((axis, medialaxis))

            # 2. compute feature size = valley bottom width

            nearest_raster = params.nearest.filename()
            measure_raster = params.measure.filename()
            distances = np.array(distances, dtype='float32')

            with rio.open(nearest_raster) as ds:

                nearest = np.array(list(ds.sample(midpoints, 1)), dtype='uint32')
                nearest = nearest.squeeze()
                valid = (nearest == axis)

            with rio.open(measure_raster) as ds:

                measures = np.array(list(ds.sample(midpoints, 1)), dtype='float32')
                measures = measures.squeeze()
                # valid = valid & (measures != ds.nodata)
                valid = (measures != ds.nodata)

            width = xr.DataArray(
                distances[valid],
                dims=('measure',),
                coords={
                    'measure': measures[valid]
                })

            dm = params.swath_length
            measure_min = np.floor(np.min(measures[valid]) / dm) * dm
            measure_max = np.ceil(np.max(measures[valid]) / dm) * dm
            bins = np.arange(measure_min, measure_max + dm, dm)
            grouped = width.groupby_bins('measure', bins).mean()

            measures = np.array([
                iv.mid for iv in grouped.measure_bins.values
            ], dtype='float32')

            values = grouped.values
            missing = np.isnan(values)

            if np.any(missing):

                values[missing] = np.interp(
                    measures[missing],
                    measures[~missing],
                    values[~missing])

            vbw = xr.DataArray(
                np.float32(grouped.values),
                dims=('swath',),
                coords={
                    'axis': (('swath',), np.full(grouped.size, axis, dtype='uint32')),
                    'measure': (('swath',), measures)
                })

            vbw_list.append(vbw)

    sink.close()

    # return measure valley bottom width
    return xr.concat(vbw_list, 'swath', 'all')

def SimplifyMedialAxis(params: Parameters, vbw: xr.Dataset = None):
    """
    Simplify medial axis using valley bottom width
    """

    def filtr(weight, width):
        """
        Simplification filter based on valley bottom width.
        Parameters obtained by trial and error,
        the wider the valley bottom, the more simplification we want.
        """

        if width > 2000:
            return weight > 1000000

        elif width > 1500:
            return weight > 500000

        elif width > 1000:
            return weight > 100000

        elif width > 500:
            return weight > 25000

        elif width > 200:
            return weight > 10000

        return weight > 2500

    if vbw is None:

        vbw = (
            xr.open_dataset(params.vbw.filename())
            .load()
            .set_index(swath=('axis', 'measure'))
        )

    else:

        vbw = vbw.set_index(swath=('axis', 'measure'))

    # assert('width_valley_bottom' in vbw)

    with fiona.open(params.medialaxis.filename()) as fs:

        def open_output_sink(filename, options):

            with fiona.open(filename, 'w', **options) as fst:
                while True:

                    axis, geometry = (yield)
                    fst.write({
                        'geometry': geometry.__geo_interface__,
                        'properties': {'AXIS': axis}
                    })

        options = dict(driver=fs.driver, schema=fs.schema, crs=fs.crs)
        sink = open_output_sink(params.simplified.filename(), options)
        next(sink)

        for feature in fs:

            axis = feature['properties']['AXIS']
            medialaxis = shape(feature['geometry'])

            coords = np.array(medialaxis.coords)
            _, weights = zip(*simplify(coords))

            measure_raster = params.measure.filename()

            with rio.open(measure_raster) as ds:
                measures = np.array(list(ds.sample(coords, 1)), dtype='float32')
                measures = measures.squeeze()

            smoothed = (
                vbw.sel(axis=axis)
                .sortby('measure')
                .rolling(measure=5, min_periods=1, center=True)
                .mean()
            )

            widths = np.interp(measures, smoothed.measure, smoothed.values)

            # simplified = LineString(
            #     smooth_chaikin(
            #         np.array([
            #             c for c, weight, width
            #             in zip(coords, weights, widths)
            #             if filtr(weight, width)
            #         ])))

            simplified = LineString([
                coord for coord, weight, width
                in zip(coords, weights, widths)
                if filtr(weight, width)
            ])

            # print(len(simplified.coords))
            sink.send((axis, simplified))

    sink.close()
