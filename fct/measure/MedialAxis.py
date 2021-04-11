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

from shapely.geometry import LineString, MultiLineString
from shapely.ops import linemerge

from ..config import (
    config,
    # LiteralParameter,
    DatasetParameter
)
from ..network.ValleyBottomFeatures import (
    MASK_EXTERIOR,
    MASK_FLOOPLAIN_RELIEF,
    MASK_VALLEY_BOTTOM,
    MASK_TERRACE,
    MASK_SLOPE,
    MASK_HOLE
)
from .. import (
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

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.distance = dict(key='ax_nearest_distance', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            # self.swaths_polygons = 'ax_swaths_refaxis_polygons'
            self.medialaxis = dict(key='ax_medialaxis', axis=axis)

def medial_segments(voronoi, groups):
    """
    Extract Voronoi vertices equidistant of two points
    from two different groups.
    """

    for (p, q), rv in zip(voronoi.ridge_points, voronoi.ridge_vertices):

        u, v = sorted(rv)

        if u == -1 or groups[p] == groups[q]:
            continue

        vertex1 = voronoi.vertices[u]
        vertex2 = voronoi.vertices[v]
        yield LineString([vertex1, vertex2])


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

    with click.progressbar(coordinates.keys()) as iterator:
        for axis in iterator:

            try:

                voronoi = Voronoi(coordinates[axis], qhull_options='Qbb Qc')

            except:

                logger.error('Cannot calculate medial axis for axis %d', axis)
                continue

            lines = list(medial_segments(voronoi, groups[axis]))
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

    sink.close()
