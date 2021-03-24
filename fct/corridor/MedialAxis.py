# coding: utf-8

"""
Corridor medial axis (Voronoi method)
"""

import os
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
from ..tileio import border
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

    def __init__(self):
        """
        Default parameter values
        """

        self.tiles = 'ax_shortest_tiles'
        self.valley_bottom = 'ax_valley_bottom_final'
        self.distance = 'ax_nearest_distance'
        self.measure = 'ax_axis_measure'
        # self.swaths_polygons = 'ax_swaths_refaxis_polygons'
        self.medialaxis = 'ax_medialaxis'

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


def BoundaryPoints(axis, params, **kwargs):
    """
    Extract boundary points from corridor swath units.

    Returns the coordinates (x, y) of the points,
    and the side of corridor of each point.
    """

    tilefile = params.tiles.filename(axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def tiles():

        with open(tilefile) as fp:
            for line in fp:
                row, col = tuple(int(x) for x in line.split(','))
                yield row, col

    coordinates_all = np.zeros((0, 2), dtype='float32')
    groups_all = np.zeros(0, dtype='bool')

    with click.progressbar(tiles(), length=length()) as iterator:
        for row, col in iterator:

            # swath_raster = params.height.tilename(row=row, col=col, **kwargs)
            valley_bottom_raster = params.valley_bottom.tilename(axis=axis, row=row, col=col, **kwargs)
            distance_raster = params.distance.tilename(axis=axis, row=row, col=col, **kwargs)

            if not os.path.exists(valley_bottom_raster):
                continue

            with rio.open(valley_bottom_raster) as ds:

                valley_bottom = ds.read(1)
                # height, width = valley_bottom.shape

                valley_bottom[
                    (valley_bottom == MASK_SLOPE) |
                    (valley_bottom == MASK_TERRACE)
                ] = MASK_HOLE
                # valley_bottom[np.array(list(border(height, width)))] = MASK_EXTERIOR
                speedup.reclass_margin(valley_bottom, MASK_HOLE, MASK_EXTERIOR, MASK_EXTERIOR)

                mask = np.uint8(
                    (valley_bottom == MASK_VALLEY_BOTTOM) |
                    (valley_bottom == MASK_FLOOPLAIN_RELIEF) |
                    (valley_bottom == MASK_HOLE)
                )
                points = list(speedup.boundary(mask, 1, 0))

                if not points:
                    continue

                points = np.array(points, dtype='int32')
                coordinates = transform.pixeltoworld(points, ds.transform)

            with rio.open(distance_raster) as ds:

                distance = ds.read(1)
                # groups = distance[points[:, 0], points[:, 1]] >= 0
                distance = distance[points[:, 0], points[:, 1]]
                valid = (distance != ds.nodata)

                groups = distance[valid] >= 0
                coordinates = coordinates[valid]

            coordinates_all = np.concatenate([coordinates_all, coordinates], axis=0)
            groups_all = np.concatenate([groups_all, groups], axis=0)

    return coordinates_all, groups_all

# def DissolveSwathBounds(axis, params):

#     swath_shapefile = params.swaths_polygons.filename(axis=axis, tileset=None)
#     boxes = list()

#     with fiona.open(swath_shapefile) as fs:
#         for feature in fs:

#             if feature['properties']['VALUE'] == 2:

#                 bounds = asShape(feature['geometry']).bounds
#                 box = Polygon.from_bounds(*bounds)
#                 boxes.append(box.buffer(10.0))

#     dissolved = unary_union(boxes)

#     return dissolved

def ClipLinestringToMask(axis, linestring, params):

    valley_bottom_raster = params.valley_bottom.filename(axis=axis)

    with rio.open(valley_bottom_raster) as ds:

        coordinates = list(linestring.coords)
        values = list(map(int, ds.sample(coordinates, 1)))

        kmin = 0
        kmax = len(coordinates)-1

        while kmin < kmax:

            if values[kmin] not in (MASK_VALLEY_BOTTOM, MASK_FLOOPLAIN_RELIEF):
                kmin += 1
            else:
                break

        while kmax > kmin:

            if values[kmax] not in (MASK_VALLEY_BOTTOM, MASK_FLOOPLAIN_RELIEF):
                kmax -= 1
            else:
                break

        return LineString(coordinates[kmin:kmax+1])

def FixOrientation(axis, medialaxis, params):

    measure_raster = params.measure.filename(axis=axis)

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

def MedialAxis(axis, params):
    """
    Calculate corridor medial axis
    using boundary points from first iteration swath units
    """

    output = params.medialaxis.filename(axis=axis)

    coordinates, groups = BoundaryPoints(axis, params)
    print(len(coordinates))

    voronoi = Voronoi(coordinates, qhull_options='Qbb Qc')
    lines = list(medial_segments(voronoi, groups))
    medialaxis = linemerge(lines)

    # envelope = DissolveSwathBounds(axis, params)

    # if hasattr(medialaxis, 'geoms'):

    #     lines = [line.intersection(envelope) for line in medialaxis.geoms]
    #     lines = sorted(lines, key=lambda line: -line.length)
    #     medialaxis = lines[0]

    # else:

    #     medialaxis = medialaxis.intersection(envelope)

    if hasattr(medialaxis, 'geoms'):

        lines = [ClipLinestringToMask(axis, line, params) for line in medialaxis.geoms]
        lines = sorted(lines, key=lambda line: -line.length)
        medialaxis = lines[0]

    else:

        medialaxis = ClipLinestringToMask(axis, medialaxis, params)

    if hasattr(medialaxis, 'geoms'):

        geoms = list()

        for geom in medialaxis.geoms:

            geoms.append(FixOrientation(axis, geom, params))

        medialaxis = MultiLineString(geoms)

    else:

        medialaxis = FixOrientation(axis, medialaxis, params)

    crs = fiona.crs.from_epsg(config.workspace.srid)
    driver = 'ESRI Shapefile'
    schema = {
        'geometry': 'LineString',
        'properties': [('AXIS', 'int')]}
    options = dict(driver=driver, crs=crs, schema=schema)

    with fiona.open(output, 'w', **options) as fst:
        # test only one geometry
        fst.write({
            'geometry': medialaxis.__geo_interface__,
            'properties': {'AXIS': axis}
        })
