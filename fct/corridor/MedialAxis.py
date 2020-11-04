# coding: utf-8

"""
Corridor medial axis (Voronoi method)

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
import numpy as np
from scipy.spatial import Voronoi

import click
import rasterio as rio
import fiona
import fiona.crs

from shapely.geometry import LineString, MultiLineString, Polygon, asShape
from shapely.ops import linemerge, unary_union

from ..config import config
from .. import (
    transform,
    speedup
)

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


def BoundaryPoints(axis):
    """
    Extract boundary points from corridor swath units.

    Returns the coordinates (x, y) of the points,
    and the side of corridor of each point.
    """

    tilefile = config.tileset().filename('ax_shortest_tiles', axis=axis)

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

            swath_raster = config.tileset().tilename(
                # 'ax_swaths_refaxis',
                # 'ax_valley_mask',
                'ax_nearest_height',
                axis=axis,
                row=row,
                col=col)

            distance_raster = config.tileset().tilename(
                'ax_nearest_distance',
                axis=axis,
                row=row,
                col=col)

            if not os.path.exists(swath_raster):
                continue

            with rio.open(swath_raster) as ds:

                swaths = ds.read(1)
                mask = np.uint8(swaths != ds.nodata)
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

def DissolveSwathBounds(axis):

    swath_shapefile = config.filename('ax_swaths_refaxis_polygons', axis=axis)
    boxes = list()

    with fiona.open(swath_shapefile) as fs:
        for feature in fs:

            if feature['properties']['VALUE'] == 2:

                bounds = asShape(feature['geometry']).bounds
                box = Polygon.from_bounds(*bounds)
                boxes.append(box.buffer(10.0))

    dissolved = unary_union(boxes)

    return dissolved

def FixOrientation(axis, medialaxis):

    measure_raster = config.tileset().filename('ax_axis_measure', axis=axis)

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

def MedialAxis(axis):
    """
    Calculate corridor medial axis
    using boundary points from first iteration swath units
    """

    output = config.filename('ax_medialaxis', axis=axis)

    coordinates, groups = BoundaryPoints(axis)

    voronoi = Voronoi(coordinates, qhull_options='Qbb Qc')
    lines = list(medial_segments(voronoi, groups))
    medialaxis = linemerge(lines)

    envelope = DissolveSwathBounds(axis)

    if hasattr(medialaxis, 'geoms'):

        lines = [line.intersection(envelope) for line in medialaxis.geoms]
        lines = sorted(lines, key=lambda line: -line.length)
        medialaxis = lines[0]

    else:

        medialaxis = medialaxis.intersection(envelope)

    if hasattr(medialaxis, 'geoms'):

        geoms = list()

        for geom in medialaxis.geoms:

            geoms.append(FixOrientation(axis, geom))

        medialaxis = MultiLineString(geoms)

    else:

        medialaxis = FixOrientation(axis, medialaxis)

    crs = fiona.crs.from_epsg(2154)
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
