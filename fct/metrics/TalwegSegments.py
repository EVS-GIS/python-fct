# coding: utf-8

"""
Swath-intersecting talweg segments
"""

from collections import defaultdict
from itertools import count

import numpy as np

import click
import xarray as xr
import fiona
import fiona.crs
from shapely.geometry import asShape, LineString

from ..config import DatasetParameter

class Parameters:
    """
    Talweg segments parameters
    """

    talweg = DatasetParameter(
        'stream/talweg polyline',
        type='input')

    polygons = DatasetParameter(
        'swaths polygons',
        type='input')

    output = DatasetParameter(
        'talweg segments', type='output')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.talweg = 'network-cartography-ready'
            self.polygons = 'swaths_refaxis_polygons'
            self.output = dict(key='talweg_segments', tiled=False)

        else:

            self.talweg = dict(key='ax_talweg', axis=axis)
            self.polygons = dict(key='ax_swaths_refaxis_polygons', axis=axis)
            self.output = dict(key='ax_talweg_segments', axis=axis, tiled=False)

def AggregateSwathPolygons(params: Parameters, **kwargs):

    shapefile = params.polygons.filename(tileset=None, **kwargs)
    geometries = dict()

    with fiona.open(shapefile) as fs:
        for feature in fs:

            if feature['properties']['VALUE'] == 2:

                axis = feature['properties']['AXIS']
                measure = feature['properties']['M']
                geometry = asShape(feature['geometry'])

                if (axis, measure) in geometries:
                    geometries[axis, measure] = geometries[axis, measure].union(geometry)
                else:
                    geometries[axis, measure] = geometry

    return geometries

def TalwegSegments(params: Parameters, **kwargs) -> xr.Dataset:
    """
    Swath-intersecting talweg segments
    """

    talweg_shapefile = params.talweg.filename(tileset=None, **kwargs)
    polygons = AggregateSwathPolygons(params)

    polygon_index = defaultdict(list)
    dataset = list()
    output = params.output.filename()

    def open_output_shapefile(output, options):

        schema = {
            'geometry': 'LineString',
            'properties': [
                ('GID', 'int'),
                ('AXIS', 'int'),
                ('MEASURE', 'float:8.1'),
                ('LENGTH', 'float:8.1'),
                ('X', 'float:8.2'),
                ('Y', 'float:8.2')
            ]
        }

        gid = count(1)

        with fiona.open(output, 'w', schema=schema, **options) as fst:
            while True:

                axis, measure, segment = (yield)
                length = segment.length
                middle = segment.interpolate(0.5, normalized=True)

                feature = {
                    'geometry': segment.__geo_interface__,
                    'properties': {
                        'GID': next(gid),
                        'AXIS': int(axis),
                        'MEASURE': float(measure),
                        'LENGTH': float(length),
                        'X': middle.x,
                        'Y': middle.y,
                    }
                }

                fst.write(feature)

    for key in polygons:
        axis, measure = key
        polygon_index[axis].append(measure)

    with click.progressbar(length=len(polygons)) as iterator:
        with fiona.open(talweg_shapefile) as fs:

            options = dict(driver=fs.driver, crs=fs.crs)

            sink = open_output_shapefile(output, options)
            next(sink)

            for feature in fs:

                axis = feature['properties']['AXIS']
                talweg = asShape(feature['geometry'])
                size = len(polygon_index[axis])

                values = np.zeros((size, 2), dtype='float32')

                for k, measure in enumerate(polygon_index[axis]):

                    polygon = polygons[axis, measure]
                    talweg_segment = talweg.intersection(polygon)
                    
                    if hasattr(talweg_segment, 'geoms'):

                        longest_segment = None
                        length_max = 0.0

                        for geom in talweg_segment.geoms:
                            if geom.length > length_max:

                                longest_segment = geom
                                length_max = geom.length

                        talweg_segment = longest_segment

                    sink.send((axis, measure, talweg_segment))
                    iterator.update(1)

            sink.close()
