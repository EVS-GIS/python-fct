# coding: utf-8

"""
Per Swath Unit Geomorphic Metrics

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import defaultdict

import numpy as np

import click
import xarray as xr
import fiona
import fiona.crs
from shapely.geometry import asShape, LineString

from ..config import DatasetParameter

class Parameters:
    """
    Talweg length parameters
    """

    talweg = DatasetParameter(
        'stream/talweg polyline',
        type='input')

    polygons = DatasetParameter(
        'swaths polygons',
        type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.talweg = 'network-cartography-ready'
            self.polygons = 'swaths_refaxis_polygons'

        else:

            self.talweg = dict(key='ax_talweg', axis=axis)
            self.polygons = dict(key='ax_swaths_refaxis_polygons', axis=axis)

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

def TalwegLength(params: Parameters, **kwargs) -> xr.Dataset:
    """
    Calculate intercepted talweg and reference axis length
    for every swath
    """

    talweg_shapefile = params.talweg.filename(tileset=None, **kwargs)
    polygons = AggregateSwathPolygons(params)

    polygon_index = defaultdict(list)
    dataset = list()

    for key in polygons:
        axis, measure = key
        polygon_index[axis].append(measure)

    with click.progressbar(length=len(polygons)) as iterator:
        with fiona.open(talweg_shapefile) as fs:
            for feature in fs:

                axis = feature['properties']['AXIS']
                talweg = asShape(feature['geometry'])
                size = len(polygon_index[axis])

                values = np.zeros((size, 2), dtype='float32')

                for k, measure in enumerate(polygon_index[axis]):

                    polygon = polygons[axis, measure]
                    talweg_length = talweg.intersection(polygon).length
                    values[k] = (measure, talweg_length)
                    iterator.update(1)

                data = xr.Dataset(
                    {
                        'length_talweg': ('swath', values[:, 1])
                    }, coords={
                        'axis': ('swath', np.full(size, axis, dtype='uint32')),
                        'measure': ('swath', values[:, 0])
                    })

                dataset.append(data)

    return xr.concat(dataset, 'swath', 'all')

# def MetricTalwegLength(axis, **kwargs):
#     """
#     Calculate intercepted talweg and reference axis length
#     for every swath
#     """

#     talweg_feature = config.filename('ax_talweg', axis=axis, **kwargs)
#     refaxis_feature = config.filename('ax_refaxis', axis=axis)
#     measure_raster = config.tileset().filename('ax_axis_measure', axis=axis, **kwargs)
#     swath_features = config.filename('ax_valley_swaths_polygons', axis=axis, **kwargs)

#     # Sort talweg segments by first point M coordinate, descending

#     talweg_fids = list()
#     segments = list()

#     with rio.open(measure_raster) as ds:
#         with fiona.open(talweg_feature) as fs:
#             for feature in fs:

#                 fid = feature['id']
#                 firstm = next(ds.sample([feature['geometry']['coordinates'][0][:2]], 1))
#                 talweg_fids.append((fid, firstm))

#     with fiona.open(talweg_feature) as fs:
#         for fid, _ in reversed(sorted(talweg_fids, key=itemgetter(1))):

#             feature = fs.get(fid)
#             segments.append(asShape(feature['geometry']))

#     with fiona.open(refaxis_feature) as fs:

#         # assert len(fs) == 1
#         refaxis_segments = list()

#         for feature in fs:

#             refaxis_segments.append(asShape(feature['geometry']))

#     talweg = MultiLineString(segments)
#     refaxis = MultiLineString(refaxis_segments)

#     with fiona.open(swath_features) as fs:

#         size = len(fs)
#         gids = np.zeros(size, dtype='uint32')
#         measures = np.zeros(size, dtype='float32')
#         lengths = np.zeros((size, 2), dtype='float32')

#         with click.progressbar(fs) as iterator:
#             for k, feature in enumerate(iterator):

#                 gid = feature['properties']['GID']
#                 polygon = asShape(feature['geometry'])

#                 gids[k] = gid
#                 measures[k] = feature['properties']['M']

#                 talweg_length = talweg.intersection(polygon).length
#                 lengths[k, 0] = talweg_length

#                 refaxis_length = refaxis.intersection(polygon).length
#                 lengths[k, 1] = refaxis_length

#     metrics = xr.Dataset(
#         {
#             'swath': ('measure', gids),
#             'talweg_length': ('measure', lengths[:, 0]),
#             'refaxis_length': ('measure', lengths[:, 1]),
#             'swath_length': 200.0
#         },
#         coords={
#             'axis': axis,
#             'measure': measures
#         })

#     # Metadata

#     return metrics
