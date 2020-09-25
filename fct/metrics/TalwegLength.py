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

from operator import itemgetter

import numpy as np

import click
import xarray as xr
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import (
    asShape,
    MultiLineString)

from ..config import config

def MetricTalwegLength(axis, **kwargs):
    """
    Calculate intercepted talweg and reference axis length
    for every swath
    """

    talweg_feature = config.filename('ax_talweg', axis=axis, **kwargs)
    refaxis_feature = config.filename('ax_refaxis', axis=axis)
    measure_raster = config.tileset().filename('ax_axis_measure', axis=axis, **kwargs)
    swath_features = config.filename('ax_valley_swaths_polygons', axis=axis, **kwargs)

    # Sort talweg segments by first point M coordinate, descending

    talweg_fids = list()
    segments = list()

    with rio.open(measure_raster) as ds:
        with fiona.open(talweg_feature) as fs:
            for feature in fs:

                fid = feature['id']
                firstm = next(ds.sample([feature['geometry']['coordinates'][0][:2]], 1))
                talweg_fids.append((fid, firstm))

    with fiona.open(talweg_feature) as fs:
        for fid, _ in reversed(sorted(talweg_fids, key=itemgetter(1))):

            feature = fs.get(fid)
            segments.append(asShape(feature['geometry']))

    with fiona.open(refaxis_feature) as fs:

        # assert len(fs) == 1
        refaxis_segments = list()

        for feature in fs:

            refaxis_segments.append(asShape(feature['geometry']))

    talweg = MultiLineString(segments)
    refaxis = MultiLineString(refaxis_segments)

    with fiona.open(swath_features) as fs:

        size = len(fs)
        gids = np.zeros(size, dtype='uint32')
        measures = np.zeros(size, dtype='float32')
        lengths = np.zeros((size, 2), dtype='float32')

        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                gid = feature['properties']['GID']
                polygon = asShape(feature['geometry'])

                gids[k] = gid
                measures[k] = feature['properties']['M']

                talweg_length = talweg.intersection(polygon).length
                lengths[k, 0] = talweg_length

                refaxis_length = refaxis.intersection(polygon).length
                lengths[k, 1] = refaxis_length

    metrics = xr.Dataset(
        {
            'swath': ('measure', gids),
            'talweg_length': ('measure', lengths[:, 0]),
            'refaxis_length': ('measure', lengths[:, 1]),
            'swath_length': 200.0
        },
        coords={
            'axis': axis,
            'measure': measures
        })

    # Metadata

    return metrics
