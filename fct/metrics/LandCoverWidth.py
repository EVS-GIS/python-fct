# coding: utf-8

"""
LandCover (Buffer) Width Metrics

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

from collections import namedtuple
import numpy as np
import xarray as xr
import click
import fiona

from ..config import config
from .CorridorWidth import swath_width

DatasetParameter = namedtuple('DatasetParameter', [
    'swath_features', # ax_swath_features
    'swath_data', # ax_swath_landcover
])


def LandCoverWidth(axis, datasets, swath_length=200.0, resolution=5.0, **kwargs):
    """
    Aggregate landCover swath data
    """

    swath_shapefile = config.filename(datasets.swath_features, axis=axis, **kwargs)

    gids = list()
    measures = list()
    values = list()

    with fiona.open(swath_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']

                swathfile = config.filename(
                    datasets.swath_data,
                    axis=axis,
                    gid=gid,
                    **kwargs)

                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                classes = data['classes']
                swath = data['swath']

                density = data['density']

                if x.shape[0] < 3:

                    gids.append(gid)
                    measures.append(measure)
                    width = np.zeros((9, 2), dtype='float32')
                    values.append(width)
                    continue

                # unit width of observations
                unit_width = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                unit_width[0] = x[1] - x[0]
                unit_width[-1] = x[-1] - x[-2]

                axis_dominant = np.ma.argmax(
                    np.ma.masked_array(
                        swath[:, :, 0],
                        np.isnan(swath[:, :, 0])
                    ), axis=1)

                nearest_dominant = np.ma.argmax(
                    np.ma.masked_array(
                        swath[:, :, 1],
                        np.isnan(swath[:, :, 1])
                    ), axis=1)

                # width = np.zeros(9, dtype='float32')
                width = np.zeros((9, 3), dtype='float32')

                for k, klass in enumerate(classes):

                    if klass == 255:
                        continue

                    # Total width
                    selection = (axis_dominant == k)
                    width[klass, 0] = swath_width(
                        selection,
                        unit_width,
                        density[:, 0],
                        swath_length,
                        resolution)

                    # Left bank width
                    selection = (nearest_dominant == k) & (x >= 0)
                    width[klass, 1] = swath_width(
                        selection,
                        unit_width,
                        density[:, 1],
                        swath_length,
                        resolution)

                    # Right bank width
                    selection = (nearest_dominant == k) & (x < 0)
                    width[klass, 2] = swath_width(
                        selection,
                        unit_width,
                        density[:, 1],
                        swath_length,
                        resolution)

                # values.append(tuple([gid, measure] + width.tolist()))

                gids.append(gid)
                measures.append(measure)
                values.append(width)

    # dtype = [
    #     ('gid', 'int'),
    #     ('measure', 'float32')
    # ] + [('lcc%d' % k, 'float32') for k in range(9)]

    # return np.sort(np.array(values, dtype=np.dtype(dtype)), order='measure')
    gids = np.array(gids, dtype='uint32')
    measures = np.array(measures, dtype='float32')
    data = np.array(values, dtype='float32')

    return xr.Dataset(
        {
            'swath': ('measure', gids),
            'lcw': (('measure', 'landcover', 'type'), data)
        },
        coords={
            'axis': axis,
            'measure': measures,
            'landcover': [
                'Water Channel',
                'Gravel Bars',
                'Natural Open',
                'Forest',
                'Grassland',
                'Crops',
                'Diffuse Urban',
                'Dense Urban',
                'Infrastructures'
            ],
            'type': [
                'total',
                'left',
                'right'
            ]
        })

def LandCoverTotalWidth(axis, subset='landcover', swath_length=200.0, resolution=5.0):
    """
    Defines
    -------

    lcw(k): total landcover width (meter) for land cover class k
    """

    datasets = DatasetParameter(
        swath_features='ax_swath_features',
        swath_data='ax_swath_landcover'
    )

    return LandCoverWidth(
        axis,
        datasets,
        swath_length=swath_length,
        resolution=resolution,
        subset=subset.upper())

def ContinuousBufferWidth(axis, subset='continuity', swath_length=200.0, resolution=5.0):
    """
    Defines
    -------

    lcw(k): continuous buffer width (meter) for land cover class k
    """

    datasets = DatasetParameter(
        swath_features='ax_swath_features',
        swath_data='ax_swath_landcover'
    )

    return LandCoverWidth(
        axis,
        datasets,
        swath_length=swath_length,
        resolution=resolution,
        subset=subset.upper())

def WriteLandCoverWidth(axis, data, output='metrics_lcw', **kwargs):

    output = config.filename(output, axis=axis, **kwargs)

    # data = lcw.merge(lcc).sortby(lcw['measure'])

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'lcw': dict(zlib=True, complevel=9, least_significant_digit=2)
        })
