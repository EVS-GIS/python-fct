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

from ..metadata import set_metadata
from ..config import config

DatasetParameter = namedtuple('DatasetParameter', [
    'landcover',
    'swath_features', # ax_swath_features
    'swath_data', # ax_swath_landcover_npz
])

def swath_width(swath_area_pixels, unit_width, long_length, resolution):
    """
    Measure width across swath profile at selected positions
    """

    # if selection.size == 0:
    #     return np.nan

    # density = swath area per unit width
    max_density = long_length / resolution**2
    clamp = np.minimum(max_density*unit_width, swath_area_pixels)
    width = np.sum(clamp) / max_density

    return width

def LandCoverWidth(axis, method, datasets, swath_length=200.0, resolution=5.0, **kwargs):
    """
    Aggregate landCover swath data

    @api    fct-metrics:landcover-width

    @input  landcover: ax_continuity
    @input  swath_polygons: ax_valley_swaths_polygons
    @input  swath_profile_landcover: ax_swath_landcover_npz

    @param  method
    @param  swath_length: 200.0
    @param  raster_resolution: 5.0
    @param  landcover_classes: 9

    @output landcover_width: metrics_landcover_width
    """

    swath_shapefile = config.filename(datasets.swath_features, axis=axis, **kwargs)

    pixel_area = resolution**2

    with fiona.open(swath_shapefile) as fs:

        size = len(fs)
        swathids = np.zeros(size, dtype='uint32')
        measures = np.zeros(size, dtype='float32')
        buffer_area = np.zeros((size, 9, 2), dtype='float32')
        buffer_width = np.zeros((size, 9, 2), dtype='float32')
        valid = np.full(size, True)

        with click.progressbar(fs) as iterator:
            for i, feature in enumerate(iterator):

                if feature['properties']['VALUE'] == 0:
                    valid[i] = False
                    continue

                gid = feature['properties']['GID']
                measure = feature['properties']['M']

                swathids[i] = gid
                measures[i] = measure

                swathfile = config.filename(
                    datasets.swath_data,
                    axis=axis,
                    gid=gid,
                    **kwargs)

                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                classes = data['landcover_classes']
                landcover_swath = data['landcover_swath']

                # density = data['density']

                if x.shape[0] < 3:
                    continue

                # unit width of observations
                unit_width = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                unit_width[0] = x[1] - x[0]
                unit_width[-1] = x[-1] - x[-2]

                # axis_dominant = np.ma.argmax(
                #     np.ma.masked_array(
                #         landcover_swath[:, :, 0],
                #         np.isnan(landcover_swath[:, :, 0])
                #     ), axis=1)

                # nearest_dominant = np.ma.argmax(
                #     np.ma.masked_array(
                #         landcover_swath[:, :, 1],
                #         np.isnan(landcover_swath[:, :, 1])
                #     ), axis=1)

                # per landcover class corridor widths

                for k, klass in enumerate(classes):

                    if klass == 255:
                        continue

                    # 1. Total width

                    # aggregate pixel area by slice (distance di, class k)
                    swath_area_k = np.sum(landcover_swath[:, k, :], axis=1)

                    width_total = swath_width(
                        swath_area_k,
                        unit_width,
                        swath_length,
                        resolution)

                    # buffer_width[i, klass, 0] = width_total

                    # 2. Left & right bank real world area

                    # aggregate pixel area by slice (class k, side)
                    area_pixels_k_lr = np.sum(landcover_swath[:, k, :], axis=0)
                    buffer_area[i, klass, :] = area_pixels_k_lr * pixel_area

                    # 3. Left & right bank width

                    area_pixels_k_total = np.sum(area_pixels_k_lr, axis=0)

                    buffer_width[i, klass, :] = (
                        area_pixels_k_lr * width_total
                        / area_pixels_k_total
                    )

    # dtype = [
    #     ('gid', 'int'),
    #     ('measure', 'float32')
    # ] + [('lcc%d' % k, 'float32') for k in range(9)]

    # return np.sort(np.array(values, dtype=np.dtype(dtype)), order='measure')
    # gids = np.array(gids, dtype='uint32')
    # measures = np.array(measures, dtype='float32')
    # landcover_width = np.array(landcover_width_values, dtype='float32')

    dataset = xr.Dataset(
        {
            'swath': ('measure', swathids[valid]),
            'buffer_area': (('measure', 'landcover', 'side'), buffer_area[valid]),
            'buffer_width': (('measure', 'landcover', 'side'), buffer_width[valid])
        },
        coords={
            'axis': axis,
            'measure': measures[valid],
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
            'side': [
                'left',
                'right'
            ]
        })

    # Metadata

    set_metadata(dataset, 'metrics_landcover_width')

    # Extra metadata

    dataset.attrs['method'] = method
    dataset.attrs['source'] = config.basename(
        datasets.landcover,
        axis=axis,
        **kwargs)

    return dataset

def LandCoverTotalWidth(axis, subset='landcover', swath_length=200.0, resolution=5.0):
    """
    Defines
    -------

    lcw(k): total landcover width (meter) for land cover class k
    """

    datasets = DatasetParameter(
        landcover='',
        swath_features='ax_valley_swaths_polygons',
        swath_data='ax_swath_landcover_npz'
    )

    return LandCoverWidth(
        axis,
        'total landcover width',
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
        landcover='',
        swath_features='ax_valley_swaths_polygons',
        swath_data='ax_swath_landcover_npz'
    )

    return LandCoverWidth(
        axis,
        'continuous buffer width from river channel',
        datasets,
        swath_length=swath_length,
        resolution=resolution,
        subset=subset.upper())

def WriteLandCoverWidth(axis, data, output='metrics_landcover_width', **kwargs):

    output = config.filename(output, axis=axis, **kwargs)

    # data = lcw.merge(lcc).sortby(lcw['measure'])

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'buffer_area': dict(zlib=True, complevel=9, least_significant_digit=0),
            'buffer_width': dict(zlib=True, complevel=9, least_significant_digit=2)
        })
