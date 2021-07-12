# coding: utf-8

"""
Elevation Swath Profile
"""

import logging
from multiprocessing import Pool
from typing import Tuple

import numpy as np
import xarray as xr
import click

import rasterio as rio
import fiona
from shapely.geometry import asShape

from ..measure.SwathPolygons import measure_to_swath_identifier
from ..tileio import as_window
from ..cli import starcall
from ..metadata import set_metadata

from ..config import (
    DatasetParameter,
    LiteralParameter
)

from ..measure.SwathBounds import SwathBounds

class Parameters:
    """
    Elevation swath profile extraction parameters
    """

    values = DatasetParameter(
        'input variable raster',
        type='input')
    nearest = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')
    axis_distance = DatasetParameter(
        'distance to measure axis (raster)',
        type='input')
    talweg_distance = DatasetParameter(
        'distance to talweg (raster)',
        type='input')
    swaths = DatasetParameter(
        'swaths raster (discretized measures)',
        type='input')
    polygons = DatasetParameter(
        'swaths polygons',
        type='input')

    output = DatasetParameter(
        'elevation swath profile',
        type='output')

    is_continuous = LiteralParameter('whether input variable is continuous (true) or discrete')
    swath_length = LiteralParameter(
        'swath disaggregation distance in measure unit (eg. meters)')
    distance_delta = LiteralParameter(
        'sample width along distance dimension (in meters)')
    percentiles = LiteralParameter(
        'profile percentiles if continuous variable')
    labels = LiteralParameter(
        'values labels if discrete variable')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.values = 'dem'

        if axis is None:

            self.nearest = 'nearest_drainage_axis'
            self.axis_distance = 'axis_distance'
            self.talweg_distance = 'nearest_distance'
            self.swaths = 'swaths_refaxis'
            self.polygons = dict(key='swaths_refaxis_polygons', tiled=False)
            self.output = 'swath_elevation'

        else:

            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.axis_distance = dict(key='ax_axis_distance', axis=axis)
            self.talweg_distance = dict(key='ax_nearest_distance', axis=axis)
            self.swaths = dict(key='ax_swaths_refaxis', axis=axis)
            self.polygons = dict(key='ax_swaths_refaxis_polygons', axis=axis, tiled=False)
            self.output = dict(key='ax_swath_elevation', axis=axis)

        self.is_continuous = True
        self.swath_length = 200.0
        self.distance_delta = 10.0
        self.percentiles = [5, 25, 50, 75, 95]
        self.labels = None
        # self.labels = {1: 'Slope', 2: 'Terrace', 3: 'Holes', 4: 'Relief', 5: 'Bottom'}

    @classmethod
    def continuous(cls, values, axis=None, percentiles=None):
        """
        Create continuous data swath profile parameters
        """

        params = cls(axis=axis)
        params.values = values
        params.is_continuous = True

        if percentiles is not None:
            params.percentiles = percentiles

        return params

    @classmethod
    def discrete(cls, values, labels, axis=None):
        """
        Create discrete data swath profile parameters
        """

        params = cls(axis=axis)
        params.values = values
        params.labels = labels
        params.is_continuous = False

        return params

def SwathProfileUnit(
        axis: int,
        measure: float,
        bounds: Tuple[float, float, float, float],
        params: Parameters,
        **kwargs) -> xr.Dataset:

    logger = logging.getLogger(__name__)

    swath_length = params.swath_length
    swath = measure_to_swath_identifier(measure, swath_length)

    with rio.open(params.values.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        values = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        nodata = ds.nodata

    with rio.open(params.nearest.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        nearest = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        # nearest_nodata = ds.nodata

    with rio.open(params.axis_distance.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        # distance_nodata = ds.nodata

    with rio.open(params.talweg_distance.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        talweg_distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        talweg_distance_nodata = ds.nodata

    with rio.open(params.swaths.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        swaths = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        # swaths_nodata = ds.nodata
        swaths = measure_to_swath_identifier(swaths, params.swath_length)
    
    try:

        assert(nearest.shape == values.shape)
        assert(distance.shape == values.shape)
        assert(talweg_distance.shape == values.shape)
        assert(swaths.shape == values.shape)

    except AssertionError:

        logger.error('Window error on swath (%d, %f)', axis, measure)
        return None

    mask = (nearest == axis) & (swaths == swath) & (values != nodata)

    if np.sum(mask) == 0:
        return None

    distance_min = np.floor(np.min(distance[mask]) / swath_length) * swath_length
    distance_max = np.ceil(np.max(distance[mask]) / swath_length) * swath_length

    distance_bins = np.arange(
        distance_min,
        distance_max + params.distance_delta,
        params.distance_delta)

    binned = np.digitize(distance, distance_bins)
    middle = np.float32(0.5*(distance_bins[1:] + distance_bins[:-1]))
    left = (talweg_distance != talweg_distance_nodata) & (talweg_distance >= 0)
    right = (talweg_distance != talweg_distance_nodata) & (talweg_distance < 0)

    if params.is_continuous:

        density = np.zeros((len(middle), 2), dtype='uint16')
        means = np.zeros((len(middle), 2), dtype='float32')
        percentiles = np.zeros((len(middle), len(params.percentiles)), dtype='float32')

        for i, _ in enumerate(middle):

            maski = mask & (binned == i+1)
            count_left = density[i, 0] = np.sum(maski & left)
            count_right = density[i, 1] = np.sum(maski & right)

            if (count_left + count_right) > 0:

                if count_left > 0:
                    means[i, 0] = np.mean(values[maski & left])

                if count_right > 0:
                    means[i, 1] = np.mean(values[maski & right])

                percentiles[i, :] = np.percentile(values[maski], params.percentiles)

        dataset = xr.Dataset({
            'density': (('sample', 'side'), density),
            'mean': (('sample', 'side'), means),
            'profile': (('sample', 'quantile'), percentiles)
        }, coords={
            'axis': (('sample',), np.full_like(middle, axis, dtype='uint32')),
            'measure': (('sample',), np.full_like(middle, measure)),
            'distance': (('sample',), middle),
            'quantile': params.percentiles,
            'side': ['left', 'right']
        })

    else:

        labels = list(sorted(params.labels.items()))

        profile = np.zeros((len(middle), len(labels), 2), dtype='uint16')

        for i, _ in enumerate(middle):

            maski = mask & (binned == i+1)

            mask_left = maski & left
            mask_right = maski & right

            count_left = np.sum(mask_left)
            count_right = np.sum(mask_right)

            for k, (label, _) in enumerate(labels):

                mask_k = (values == label)

                if count_left > 0:
                    profile[i, k, 0] = np.sum(mask_k & mask_left)

                if count_right > 0:
                    profile[i, k, 1] = np.sum(mask_k & mask_right)

        dataset = xr.Dataset({
            'profile': (('sample', 'label', 'side'), profile)
        }, coords={
            'axis': (('sample',), np.full_like(middle, axis, dtype='uint32')),
            'measure': (('sample',), np.full_like(middle, measure)),
            'distance': (('sample',), middle),
            'label': [label for _, label in labels],
            'side': ['left', 'right']
        })

    # return dataset.set_index(sample=('axis', 'measure', 'distance'))
    return dataset

# def SwathBounds(params: Parameters, **kwargs):

#     shapefile = params.polygons.filename(tileset=None, **kwargs)
#     geometries = dict()

#     with fiona.open(shapefile) as fs:
#         for feature in fs:

#             if feature['properties']['VALUE'] == 2:

#                 axis = feature['properties']['AXIS']
#                 measure = feature['properties']['M']
#                 geometry = asShape(feature['geometry'])

#                 if (axis, measure) in geometries:
#                     geometries[axis, measure] = geometries[axis, measure].union(geometry)
#                 else:
#                     geometries[axis, measure] = geometry

#     return {
#         (axis, measure): geometries[axis, measure].bounds
#         for axis, measure in geometries
#     }

def SwathProfile(params: Parameters, processes=1, **kwargs):

    swath_bounds = SwathBounds(params.polygons)

    # shapefile = params.polygons.filename(tileset=None)
    # geometries = dict()

    # with fiona.open(shapefile) as fs:
    #     for feature in fs:

    #         if feature['properties']['VALUE'] == 2:

    #             axis = feature['properties']['AXIS']
    #             measure = feature['properties']['M']
    #             geometry = asShape(feature['geometry'])

    #             if (axis, measure) in geometries:
    #                 geometries[axis, measure] = geometries[axis, measure].union(geometry)
    #             else:
    #                 geometries[axis, measure] = geometry

    def length():

        # with fiona.open(shapefile) as fs:
        #     return sum(1 for f in fs if f['properties']['VALUE'] == 2)

        # return len(geometries)
        return len(swath_bounds)

    def arguments():

        # with fiona.open(shapefile) as fs:
        #     for feature in fs:

        #         if feature['properties']['VALUE'] != 2:
        #             continue

        #         axis = feature['properties']['AXIS']
        #         measure = feature['properties']['M']
        #         bounds = asShape(feature['geometry']).bounds

        for (axis, measure), bounds in swath_bounds.items():

            yield (
                SwathProfileUnit,
                axis,
                measure,
                bounds,
                params,
                kwargs
            )

    data = None

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for values in iterator:

                if values is None:
                    continue

                if data is None:
                    data = values
                else:
                    data = xr.concat([data, values], 'sample', 'all')

    return data
