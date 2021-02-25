# coding: utf-8

"""
Elevation Swath Profile
"""

from multiprocessing import Pool
from typing import Tuple

import numpy as np
import xarray as xr
import click

import rasterio as rio
import fiona
from shapely.geometry import asShape

from ..network.SwathPolygons import measure_to_swath_identifier
from ..tileio import as_window
from ..cli import starcall
from ..metadata import set_metadata

from ..config import (
    DatasetParameter,
    LiteralParameter
)

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
    distance = DatasetParameter(
        'distance to drainage pixels (raster)',
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

    def __init__(self):
        """
        Default parameter values
        """

        self.values = 'dem'
        self.nearest = 'nearest_drainage_axis'
        self.distance = 'nearest_distance'
        self.swaths = 'swaths_refaxis'
        self.polygons = 'swaths_refaxis_polygons'
        self.output = 'swath_elevation'

        self.is_continuous = True
        self.swath_length = 200.0
        self.distance_delta = 10.0
        self.percentiles = [5, 25, 50, 75, 95]
        self.labels = None
        # self.labels = {1: 'Slope', 2: 'Terrace', 3: 'Holes', 4: 'Relief', 5: 'Bottom'}

    @classmethod
    def continuous(cls, values, percentiles=None):

        params = cls()
        params.values = values
        params.is_continuous = True

        if percentiles is not None:
            params.percentiles = percentiles

        return params

    @classmethod
    def discrete(cls, values, labels):

        params = cls()
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

    with rio.open(params.distance.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        # distance_nodata = ds.nodata

    with rio.open(params.swaths.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        swaths = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        # swaths_nodata = ds.nodata
        swaths = measure_to_swath_identifier(swaths, params.swath_length)
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

    if params.is_continuous:

        density = np.zeros(len(middle), dtype='uint16')
        percentiles = np.zeros((len(middle), len(params.percentiles)), dtype='float32')

        for i, _ in enumerate(middle):

            maski = mask & (binned == i+1)
            count = density[i] = np.sum(maski)

            if count > 0:

                percentiles[i, :] = np.percentile(values[maski], params.percentiles)

        dataset = xr.Dataset({
            'axis': (('sample',), np.full_like(middle, axis, dtype='uint32')),
            'measure': (('sample',), np.full_like(middle, measure)),
            'distance': (('sample',), middle),
            'density': (('sample',), density),
            'height': (('sample', 'quantile'), percentiles)
        }, coords={
            'quantile': params.percentiles
        })

    else:

        labels = list(sorted(params.labels.items()))

        profile = np.zeros((len(middle), len(labels)), dtype='uint16')

        for i, _ in enumerate(middle):

            maski = mask & (binned == i+1)
            count = np.sum(maski)

            if count > 0:

                for k, (label, _) in enumerate(labels):

                    profile[i, k] = np.sum(maski & (values == label))

        dataset = xr.Dataset({
            'axis': (('sample',), np.full_like(middle, axis, dtype='uint32')),
            'measure': (('sample',), np.full_like(middle, measure)),
            'distance': (('sample',), middle),
            'profile': (('sample', 'klasses'), profile)
        }, coords={
            'klasses': [label for _, label in labels]
        })

    # return dataset.set_index(sample=('axis', 'measure', 'distance'))
    return dataset

def SwathProfile(params: Parameters, processes=1, **kwargs):

    shapefile = params.polygons.filename(tileset=None)
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

    def length():

        # with fiona.open(shapefile) as fs:
        #     return sum(1 for f in fs if f['properties']['VALUE'] == 2)

        return len(geometries)

    def arguments():

        # with fiona.open(shapefile) as fs:
        #     for feature in fs:

        #         if feature['properties']['VALUE'] != 2:
        #             continue

        #         axis = feature['properties']['AXIS']
        #         measure = feature['properties']['M']
        #         bounds = asShape(feature['geometry']).bounds

        for (axis, measure), geometry in geometries.items():

            yield (
                SwathProfileUnit,
                axis,
                measure,
                geometry.bounds,
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
