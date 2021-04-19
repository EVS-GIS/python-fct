# coding: utf-8

"""
Calculate swaths max drainage
"""

from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, Tuple, Callable, Union

import numpy as np
from scipy.interpolate import interp1d
import xarray as xr

import click
import rasterio as rio

from ..cli import starcall
from ..config import (
    DatasetParameter,
    LiteralParameter
)
from ..metadata import set_metadata


SwathDrainageDict = Dict[Tuple[int, float], float]

class Parameters:
    """
    Valley bottom delineation creation parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')
    drainage = DatasetParameter(
        'drainage raster',
        type='input')
    axis = DatasetParameter(
        'nearest axis raster',
        type='input')
    measure = DatasetParameter(
        'measure along reference axis',
        type='input')
    output = DatasetParameter(
        'netcdf drainage area dataset',
        type='output')

    swath_length = LiteralParameter(
        'swath disaggregation distance in measure unit (eg. meters)')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.drainage = 'acc'
        self.swath_length = 200.0

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.axis = 'nearest_drainage_axis' # 'axis_nearest'
            self.measure = 'axis_measure'
            self.output = dict(key='drainage_area', tiled=False)

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.axis = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            self.output = dict(key='metrics_drainage_area', axis=axis, tiled=False)

def calculate_swaths(measure_raster: str, swath_length: float):
    """
    Transform measure raster into swath identifiers
    """

    with rio.open(measure_raster) as ds:
        measure = ds.read(1)

    measure_min = np.floor(np.min(measure) / swath_length) * swath_length
    measure_max = np.ceil(np.max(measure) / swath_length) * swath_length
    breaks = np.arange(measure_min, measure_max + swath_length, swath_length)
    swaths = np.uint32(np.digitize(measure, breaks))

    measures = np.round(0.5 * (breaks[:-1] + breaks[1:]), 1)

    return swaths, measures

def SwathDrainageTile(row: int, col: int, params: Parameters, **kwargs):
    """
    Return tile's swaths max drainage area
    """

    axis_raster = params.axis.tilename(row=row, col=col, **kwargs)
    drainage_raster = params.drainage.tilename(row=row, col=col, **kwargs)
    measure_raster = params.measure.tilename(row=row, col=col, **kwargs)

    result = dict()

    if not measure_raster.exists():
        return result

    swaths, measures = calculate_swaths(measure_raster, params.swath_length)

    with rio.open(drainage_raster) as ds:
        drainage = ds.read(1)

    with rio.open(axis_raster) as ds:
        axis = ds.read(1)
        axis_nodata = ds.nodata

    for ax in np.unique(axis):

        if ax == axis_nodata:
            continue

        for sw in np.unique(swaths[axis == ax]):

            if sw == 0 or (sw-1) >= len(measures):
                continue

            sw_measure = measures[sw-1]
            sw_mask = (axis == ax) & (swaths == sw)

            # compute swath drainage in separate loop
            # in order to avoid tile boundary problems
            sw_drainage = np.max(drainage[sw_mask])

            result[ax, sw_measure] = sw_drainage

    return result

def ensure_monotonic(drainage: SwathDrainageDict) -> SwathDrainageDict:
    """
    Clamp drainage values to ensure monotonic increase
    in measure descending order
    """

    values = defaultdict(list)
    monotonic = dict()

    for (ax, measure), value in drainage.items():

        values[ax].append((measure, value))

    for ax in values:

        value_min = 0.0

        # apply some sort of median filter to remove errors
        data = np.array(values[ax])
        data = (
            xr.DataArray(data[:, 1], dims=('measure',), coords=dict(measure=data[:, 0]))
            .sortby('measure', ascending=False)
            .rolling(measure=3, min_periods=1, center=True)
            .median()
        )

        for measure, value in zip(data.measure.values, data.values):

            if value < value_min:
                value = value_min
            else:
                value_min = value

            monotonic[ax, measure] = value

    return monotonic

def SwathDrainage(params: Parameters, processes: int = 1, **kwargs) -> SwathDrainageDict:
    """
    Calculate swaths max drainage area
    """

    tilefile = params.tiles.filename()

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                SwathDrainageTile,
                row,
                col,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        drainage = dict()

        with click.progressbar(pooled, length=length()) as iterator:
            for t_drainage in iterator:

                drainage.update({
                    k: max(drainage[k], t_drainage[k])
                    for k in t_drainage.keys() & drainage.keys()
                })

                drainage.update({
                    k: t_drainage[k]
                    for k in t_drainage.keys() - drainage.keys()
                })

        return ensure_monotonic(drainage)

def WriteDrainageToDisk(drainage: SwathDrainageDict, params: Parameters, **kwargs):

    size = len(drainage)
    axis = np.zeros(size, dtype='uint32')
    measure = np.zeros(size, dtype='float32')
    values = np.zeros(size, dtype='float32')

    for k, (ax, m) in enumerate(drainage):

        axis[k] = ax
        measure[k] = m
        values[k] = drainage[ax, m]

    data = xr.Dataset(
        {
            'drainage_area': (('swath',), values)
        }, coords={
            'axis': (('swath',), axis),
            'measure': (('swath',), measure)
        })

    set_metadata(data, 'metrics_drainage_area')
    output = params.output.filename(**kwargs)

    data.to_netcdf(
        output,
        'w',
        encoding={
            'axis': dict(zlib=True, complevel=9),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'drainage_area': dict(zlib=True, complevel=9, least_significant_digit=3)
        })

def ReadDrainageFromDisk(filename: str) -> SwathDrainageDict:

    data = xr.open_dataset(filename).load()

    drainage = {
        (data.axis.values[k], data.measure.values[k]): data.drainage_area.values[k]
        for k in data.swath
    }

    return drainage

def create_interpolate_drainage_fun(drainage: Union[xr.Dataset, SwathDrainageDict], axis: int) -> Callable:
    """
    create drainage interpolation function
    """

    if isinstance(drainage, xr.Dataset):

        data = drainage.sel(axis=axis)

        xy = np.column_stack([
            data.measure.values,
            data.drainage_area.values
        ])

    else:

        values = [
            (measure, value)
            for (ax, measure), value in drainage.items()
            if ax == axis
        ]

        xy = np.array(values, dtype='float32')

    return interp1d(xy[:, 0], xy[:, 1], assume_sorted=False, fill_value='extrapolate')
