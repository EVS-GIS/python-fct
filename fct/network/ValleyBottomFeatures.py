# coding: utf-8

"""
Delineate valley bottom features from drainage dependent thresholds
"""

from collections import namedtuple
from multiprocessing import Pool
from typing import List, Callable

import numpy as np
from scipy.signal import convolve2d

import click
import rasterio as rio
from rasterio import features

from ..config import (
    DatasetParameter,
    LiteralParameter
)
from .. import speedup
from ..cli import starcall

from .SwathDrainage import (
    calculate_swaths,
    SwathDrainageDict
)

MASK_EXTERIOR = 0 # NODATA
MASK_SLOPE = 1
MASK_TERRACE = 2
MASK_HOLE = 3
MASK_FLOOPLAIN_RELIEF = 4
MASK_VALLEY_BOTTOM = 5

ValleyBottomThreshold = namedtuple(
    'ValleyBottomThreshold',
    ('drainage_min', 'distance_min', 'distance_max', 'height_max', 'slope_max'))

class Parameters:
    """
    Valley bottom delineation creation parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    dem = DatasetParameter('elevation raster (DEM)', type='input')
    height = DatasetParameter('height raster (HAND)', type='input')
    axis = DatasetParameter('nearest axis raster', type='input')
    measure = DatasetParameter('measure along reference axis', type='input')
    distance = DatasetParameter('distance raster to talweg', type='input')
    output = DatasetParameter('valley bottom features (raster)', type='output')
    slope = DatasetParameter('slope raster', type='output')

    swath_length = LiteralParameter(
        'swath disaggregation distance in measure unit (eg. meters)')
    # distance_min = LiteralParameter(
    #     'minimum distance before applying stop criteria, expressed in real distance units (eg. meters))')
    height_max = LiteralParameter(
        'maximum height above talweg')
    thresholds = LiteralParameter(
        'drainage dependent valley bottom parameters')
    patch_min_pixels = LiteralParameter(
        'minimum patch area expressed in pixels')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.dem = 'dem'

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.height = 'nearest_height'
            self.axis = 'nearest_drainage_axis' # 'axis_nearest'
            self.measure = 'axis_measure'
            self.distance = 'nearest_distance'
            self.output = 'valley_bottom_features'
            self.slope = 'slope'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.height = dict(key='ax_nearest_height', axis=axis)
            self.axis = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            self.distance = dict(key='ax_nearest_distance', axis=axis)
            self.output = dict(key='ax_valley_bottom_features', axis=axis)
            self.slope = 'off'

        self.swath_length = 200.0
        # self.distance_min = 20.0
        self.height_max = 20.0
        self.patch_min_pixels = 100

        self.thresholds = [
            # drainage area km², distance min, distance max, max height (depth), max slope (%)
            # ValleyBottomThreshold(0, 20.0, 100.0, 1.0, 25.0),
            # ValleyBottomThreshold(30, 20.0, 400.0, 2.0, 20.0),
            # ValleyBottomThreshold(300, 50.0, 1500.0, 4.0, 15.0),
            # ValleyBottomThreshold(1000, 100.0, 2500.0, 4.0, 7.0)
            ValleyBottomThreshold(0, 20.0, 100.0, 2.0, 10.0),
            ValleyBottomThreshold(30, 20.0, 400.0, 4.0, 7.0),
            ValleyBottomThreshold(250, 20.0, 1500.0, 5.0, 5.0),
            ValleyBottomThreshold(1000, 20.0, 2500.0, 6.0, 3.5),
            ValleyBottomThreshold(5000, 20.0, 3500.0, 6.5, 3.5),
            ValleyBottomThreshold(11500, 20.0, 4000.0, 7.5, 3.0),
            ValleyBottomThreshold(13000, 20.0, 4000.0, 8.5, 3.0)
        ]

def make_resolve_thresholds_fun(thresholds: List[ValleyBottomThreshold]) -> Callable:
    """
    Create a new function to resolve drainage dependent thresholds
    from on drainage-ordered list of threshold values.
    """

    def resolve_thresholds(sw_drainage: float) -> ValleyBottomThreshold:
        """
        Return drainage dependent thresholds
        """

        threshold = thresholds[0]

        for t in thresholds[1:]:

            if sw_drainage < t.drainage_min:
                break

            threshold = t

        return threshold

    return resolve_thresholds

def calculate_slope(dem_raster: str, nodata: float = 999.0):
    """
    Calculate slope in % from DEM elevations
    """

    with rio.open(dem_raster) as ds:
        arr = ds.read(1)
        xres = ds.res[0]
        yres = ds.res[1]

    x = np.array([[-1 / (8 * xres), 0, 1 / (8 * xres)],
                  [-2 / (8 * xres), 0, 2 / (8 * xres)],
                  [-1 / (8 * xres), 0, 1 / (8 * xres)]])
    y = np.array([[1 / (8 * yres), 2 / (8 * yres), 1 / (8 * yres)],
                  [0, 0, 0],
                  [-1 / (8 * yres), -2 / (8 * yres), -1 / (8 * yres)]])

    x_grad = convolve2d(arr, x, mode='same', boundary='symm')
    y_grad = convolve2d(arr, y, mode='same', boundary='symm')
    slope = 100.0 * np.sqrt(x_grad ** 2 + y_grad ** 2)
    # slope = np.arctan(np.sqrt(x_grad ** 2 + y_grad ** 2)) * (180. / np.pi)
    slope = slope.astype(ds.dtypes[0])

    slope[arr == ds.nodata] = nodata

    return slope

def ClassifyValleyBottomTile(
        row: int,
        col: int,
        params: Parameters,
        drainage: SwathDrainageDict,
        **kwargs):
    """
    Classify valley bottom features - tile algorithm
    """

    dem_raster = params.dem.tilename(row=row, col=col, **kwargs)
    # drainage_raster = params.drainage.tilename(row=row, col=col, **kwargs)
    height_raster = params.height.tilename(row=row, col=col, **kwargs)
    axis_raster = params.axis.tilename(row=row, col=col, **kwargs)
    measure_raster = params.measure.tilename(row=row, col=col, **kwargs)
    distance_raster = params.distance.tilename(row=row, col=col, **kwargs)

    if not measure_raster.exists():
        return

    output = params.output.tilename(row=row, col=col, **kwargs)

    slope = calculate_slope(dem_raster)
    swaths, measures = calculate_swaths(measure_raster, params.swath_length)

    # with rio.open(drainage_raster) as ds:
    #     drainage = ds.read(1)

    with rio.open(axis_raster) as ds:
        axis = ds.read(1)
        axis_nodata = ds.nodata

    with rio.open(distance_raster) as ds:
        distance = np.abs(ds.read(1))

    if isinstance(params.thresholds, Callable):

        resolve_thresholds = params.thresholds

    else:

        resolve_thresholds = make_resolve_thresholds_fun(params.thresholds)

    with rio.open(height_raster) as ds:

        height = ds.read(1)
        out = np.full_like(height, MASK_EXTERIOR, dtype='uint8')

        for ax in np.unique(axis):

            if ax == axis_nodata:
                continue

            for sw in np.unique(swaths[axis == ax]):

                if sw == 0 or (sw-1) >= len(measures):
                    continue

                sw_measure = measures[sw-1]
                sw_mask = (axis == ax) & (swaths == sw)

                sw_drainage = drainage[ax, sw_measure]
                sw_height_max = params.height_max
                thresholds = resolve_thresholds(sw_drainage)

                out[
                    sw_mask &
                    (distance <= thresholds.distance_max) &
                    (height <= sw_height_max)
                ] = MASK_VALLEY_BOTTOM

                out[
                    sw_mask &
                    (out == MASK_VALLEY_BOTTOM) &
                    (slope > thresholds.slope_max) &
                    (distance > thresholds.distance_min)
                ] = MASK_FLOOPLAIN_RELIEF

        out = features.sieve(out, params.patch_min_pixels)
        speedup.reclass_margin(out, MASK_FLOOPLAIN_RELIEF, MASK_EXTERIOR, MASK_SLOPE)

        for ax in np.unique(axis):

            if ax == axis_nodata:
                continue

            for sw in np.unique(swaths[axis == ax]):

                if sw == 0 or (sw-1) >= len(measures):
                    continue

                sw_measure = measures[sw-1]
                sw_mask = (axis == ax) & (swaths == sw)

                sw_drainage = drainage[ax, sw_measure]
                sw_height_max = params.height_max
                thresholds = resolve_thresholds(sw_drainage)

                out[
                    sw_mask &
                    (out == MASK_SLOPE) &
                    (height <= thresholds.height_max) &
                    (distance > thresholds.distance_min)
                ] = MASK_FLOOPLAIN_RELIEF

                out[
                    sw_mask &
                    (out == MASK_VALLEY_BOTTOM) &
                    (height > thresholds.height_max) &
                    (distance > thresholds.distance_min)
                ] = MASK_TERRACE

        profile = ds.profile.copy()

        if not params.slope.none:

            profile.update(nodata=999.0, compress='deflate')
            with rio.open(params.slope.tilename(row=row, col=col, **kwargs), 'w', **profile) as dst:
                dst.write(slope, 1)

        profile.update(dtype='uint8', nodata=0, compress='deflate')
        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def ClassifyValleyBottomFeatures(params: Parameters, drainage: SwathDrainageDict, processes: int = 1, **kwargs):
    """
    Classify valley bottom features :

    - apply drainage dependent thresholds
      on input rasters (height, slope, distance)

    - optionally, output slopes calculated from DEM.
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
                ClassifyValleyBottomTile,
                row,
                col,
                params,
                drainage,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass
