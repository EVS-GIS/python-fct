# coding: utf-8

"""
Create Valley Bottom Mask from Height Raster
"""

from collections import namedtuple
from multiprocessing import Pool

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

MASK_EXTERIOR = 0 # NODATA
MASK_SLOPE = 1
MASK_TERRACE = 2
MASK_FLOOPLAIN_RELIEF = 3
MASK_VALLEY_BOTTOM = 4

ValleyBottomThreshold = namedtuple(
    'ValleyBottomThreshold',
    ('drainage_min', 'distance_min', 'distance_max', 'height_max', 'slope_max'))

class Parameters:
    """
    Valley bottom mask creation parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    dem = DatasetParameter('elevation raster (DEM)', type='input')
    drainage = DatasetParameter('drainage raster', type='input')
    height = DatasetParameter('height raster (HAND)', type='input')
    axis = DatasetParameter('nearest axis raster', type='input')
    measure = DatasetParameter('measure along reference axis', type='input')
    distance = DatasetParameter('distance raster to talweg', type='input')
    output = DatasetParameter('valley bottom mask (raster)', type='output')
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

    def __init__(self):
        """
        Default parameter values
        """

        self.tiles = 'shortest_tiles'
        self.dem = 'dem'
        self.drainage = 'acc'
        self.height = 'nearest_height'
        self.axis = 'nearest_drainage_axis' # 'axis_nearest'
        self.measure = 'axis_measure'
        self.distance = 'nearest_distance'
        self.output = 'valley_bottom_mask'
        self.slope = 'slope'

        self.swath_length = 200.0
        # self.distance_min = 20.0
        self.height_max = 20.0
        self.thresholds = [
            # drainage area km², distance min, distance max, max height (depth), max slope (%)
            # ValleyBottomThreshold(0, 20.0, 100.0, 1.0, 25.0),
            # ValleyBottomThreshold(30, 20.0, 400.0, 2.0, 20.0),
            # ValleyBottomThreshold(300, 50.0, 1500.0, 4.0, 15.0),
            # ValleyBottomThreshold(1000, 100.0, 2500.0, 4.0, 7.0)
            ValleyBottomThreshold(0, 20.0, 100.0, 2.0, 10.0),
            ValleyBottomThreshold(30, 20.0, 400.0, 4.0, 7.0),
            ValleyBottomThreshold(300, 20.0, 1500.0, 5.0, 5.0),
            ValleyBottomThreshold(1000, 20.0, 2500.0, 6.0, 3.5)
        ]
        self.patch_min_pixels = 100

def calculate_slope(dem_raster, nodata=999.0):
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

def calculate_swaths(measure_raster, swath_length):
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

def SwathDrainageTile(row, col, params, **kwargs):

    axis_raster = params.axis.tilename(row=row, col=col, **kwargs)
    drainage_raster = params.drainage.tilename(row=row, col=col, **kwargs)
    measure_raster = params.measure.tilename(row=row, col=col, **kwargs)

    swaths, measures = calculate_swaths(measure_raster, params.swath_length)

    with rio.open(drainage_raster) as ds:
        drainage = ds.read(1)

    with rio.open(axis_raster) as ds:
        axis = ds.read(1)
        axis_nodata = ds.nodata

    result = dict()

    for ax in np.unique(axis):

        if ax == axis_nodata:
            continue

        for sw in np.unique(swaths[axis == ax]):

            if sw == 0:
                continue

            sw_measure = measures[sw-1]
            sw_mask = (axis == ax) & (swaths == sw)

            # compute swath drainage in separate loop
            # in order to avoid tile boundary problems
            sw_drainage = np.max(drainage[sw_mask])

            result[ax, sw_measure] = sw_drainage

    return result

def ValleyBottomMaskTile(row, col, params, drainage, **kwargs):

    dem_raster = params.dem.tilename(row=row, col=col, **kwargs)
    # drainage_raster = params.drainage.tilename(row=row, col=col, **kwargs)
    height_raster = params.height.tilename(row=row, col=col, **kwargs)
    axis_raster = params.axis.tilename(row=row, col=col, **kwargs)
    measure_raster = params.measure.tilename(row=row, col=col, **kwargs)
    distance_raster = params.distance.tilename(row=row, col=col, **kwargs)

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

    def resolve_thresholds(sw_drainage):

        thresholds = params.thresholds[0]

        for t in params.thresholds[1:]:
            
            if sw_drainage < t.drainage_min:
                break

            thresholds = t

        return thresholds

    with rio.open(height_raster) as ds:
    
        height = ds.read(1)
        out = np.full_like(height, MASK_EXTERIOR, dtype='uint8')

        for ax in np.unique(axis):

            if ax == axis_nodata:
                continue

            for sw in np.unique(swaths[axis == ax]):

                if sw == 0:
                    continue

                sw_measure = measures[sw-1]
                sw_mask = (axis == ax) & (swaths == sw)

                # TODO compute swath drainage in separate loop
                #      in order to avoid tile boundary problems
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

                if sw == 0:
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

        profile.update(nodata=999.0, compress='deflate')
        with rio.open(params.slope.tilename(row=row, col=col, **kwargs), 'w', **profile) as dst:
            dst.write(slope, 1)

        profile.update(dtype='uint8', nodata=0, compress='deflate')
        with rio.open(output, 'w', **profile) as dst:
            dst.write(out, 1)

def SwathDrainage(params, processes=1, **kwargs):

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
                
                drainage.update({k: max(drainage[k], t_drainage[k]) for k in t_drainage.keys() & drainage.keys()})
                drainage.update({k: t_drainage[k] for k in t_drainage.keys() - drainage.keys()})

        return drainage

def ValleyBottomMask(params, drainage, processes=1, **kwargs):
    """
    Creates a raster buffer with distance buffer_width pixels
    around data pixels and crop out data outside of the resulting buffer
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
                ValleyBottomMaskTile,
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
