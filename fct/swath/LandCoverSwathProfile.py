# coding: utf-8

"""
LandCover Swath Profile

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
from multiprocessing import Pool
import numpy as np

import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape
import click

from ..tileio import as_window
from ..cli import starcall
from ..config import config

DatasetParameter = namedtuple('DatasetParameter', [
    'landcover', # landcover, ax_continuity
    'swaths', # ax_dgo
    'swath_features', # ax_dgo_vector
    'axis_distance', # ax_axis_distance
    'drainage_distance', # ax_nearest_distance, ax_talweg_distance
    'height', # ax_nearest_height
    'output', # ax_swath_landcover
])

def UnitLandCoverSwath(
        axis,
        gid,
        bounds,
        datasets,
        step=10.0,
        maxz=20.0,
        **kwargs):
    """
    Calculate land cover swath profile for longitudinal unit (axis, gid)
    """

    def _rasterfile(name):
        return config.tileset().filename(name, axis=axis, **kwargs)

    landcover_raster = _rasterfile(datasets.landcover)
    swath_raster = _rasterfile(datasets.swaths)
    axis_distance_raster = _rasterfile(datasets.axis_distance)
    nearest_distance_raster = _rasterfile(datasets.drainage_distance)
    hand_raster = _rasterfile(datasets.height)

    with rio.open(landcover_raster) as ds:

        window = as_window(bounds, ds.transform)
        landcover = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(hand_raster) as ds1:
            hand = ds1.read(1, window=window, boundless=True, fill_value=ds1.nodata)

        with rio.open(axis_distance_raster) as ds2:
            window2 = as_window(bounds, ds2.transform)
            axis_distance = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)

        with rio.open(nearest_distance_raster) as ds3:
            window3 = as_window(bounds, ds3.transform)
            nearest_distance = ds3.read(1, window=window3, boundless=True, fill_value=ds3.nodata)

        with rio.open(swath_raster) as ds4:
            window4 = as_window(bounds, ds4.transform)
            mask = (ds4.read(1, window=window4, boundless=True, fill_value=ds4.nodata) == gid)

        assert(all([
            hand.shape == landcover.shape,
            axis_distance.shape == landcover.shape,
            nearest_distance.shape == landcover.shape,
            mask.shape == landcover.shape
        ]))

        mask = mask & (hand < maxz)
        del hand

        if np.sum(mask) == 0:

            click.secho('No data for swath (%d, %d)' % (axis, gid), fg='yellow')
            values = dict(
                x=np.zeros(0, dtype='float32'),
                density=np.zeros(0, dtype='float32'),
                classes=np.zeros(0, dtype='uint32'),
                swath=np.zeros((0, 0), dtype='float32')
            )
            return gid, values

        min_distance = min(np.min(axis_distance[mask]), np.min(nearest_distance[mask]))
        max_distance = max(np.max(axis_distance[mask]), np.max(nearest_distance[mask]))
        bins = np.arange(min_distance, max_distance + step, step)

        x = 0.5*(bins[1:] + bins[:-1])
        axis_distance_binned = np.digitize(axis_distance, bins)
        nearest_distance_binned = np.digitize(nearest_distance, bins)

        # Profile density

        density = np.zeros((x.shape[0], 2), dtype='uint32')

        # Land cover classes count

        classes = np.unique(landcover[mask])
        swath = np.zeros((len(x), len(classes), 2), dtype='uint16')

        for i in range(1, len(bins)):

            mask0 = mask & (axis_distance_binned == i)
            mask1 = mask & (nearest_distance_binned == i)

            density[i-1, 0] = np.sum(mask0)
            density[i-1, 1] = np.sum(mask1)

            for k, value in enumerate(classes):

                data = (landcover == value)

                if density[i-1, 0] > 0:
                    swath[i-1, k, 0] = np.sum(data[mask0])

                if density[i-1, 1] > 0:
                    swath[i-1, k, 1] = np.sum(data[mask1])

        values = dict(
            x=x,
            density=density,
            classes=classes,
            swath=swath
        )

        return gid, values

def LandCoverSwath(axis, processes=1, **kwargs):
    """
    Generate landcover swath for every longitudinal unit
    defined by procedure fct.metrics.SpatialReferencing.SpatialReference

    Parameters
    ----------

    axis: int

        Axis identifier

    processes: int

        Number of parallel processes to execute
        (defaults to one)

    Keyword arguments
    -----------------

    step: float

        Width step,
        ie. distance between swath measures
        in the direction perpendicular to stream or reference axis

    maxz: float

        Truncate data with height above nearest drainage (HAND) > maxz,
        defaults to 20.0 m

    landcover: str, logical name

        Landcover raster dataset,
        defaults to `ax_continuity` (continuous landcover buffer swath)
        Other values: `landcover` (total landcover swath)

    swaths: str, logical name

        Swath unit raster,
        defaults to `ax_swaths`

    swath_features: str, logical name

        Shapefile of swath entities,
        defaults to `ax_swath_features`

    axis_distance: str, logical name

        Signed distance to reference axis,
        generates swath axis perpendicular to reference axis.
        Defaults to `ax_axis_distance`

    drainage_distance: str, logical name

        Signed distance to nearest drainage,
        distinguishes between left bank and right bank.
        Defaults to `ax_talweg_distance`

    height: str, logical name

        Height above nearest drainage raster (HAND),
        defaults to `ax_nearest_height`

    output: str, logical name

        Output file for each generated swath data,
        defaults to `ax_swath_landcover`

    Other keywords are passed to dataset filename templates.
    """

    defaults = dict(
        landcover='ax_continuity',
        swaths='ax_valley_swaths',
        swath_features='ax_valley_swaths_polygons',
        axis_distance='ax_axis_distance',
        drainage_distance='ax_nearest_distance',
        height='ax_nearest_height',
        output='ax_swath_landcover'
    )

    defaults.update({k: kwargs[k] for k in kwargs.keys() & defaults.keys()})
    datasets = DatasetParameter(**defaults)
    kwargs = {k: kwargs[k] for k in kwargs.keys() - defaults.keys()}

    swath_shapefile = config.filename(datasets.swath_features, axis=axis, **kwargs)
    profiles = dict()

    with fiona.open(swath_shapefile) as fs:
        length = len(fs)

    def arguments():

        with fiona.open(swath_shapefile) as fs:
            for feature in fs:

                if feature['properties']['VALUE'] == 0:
                    continue

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                geometry = asShape(feature['geometry'])

                profiles[axis, gid] = [axis, gid, measure]

                yield (
                    UnitLandCoverSwath,
                    axis,
                    gid,
                    geometry.bounds,
                    datasets,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length) as iterator:
            for gid, values in iterator:

                profile = profiles[axis, gid]

                output = config.filename(
                    datasets.output,
                    axis=axis,
                    gid=gid,
                    **kwargs)

                np.savez(
                    output,
                    profile=profile,
                    **values)
