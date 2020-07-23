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

def UnitLandCoverSwath(
        axis,
        gid,
        bounds,
        dataset='continuity',
        step=10.0,
        maxz=20.0):
    """
    Calculate land cover swath profile for longitudinal unit (axis, gid)
    """

    dgo_raster = config.filename('ax_dgo', axis=axis)
    axis_distance_raster = config.filename('ax_axis_distance', axis=axis)
    nearest_distance_raster = config.filename('ax_nearest_distance', axis=axis)
    hand_raster = config.filename('ax_relative_elevation', axis=axis)

    if dataset == 'landcover':
        landcover_raster = config.filename('landcover')
    elif dataset == 'continuity':
        landcover_raster = config.filename('ax_continuity', axis=axis)
    else:
        raise ValueError('incorrect parameter dataset: %s' % dataset)

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

        with rio.open(dgo_raster) as ds4:
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
            return axis, gid, values

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

def LandCoverSwath(axis, dataset='continuity', processes=1, **kwargs):
    """
    Generate landcover swath for every longitudinal unit
    defined by procedure fct.metrics.SpatialReferencing.SpatialReference

    Parameters
    ----------

    axis: int

        Axis identifier

    dataset: str

        possible values :
        - `landcover`: swath from global landcover raster
        - `continuity`: swath from axis landcover continuity map

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

        Exclude data with height above nearest drainage (hand) > maxz,
        defaults to 20.0 m
    """

    if dataset not in ('landcover', 'continuity'):
        click.secho('Unknown swath dataset %s' % dataset, fg='yellow')
        return

    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)

    kwargs.update(dataset=dataset)
    profiles = dict()

    with fiona.open(dgo_shapefile) as fs:
        length = len(fs)

    def arguments():

        with fiona.open(dgo_shapefile) as fs:
            for feature in fs:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                geometry = asShape(feature['geometry'])

                profiles[axis, gid] = [axis, gid, measure]

                yield (
                    UnitLandCoverSwath,
                    axis,
                    gid,
                    geometry.bounds,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length) as iterator:
            for gid, values in iterator:

                profile = profiles[axis, gid]

                output = config.filename(
                    'ax_swath_landcover',
                    axis=axis,
                    gid=gid,
                    subset=dataset.upper())

                np.savez(
                    output,
                    profile=profile,
                    **values)
