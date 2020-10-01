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

import os
from collections import namedtuple
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape
import xarray as xr

from ..tileio import as_window
from ..cli import starcall
from ..config import config
from ..metadata import set_metadata

DatasetParameter = namedtuple('DatasetParameter', [
    # 'landcover', # landcover, ax_continuity
    'swath_raster', # ax_dgo
    'swath_polygons', # ax_dgo_vector
    'axis_distance', # ax_axis_distance
    'drainage_distance', # ax_nearest_distance, ax_talweg_distance
    'drainage_height',
    'output', # ax_swath_valleybottom_npz
])

def ValleyBottomSwath(
        axis,
        gid,
        bounds,
        datasets,
        step=10.0,
        valley_bottom_mask=None,
        **kwargs):
    """
    Calculate land cover swath profile for longitudinal unit (axis, gid)
    """

    def _rasterfile(name):
        return config.tileset().filename(name, axis=axis, **kwargs)

    # landcover_raster = _rasterfile(datasets.landcover)
    swath_raster = _rasterfile(datasets.swath_raster)
    axis_distance_raster = _rasterfile(datasets.axis_distance)
    nearest_distance_raster = _rasterfile(datasets.drainage_distance)
    hand_raster = _rasterfile(datasets.drainage_height)
    valleybottom_raster = _rasterfile('ax_valley_mask_refined')
    mask_raster = _rasterfile(valley_bottom_mask)

    with rio.open(hand_raster) as ds:
        window = as_window(bounds, ds.transform)
        hand = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(axis_distance_raster) as ds:
        window = as_window(bounds, ds.transform)
        axis_distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(nearest_distance_raster) as ds:
        window = as_window(bounds, ds.transform)
        nearest_distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(valleybottom_raster) as ds:
        window = as_window(bounds, ds.transform)
        valleybottom = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(swath_raster) as ds:
        window = as_window(bounds, ds.transform)
        swath_mask = (ds.read(1, window=window, boundless=True, fill_value=ds.nodata) == gid)

    with rio.open(mask_raster) as ds:

        window = as_window(bounds, ds.transform)
        mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        mask = (mask == 0) & swath_mask

    # with rio.open(landcover_raster) as ds:

    #     window = as_window(bounds, ds.transform)
    #     landcover = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        assert hand.shape == mask.shape
        assert axis_distance.shape == mask.shape
        assert nearest_distance.shape == mask.shape
        # assert mask.shape == landcover.shape

        if np.sum(mask) == 0:

            click.secho('No data for swath (%d, %d)' % (axis, gid), fg='yellow')
            values = dict(
                x=np.zeros(0, dtype='float32'),
                density=np.zeros(0, dtype='float32'),
                classes=np.zeros(0, dtype='uint32'),
                swath=np.zeros((0, 0), dtype='float32')
            )
            return gid, values

        xmin = np.min(axis_distance[mask])
        xmax = np.max(axis_distance[mask])

        if (xmax - xmin) < 2000.0:
            xbins = np.arange(xmin, xmax + step, step)
        else:
            xbins = np.linspace(xmin, xmax, 200)

        x = 0.5*(xbins[1:] + xbins[:-1])
        axis_distance_binned = np.digitize(axis_distance, xbins)

        # Valley bottom area at height h

        heights = np.arange(5.0, 15.5, 0.5)
        valley_bottom_area_h = np.zeros(len(heights), dtype='uint32')

        for k, height_k in enumerate(heights):
            valley_bottom_area_h[k] = np.sum((mask == 1) & (hand <= height_k))

        # Valley bottom area on each side of talweg

        valley_bottom_area_lr = np.zeros(2, dtype='uint32')
        valley_bottom_area_lr[0] = np.sum(nearest_distance[mask] >= 0) # left side
        valley_bottom_area_lr[1] = np.sum(nearest_distance[mask] < 0) # right side

        # Valley bottom swath

        valley_bottom_swath = np.zeros(len(x), dtype='uint32')

        for i in range(1, len(xbins)):

            mask0 = mask & (axis_distance_binned == i)
            data = (valleybottom == 0)
            valley_bottom_swath[i-1] = np.sum(data[mask0])

        values = dict(
            x=x,
            valley_bottom_area_h=valley_bottom_area_h,
            valley_bottom_area_lr=valley_bottom_area_lr,
            valley_bottom_swath=valley_bottom_swath
        )

        return gid, values

def ValleyBottomSwathProfile(axis, processes=1, **kwargs):
    """
    Valley bottom swath metrics :

    - valley_bottom_area_h =
        valley bottom pixel area at height h above talweg,
        as defined by ax_drainage_height (HAND)

    - valley_bottom_area_lr =
        valley bottom pixel area on each side of talweg
        as defined by ax_drainage_distance

    - valley_bottom_swath =
         pixel area of slice at distance x from reference axis,
         as defined by ax_axis_distance

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

    swath_raster: str, logical name

        Swath unit raster,
        defaults to `ax_swaths`

    swath_polygons: str, logical name

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

    valley_bottom_mask: str, logical name

        Valley bottom mask

    output: str, logical name

        Output file for each generated swath data,
        defaults to `ax_swath_valleybottom_npz`

    Other keywords are passed to dataset filename templates.
    """

    defaults = dict(
        # landcover='ax_continuity',
        swath_raster='ax_valley_swaths',
        swath_polygons='ax_valley_swaths_polygons',
        axis_distance='ax_axis_distance',
        drainage_distance='ax_nearest_distance',
        drainage_height='ax_nearest_height',
        output='ax_swath_valleybottom_npz'
    )

    defaults.update({k: kwargs[k] for k in kwargs.keys() & defaults.keys()})
    datasets = DatasetParameter(**defaults)
    kwargs = {k: kwargs[k] for k in kwargs.keys() - defaults.keys()}

    swath_shapefile = config.filename(datasets.swath_polygons, axis=axis, **kwargs)
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
                    ValleyBottomSwath,
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

def ExportValleyBottomSwathsToNetCDF(axis, **kwargs):
    """
    Reads back landcover swath profile from disk,
    and bundles everyting into one netcdf file.
    """

    defaults = dict(
        # landcover='ax_continuity',
        swath_raster='ax_valley_swaths',
        swath_polygons='ax_valley_swaths_polygons',
        axis_distance='ax_axis_distance',
        drainage_distance='ax_nearest_distance',
        drainage_height='ax_nearest_height',
        output='ax_swath_valleybottom_npz'
    )

    defaults.update({k: kwargs[k] for k in kwargs.keys() & defaults.keys()})
    datasets = DatasetParameter(**defaults)
    kwargs = {k: kwargs[k] for k in kwargs.keys() - defaults.keys()}

    swath_bounds = config.filename('ax_valley_swaths_bounds', axis=axis)

    defs = xr.open_dataset(swath_bounds)
    defs = defs.load().sortby('measure')
    length = defs['swath'].shape[0]
    nclasses = 9

    heights = np.arange(5.0, 15.5, 0.5)

    # x=x,
    # valley_bottom_area_h=valley_bottom_area_h,
    # valley_bottom_area_lr=valley_bottom_area_lr,
    # valley_bottom_swath=valley_bottom_swath

    swids = np.zeros(length, dtype='uint32')
    measures = np.zeros(length, dtype='float32')

    sw_measure = np.zeros(0, dtype='float32')
    sw_distance = np.zeros(0, dtype='float32')
    valley_bottom_area_h = np.zeros((length, len(heights)), dtype='uint32')
    valley_bottom_area_lr = np.zeros((length, 2), dtype='uint32')
    sw_valley_bottom = np.zeros(0, dtype='uint32')

    with click.progressbar(defs['swath'].values) as iterator:
        for k, swid in enumerate(iterator):

            coordm = defs['measure'].sel(swath=swid).values
            swathfile = config.filename(datasets.output, axis=axis, gid=swid, **kwargs)

            swids[k] = swid
            measures[k] = coordm

            if not os.path.exists(swathfile):
                continue

            data = np.load(swathfile, allow_pickle=True)

            valley_bottom_area_h[k] = data['valley_bottom_area_h']
            valley_bottom_area_lr[k] = data['valley_bottom_area_lr']

            sw_measure = np.concatenate([
                sw_measure,
                np.full_like(data['x'], coordm, dtype='float32')
            ])

            sw_distance = np.concatenate([
                sw_distance,
                np.float32(data['x'])
            ])

            sw_valley_bottom = np.concatenate([
                sw_valley_bottom,
                data['valley_bottom_swath']
            ])

    dataset = xr.Dataset({
        'valley_bottom_area_h': (('measure', 'height'), valley_bottom_area_h),
        'valley_bottom_area_lr': (('measure', 'side'), valley_bottom_area_lr),
        'sw_measure': ('profile', sw_measure),
        'sw_axis_distance': ('profile', sw_distance),
        'sw_valley_bottom': ('profile', sw_valley_bottom)
    }, coords={
        'axis': axis,
        'measure': measures,
        'swath': ('measure', swids),
        'height': np.float32(heights),
        'side': [
            'left',
            'right'
        ]
    })

    # dataset.set_index(profile=['sw_measure', 'sw_axis_distance'])

    set_metadata(dataset, 'swath_valleybottom')

    output = config.filename('swath_valleybottom', axis=axis, **kwargs)
    dataset.to_netcdf(output, 'w')

    return dataset
