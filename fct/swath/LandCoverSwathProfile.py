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
    'landcover', # landcover, ax_continuity
    'swath_raster', # ax_dgo
    'swath_polygons', # ax_dgo_vector
    'axis_distance', # ax_axis_distance
    'drainage_distance', # ax_nearest_distance, ax_talweg_distance
    'output', # ax_swath_landcover_npz
])

def LandCoverSwath(
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

    landcover_raster = _rasterfile(datasets.landcover)
    swath_raster = _rasterfile(datasets.swath_raster)
    axis_distance_raster = _rasterfile(datasets.axis_distance)
    nearest_distance_raster = _rasterfile(datasets.drainage_distance)
    # hand_raster = _rasterfile(datasets.height)
    # valleybottom_raster = _rasterfile('ax_valley_mask_refined')

    # with rio.open(hand_raster) as ds:
    #     window = as_window(bounds, ds.transform)
    #     hand = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(axis_distance_raster) as ds:
        window = as_window(bounds, ds.transform)
        axis_distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(nearest_distance_raster) as ds:
        window = as_window(bounds, ds.transform)
        nearest_distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    # with rio.open(valleybottom_raster) as ds:
    #     window = as_window(bounds, ds.transform)
    #     valleybottom = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(swath_raster) as ds:
        window = as_window(bounds, ds.transform)
        swath_mask = (ds.read(1, window=window, boundless=True, fill_value=ds.nodata) == gid)

    if valley_bottom_mask:

        mask_raster = _rasterfile(valley_bottom_mask)

        with rio.open(mask_raster) as ds:
            window = as_window(bounds, ds.transform)
            mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
            mask = (mask == 0) & swath_mask

    else:

        mask = swath_mask

    with rio.open(landcover_raster) as ds:

        window = as_window(bounds, ds.transform)
        landcover = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        try:

            # assert hand.shape == landcover.shape
            assert axis_distance.shape == landcover.shape
            assert nearest_distance.shape == landcover.shape
            assert mask.shape == landcover.shape

        except AssertionError:

            # print('landcover', landcover.shape)
            # print('axis_distance', axis_distance.shape)
            # print('nearest_distance', nearest_distance.shape)
            # print('mask', mask.shape)

            click.secho('No data for swath (%d, %d)' % (axis, gid), fg='yellow')
            values = dict(
                x=np.zeros(0, dtype='float32'),
                density=np.zeros(0, dtype='float32'),
                landcover_classes=np.zeros(0, dtype='uint32'),
                landcover_swath=np.zeros((0, 0), dtype='float32')
            )
            return gid, values

        if np.sum(mask) == 0:

            click.secho('No data for swath (%d, %d)' % (axis, gid), fg='yellow')
            values = dict(
                x=np.zeros(0, dtype='float32'),
                density=np.zeros(0, dtype='float32'),
                landcover_classes=np.zeros(0, dtype='uint32'),
                landcover_swath=np.zeros((0, 0), dtype='float32')
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
        # nearest_distance_binned = np.digitize(nearest_distance, xbins)

        # Profile density

        density = np.zeros(x.shape[0], dtype='uint32')

        # Land cover classes count

        classes = np.unique(landcover[mask])
        landcover_swath = np.zeros((len(x), len(classes), 2), dtype='uint32')

        for i in range(1, len(xbins)):

            mask_i = mask & (axis_distance_binned == i)
            # mask1 = mask & (nearest_distance_binned == i)

            density[i-1] = np.sum(mask_i)
            # density[i-1, 1] = np.sum(mask1)

            for k, value in enumerate(classes):

                class_k = mask_i & (landcover == value)

                if np.sum(class_k) > 0:

                    # left
                    landcover_swath[i-1, k, 0] = np.sum(nearest_distance[class_k] >= 0)
                    # right
                    landcover_swath[i-1, k, 1] = np.sum(nearest_distance[class_k] < 0)

                # if density[i-1, 1] > 0:
                #     landcover_swath[i-1, k, 1] = np.sum(data[mask1])

        values = dict(
            x=x,
            density=density,
            landcover_classes=classes,
            landcover_swath=landcover_swath
        )

        return gid, values

def LandCoverSwathProfile(axis, processes=1, **kwargs):
    """
    Generate landcover swath for every longitudinal unit

    @api    fct-swath:profile-landcover

    @input  landcover: ax_continuity
    @input  swath_raster: ax_valley_swaths
    @input  swath_polygons: ax_valley_swaths_polygons
    @input  axis_distance: ax_axis_distance
    @input  drainage_distance: ax_nearest_distance
    @input  drainage_height: ax_nearest_height
    @input  mask: ax_valley_mask_refined

    @param  min_slice_width: 10.0
    @param  max_slice_count: 200
    @param  nclasses: 9

    @output swath_landcover_npz: ax_swath_landcover_npz

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
        defaults to `ax_swath_landcover_npz`

    Other keywords are passed to dataset filename templates.
    """

    defaults = dict(
        landcover='ax_continuity',
        swath_raster='ax_swaths_refaxis',
        swath_polygons='ax_swaths_refaxis_polygons',
        axis_distance='ax_axis_distance',
        drainage_distance='ax_nearest_distance',
        output='ax_swath_landcover_npz'
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
                    LandCoverSwath,
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

def ExportLandcoverSwathsToNetCDF(axis, **kwargs):
    """
    Reads landcover swath profile back from disk,
    and bundles everyting into one netcdf file.

    @api    fct-swath:export-landcover

    @input  landcover: ax_continuity
    @input  swath_bounds: ax_valley_swaths_bounds
    @input  swath_landcover_npz: ax_swath_landcover_npz

    @output swath_landcover: swath_landcover
    """

    defaults = dict(
        landcover='ax_continuity',
        swath_raster='ax_swaths_refaxis',
        swath_polygons='ax_swaths_refaxis_polygons',
        axis_distance='ax_axis_distance',
        drainage_distance='ax_nearest_distance',
        output='ax_swath_landcover_npz'
    )

    defaults.update({k: kwargs[k] for k in kwargs.keys() & defaults.keys()})
    datasets = DatasetParameter(**defaults)
    kwargs = {k: kwargs[k] for k in kwargs.keys() - defaults.keys()}

    swath_bounds = config.filename('ax_swaths_refaxis_bounds', axis=axis)

    defs = xr.open_dataset(swath_bounds)
    defs = defs.load().sortby('measure')
    length = defs['swath'].shape[0]
    nclasses = 9

    # x=x,
    # density=density,
    # landcover_classes=classes,
    # landcover_swath=landcover_swath

    swids = np.zeros(length, dtype='uint32')
    measures = np.zeros(length, dtype='float32')

    sw_measure = np.zeros(0, dtype='float32')
    sw_distance = np.zeros(0, dtype='float32')
    sw_density = np.zeros(0, dtype='uint32')
    sw_landcover = np.zeros((0, nclasses, 2), dtype='uint32')

    with click.progressbar(defs['swath'].values) as iterator:
        for k, swid in enumerate(iterator):

            coordm = defs['measure'].sel(swath=swid).values
            swathfile = config.filename(datasets.output, axis=axis, gid=swid, **kwargs)

            swids[k] = swid
            measures[k] = coordm

            if not os.path.exists(swathfile):
                continue

            data = np.load(swathfile, allow_pickle=True)

            classes = data['landcover_classes']

            sw_measure = np.concatenate([
                sw_measure,
                np.full_like(data['density'], coordm, dtype='float32')
            ])
            sw_distance = np.concatenate([
                sw_distance,
                np.float32(data['x'])
            ])
            sw_density = np.concatenate([
                sw_density,
                data['density']
            ])

            landcover_swath = data['landcover_swath']
            expanded_landcover_swath = np.zeros((landcover_swath.shape[0], nclasses, 2), dtype='uint32')

            for k, klass in enumerate(classes):

                if klass == 255:
                    continue

                expanded_landcover_swath[:, klass, :] = landcover_swath[:, k, :]

            sw_landcover = np.concatenate([
                sw_landcover,
                expanded_landcover_swath
            ])

    dataset = xr.Dataset({
        'sw_measure': ('profile', sw_measure),
        'sw_axis_distance': ('profile', sw_distance),
        'sw_density': ('profile', sw_density),
        'sw_landcover': (('profile', 'landcover', 'side'), sw_landcover)
    }, coords={
        'axis': axis,
        'measure': measures,
        'swath': ('measure', swids),
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

    # dataset.set_index(profile=['sw_measure', 'sw_axis_distance'])

    set_metadata(dataset, 'swath_landcover')

    dataset.attrs['source'] = config.basename(
        datasets.landcover,
        axis=axis,
        **kwargs)

    output = config.filename('swath_landcover', axis=axis, **kwargs)
    dataset.to_netcdf(output, 'w')

    return dataset

def ExportContinuitySwathsToNetCDF(axis, **kwargs):
    """
    Reads landcover swath profile back from disk,
    and bundles everyting into one netcdf file.

    @api    fct-swath:export-landcover

    @input  landcover: ax_continuity
    @input  swath_bounds: ax_valley_swaths_bounds
    @input  swath_landcover_npz: ax_swath_landcover_npz

    @output swath_landcover: swath_landcover
    """

    defaults = dict(
        landcover='ax_continuity_variant_remapped',
        swath_raster='ax_swaths_refaxis',
        swath_polygons='ax_swaths_refaxis_polygons',
        axis_distance='ax_axis_distance',
        drainage_distance='ax_nearest_distance',
        output='ax_swath_landcover_npz'
    )

    defaults.update({k: kwargs[k] for k in kwargs.keys() & defaults.keys()})
    datasets = DatasetParameter(**defaults)
    kwargs = {k: kwargs[k] for k in kwargs.keys() - defaults.keys()}

    swath_bounds = config.filename('ax_swaths_refaxis_bounds', axis=axis)

    defs = xr.open_dataset(swath_bounds)
    defs = defs.load().sortby('measure')
    length = defs['swath'].shape[0]
    nclasses = 6

    # x=x,
    # density=density,
    # landcover_classes=classes,
    # landcover_swath=landcover_swath

    swids = np.zeros(length, dtype='uint32')
    measures = np.zeros(length, dtype='float32')

    sw_measure = np.zeros(0, dtype='float32')
    sw_distance = np.zeros(0, dtype='float32')
    sw_density = np.zeros(0, dtype='uint32')
    sw_continuity = np.zeros((0, nclasses, 2), dtype='uint32')

    klass_labels = [1, 10, 20, 30, 40, 50]

    with click.progressbar(defs['swath'].values) as iterator:
        for i, swid in enumerate(iterator):

            coordm = defs['measure'].sel(swath=swid).values
            swathfile = config.filename(datasets.output, axis=axis, gid=swid, **kwargs)

            swids[i] = swid
            measures[i] = coordm

            if not os.path.exists(swathfile):
                continue

            data = np.load(swathfile, allow_pickle=True)

            classes = data['landcover_classes']

            sw_measure = np.concatenate([
                sw_measure,
                np.full_like(data['density'], coordm, dtype='float32')
            ])
            sw_distance = np.concatenate([
                sw_distance,
                np.float32(data['x'])
            ])
            sw_density = np.concatenate([
                sw_density,
                data['density']
            ])

            landcover_swath = data['landcover_swath']
            expanded_landcover_swath = np.zeros(
                (landcover_swath.shape[0], nclasses, 2),
                dtype='uint32'
            )

            for k, klass in enumerate(classes):

                if klass == 255:
                    continue

                klass_index = klass_labels.index(klass)

                expanded_landcover_swath[:, klass_index, :] = landcover_swath[:, k, :]

            sw_continuity = np.concatenate([
                sw_continuity,
                expanded_landcover_swath
            ])

    dataset = xr.Dataset({
        'sw_measure': ('profile', sw_measure),
        'sw_axis_distance': ('profile', sw_distance),
        'sw_density': ('profile', sw_density),
        'sw_continuity': (('profile', 'continuity', 'side'), sw_continuity)
    }, coords={
        'axis': axis,
        'measure': measures,
        'swath': ('measure', swids),
        # 'landcover': [
        #     'Water Channel',
        #     'Gravel Bars',
        #     'Natural Open',
        #     'Forest',
        #     'Grassland',
        #     'Crops',
        #     'Diffuse Urban',
        #     'Dense Urban',
        #     'Infrastructures'
        # ],
        'continuity': [
            'Active channel',
            'Riparian corridor',
            'Semi-natural',
            'Reversible',
            'Disconnected',
            'Built environment'
        ],
        'side': [
            'left',
            'right'
        ]
    })

    # dataset.set_index(profile=['sw_measure', 'sw_axis_distance'])

    set_metadata(dataset, 'swath_continuity')

    dataset.attrs['source'] = config.basename(
        datasets.landcover,
        axis=axis,
        **kwargs)

    output = config.filename('swath_continuity', axis=axis, **kwargs)
    dataset.to_netcdf(output, 'w')

    return dataset
