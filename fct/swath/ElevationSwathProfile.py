# coding: utf-8

"""
Elevation Swath Profile

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
from multiprocessing import Pool

import numpy as np
import xarray as xr
import click

import rasterio as rio
from rasterio import features
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..config import config
from ..tileio import as_window
from ..ransac import LinearModel, ransac
from ..cli import starcall
from ..metadata import set_metadata

def TileCropInvalidRegions(axis, tile):
    """
    DOCME
    """

    invalid_shapefile = config.filename('ax_swaths_refaxis_polygons', axis=axis)
    regions_raster = config.tileset().tilename(
        'ax_swaths_refaxis',
        axis=axis,
        row=tile.row,
        col=tile.col)

    if not os.path.exists(regions_raster):
        return

    with rio.open(regions_raster) as ds:

        data = ds.read(1)
        transform = ds.transform
        nodata = ds.nodata
        profile = ds.profile.copy()

    with fiona.open(invalid_shapefile) as fs:

        def accept(feature):
            return all([
                feature['properties']['AXIS'] == axis,
                feature['properties']['VALUE'] == 0
            ])

        geometries = [
            f['geometry'] for f in fs.filter(bbox=tile.bounds)
            if accept(f)
        ]

        if geometries:

            mask = features.rasterize(
                geometries,
                out_shape=data.shape,
                transform=transform,
                fill=0,
                default_value=1,
                dtype='uint8')

            data[mask == 1] = nodata

            with rio.open(regions_raster, 'w', **profile) as dst:
                dst.write(data, 1)

def CropInvalidRegions(axis, processes=1, **kwargs):

    # tileindex = config.tileset().tileindex

    tileset = config.tileset()

    def arguments():

        for tile in tileset.tiles():
            yield (
                TileCropInvalidRegions,
                axis,
                tile,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())
        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass

def _UnitSwathProfile(axis, gid, bounds):
    """
    Calculate Elevation Swath Profile for Valley Unit (axis, gid)
    """

    # TODO split valley floor fit code into separate module

    tileset = config.tileset()
    swath_raster = tileset.filename('ax_swaths_refaxis', axis=axis)
    measure_raster = tileset.filename('ax_axis_measure', axis=axis)
    distance_raster = tileset.filename('ax_axis_distance', axis=axis)
    talweg_distance_raster = tileset.filename('ax_nearest_distance', axis=axis)
    elevation_raster = tileset.filename('dem')
    hand_raster = tileset.filename('ax_nearest_height', axis=axis)
    # hand_raster = tileset.filename('ax_valley_mask', axis=axis)

    with rio.open(elevation_raster) as ds:
        window = as_window(bounds, ds.transform)
        elevations = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        elevation_nodata = ds.nodata

    with rio.open(measure_raster) as ds:
        window = as_window(bounds, ds.transform)
        measure = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(hand_raster) as ds:
        window = as_window(bounds, ds.transform)
        hand = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        hand_nodata = ds.nodata

    with rio.open(swath_raster) as ds:
        window = as_window(bounds, ds.transform)
        mask = (ds.read(1, window=window, boundless=True, fill_value=ds.nodata) == gid)

    with rio.open(talweg_distance_raster) as ds:
        window = as_window(bounds, ds.transform)
        talweg_distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)
        mask1k = (talweg_distance >= -1000) & (talweg_distance <= 1000)
        del talweg_distance

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        assert elevations.shape == distance.shape
        assert measure.shape == distance.shape
        assert hand.shape == distance.shape
        assert mask.shape == distance.shape
        assert mask1k.shape == distance.shape

        mask = mask & (elevations != elevation_nodata)

        # Fit valley floor

        slope = np.nan
        z0 = np.nan
        with_valley_floor_estimates = False

        def fit_valley_floor(fit_mask=None, error_threshold=1.0, iterations=100):

            nonlocal slope, z0

            if fit_mask is None:
                mask0 = mask & mask1k
            else:
                mask0 = mask & mask1k & fit_mask

            size = elevations.shape[0]*elevations.shape[1]
            matrix = np.stack([
                measure.reshape(size),
                np.ones(size, dtype='float32'),
                elevations.reshape(size)
            ]).T
            matrix = matrix[mask0.reshape(size), :]
            samples = matrix.shape[0] // 10
            model = LinearModel([0, 1], [2])

            (slope, z0), _, _ = ransac(matrix, model, samples, iterations, error_threshold, 2*samples)

            relative = elevations - (z0 + slope*measure)

            return relative

        try:

            relative = fit_valley_floor()
            with_valley_floor_estimates = True

        except RuntimeError:

            try:

                relative = fit_valley_floor(fit_mask=(hand <= 10.0))
                with_valley_floor_estimates = True

            except RuntimeError:

                with_valley_floor_estimates = False

        # Valley Bottom pixel area (to calculate areal width)

        # heights = np.arange(5.0, 15.5, 0.5)
        # valley_area = np.zeros(len(heights), dtype='uint32')

        # for k, h in enumerate(heights):
        #     valley_area[k] = np.sum((mask == 1) & (hand <= h))

        # Swath bins
        xmin = np.min(distance[mask])
        xmax = np.max(distance[mask])

        if (xmax - xmin) < 2000.0:
            xbins = np.arange(xmin, xmax + 10.0, 10.0)
        else:
            xbins = np.linspace(xmin, xmax, 200)

        binned = np.digitize(distance, xbins)
        x = 0.5*(xbins[1:] + xbins[:-1])

        # Profile density
        density = np.zeros_like(x, dtype='int32')

        # Absolute elevation swath profile
        swath_absolute = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        # Relative-to-stream elevation swath profile
        swath_rel_stream = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        # Relative-to-valley-floor elevation swath profile
        swath_rel_valley = np.full((len(xbins)-1, 5), np.nan, dtype='float32')

        mask_hand = (hand != hand_nodata)

        for i in range(1, len(xbins)):

            maski = mask & (binned == i)
            density[i-1] = np.sum(maski)

        # for i in range(1, len(xbins)):

            swath_elevations = elevations[maski]
            if swath_elevations.size:
                swath_absolute[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # for i in range(1, len(xbins)):

            swath_elevations = hand[maski & mask_hand]
            if swath_elevations.size:
                swath_rel_stream[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        # for i in range(1, len(xbins)):

            if with_valley_floor_estimates:

                swath_elevations = relative[maski]
                if swath_elevations.size:
                    swath_rel_valley[i-1, :] = np.percentile(swath_elevations, [5, 25, 50, 75, 95])

        if not with_valley_floor_estimates:
            swath_rel_valley = np.array([])

        values = dict(
            x=x,
            # area_valley_bottom=valley_area,
            slope_valley_floor=slope,
            z0_valley_floor=z0,
            density=density,
            absz=swath_absolute,
            hand=swath_rel_stream,
            havf=swath_rel_valley
        )

        return axis, gid, values

def UnitSwathProfile(axis, gid, bounds):

    try:

        return _UnitSwathProfile(axis, gid, bounds)

    except ValueError:

        return axis, gid, None

def SwathProfiles(axis, processes=1):
    """
    Calculate elevation swath profiles

    @api    fct-swath:profile-elevation

    @input  dem: dem
    @input  swath_bounds: ax_swaths_refaxis_bounds
    #input  swath_polygons: ax_swaths_refaxis_polygons
    @input  swath_raster: ax_swaths_refaxis
    @input  axis_measure: ax_axis_measure
    @input  axis_distance: ax_axis_distance
    @input  talweg_distance: ax_talweg_distance
    @input  height: ax_nearest_height

    @param  min_slice_width: 10.0
    @param  max_slice_count: 200
    @param  max_fit_distance: 1000.0
    @param  max_fit_height: 10.0
    @param  quantiles: [.05, .25, .5, .75, .95]

    @output swath_profile: ax_swath_elevation_npz
    """

    # swath_shapefile = config.filename('ax_swaths_refaxis_polygons', axis=axis)
    swath_bounds = config.filename('ax_swaths_refaxis_bounds', axis=axis)
    relative_errors = 0
    invalid_swaths = 0

    kwargs = dict()

    defs = xr.open_dataset(swath_bounds)
    defs.load()
    defs = defs.sortby('measure')

    length = defs['swath'].shape[0]

    def arguments():

        for k in range(length):

            gid = defs['swath'].values[k]
            bounds = tuple(defs['bounds'].values[k, :])

            # if gid < 314:
            #     continue

            yield (
                UnitSwathProfile,
                axis,
                gid,
                bounds,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length) as iterator:

            for _, gid, values in iterator:

                # profile = profiles[axis, gid]
                measure = defs['measure'].sel(swath=gid).values
                profile = [axis, gid, measure]

                if values is None:
                    invalid_swaths += 1
                    continue

                if values['havf'].size == 0:
                    relative_errors += 1

                output = config.filename('ax_swath_elevation_npz', axis=axis, gid=gid)

                np.savez(
                    output,
                    profile=profile,
                    **values)

    if invalid_swaths:
        click.secho('%d invalid swath units' % invalid_swaths, fg='yellow')

    if relative_errors:
        click.secho(
            '%d swath units without relative-to-valley-bottom profile' % relative_errors,
            fg='yellow')

def ExportElevationSwathsToNetCDF(axis):
    """
    Reads back elevation swath profile from disk,
    and bundles everyting into one netcdf file.

    @api    fct-swath:export-elevation

    @input  swath_elevation_npz: ax_swath_elevation_npz
    @input  swath_bounds: ax_swaths_refaxis_bounds

    @output swath_elevation: swath_elevation
    """

    swath_bounds = config.filename('ax_swaths_refaxis_bounds', axis=axis)

    defs = xr.open_dataset(swath_bounds)
    defs = defs.load().sortby('measure')
    length = defs['swath'].shape[0]

    swids = np.zeros(length, dtype='uint32')
    measures = np.zeros(length, dtype='float32')
    slope_fp = np.zeros(length, dtype='float32')
    z0_fp = np.zeros(length, dtype='float32')

    density = np.zeros(0, dtype='uint32')
    sw_measure = np.zeros(0, dtype='float32')
    distance = np.zeros(0, dtype='float32')
    absz = np.zeros((0, 5), dtype='float32')
    hand = np.zeros((0, 5), dtype='float32')
    havf = np.zeros((0, 5), dtype='float32')

    with click.progressbar(defs['swath'].values) as iterator:
        for k, swid in enumerate(iterator):

            measure = defs['measure'].sel(swath=swid).values
            swathfile = config.filename('ax_swath_elevation_npz', axis=axis, gid=swid)

            if not os.path.exists(swathfile):
                continue
                
            data = np.load(swathfile, allow_pickle=True)

            swids[k] = swid
            measures[k] = measure
            slope_fp[k] = data['slope_valley_floor']
            z0_fp[k] = data['z0_valley_floor']

            density = np.concatenate([density, data['density']])
            sw_measure = np.concatenate([sw_measure, np.full_like(data['density'], measure, dtype='float32')])
            distance = np.concatenate([distance, np.float32(data['x'])])
            absz = np.concatenate([absz, data['absz']])
            hand = np.concatenate([hand, data['hand']])

            if data['havf'].size == 0:
                havf = np.concatenate([havf, np.full_like(data['absz'], np.nan)])
            else:
                havf = np.concatenate([havf, data['havf']])

    dataset = xr.Dataset({
        'slope_floodplain': ('measure', slope_fp),
        'z0_floodplain': ('measure', z0_fp),
        'sw_density': ('profile', density),
        'sw_measure': ('profile', sw_measure),
        'sw_axis_distance': ('profile', distance),
        'sw_elevation_abs': (('profile', 'quantile'), absz),
        'sw_height_drainage': (('profile', 'quantile'), hand),
        'sw_height_floodplain': (('profile', 'quantile'), havf)
    }, coords={
        'axis': axis,
        'measure': measures,
        'swath': ('measure', swids),
        'quantile': [.05, .25, .5, .75, .95]
    })

    # dataset.set_index(profile=['sw_measure', 'sw_axis_distance'])

    set_metadata(dataset, 'swath_elevation')
    output = config.filename('swath_elevation', axis=axis)

    dataset.to_netcdf(output, 'w')

    return dataset
