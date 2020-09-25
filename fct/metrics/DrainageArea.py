# coding: utf-8

"""
Per Swath Unit Geomorphic Metrics

***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 3 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import numpy as np

import click
import xarray as xr
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..tileio import as_window
from ..config import config

def MetricDrainageArea(axis, **kwargs):
    """
    Defines
    -------

    drainage: upstream drainage area in km²
    """

    accumulation_raster = config.tileset().filename('acc')
    swath_raster = config.tileset().filename('ax_valley_swaths', axis=axis, **kwargs)
    swath_features = config.filename('ax_valley_swaths_polygons', axis=axis, **kwargs)

    with fiona.open(swath_features) as fs:

        gids = np.zeros(len(fs), dtype='uint32')
        measures = np.zeros(len(fs), dtype='float32')
        drainage = np.zeros(len(fs), dtype='float32')

        with rio.open(accumulation_raster) as ds:

            with click.progressbar(fs) as iterator:
                for k, feature in enumerate(iterator):

                    gid = feature['properties']['GID']
                    gids[k] = gid
                    measures[k] = feature['properties']['M']
                    geometry = asShape(feature['geometry'])

                    window = as_window(geometry.bounds, ds.transform)
                    acc = ds.read(
                        1,
                        window=window,
                        boundless=True,
                        fill_value=ds.nodata)

                    with rio.open(swath_raster) as ds_swath:

                        window = as_window(geometry.bounds, ds_swath.transform)
                        swathid = ds_swath.read(
                            1,
                            window=window,
                            boundless=True,
                            fill_value=ds_swath.nodata)

                    mask = (acc != ds.nodata) & (swathid == gid)
                    if np.sum(mask) > 0:
                        drainage[k] = np.max(acc[mask])

    metrics = xr.Dataset(
        {
            'drainage': ('measure', drainage),
            'swath': ('measure', gids),
        },
        coords={
            'axis': axis,
            'measure': measures
        })

    # Metadata

    metrics['drainage'].attrs['long_name'] = 'upstream drainage area'
    metrics['drainage'].attrs['units'] = 'km²'

    metrics['axis'].attrs['long_name'] = 'stream identifier'
    metrics['swath'].attrs['long_name'] = 'swath identifier'
    metrics['measure'].attrs['long_name'] = 'position along reference axis'
    metrics['measure'].attrs['units'] = 'm'

    return metrics

def MetricElevation(axis, **kwargs):
    """
    Defines
    -------

    zmin: minimum z along mapped talweg
    """

    elevation_raster = config.tileset().filename('dem', **kwargs)
    talweg_feature = config.filename('ax_talweg', axis=axis)
    swath_raster = config.tileset().filename('ax_valley_swaths', axis=axis, **kwargs)
    swath_features = config.filename('ax_valley_swaths_polygons', axis=axis, **kwargs)

    z = np.array([])
    swathid = np.array([])

    with fiona.open(talweg_feature) as fs:

        for feature in fs:

            coordinates = np.array(feature['geometry']['coordinates'], dtype='float32')

            with rio.open(elevation_raster) as ds:

                this_z = np.array(list(ds.sample(coordinates[:, :2], 1)))
                this_z = this_z[:, 0]
                this_z[this_z == ds.nodata] = np.nan

                z = np.concatenate([z, this_z], axis=0)

            with rio.open(swath_raster) as ds:

                this_swathid = np.array(list(ds.sample(coordinates[:, :2], 1)))
                this_swathid = this_swathid[:, 0]
                # swathid[swathid == ds.nodata] = 0

                swathid = np.concatenate([swathid, this_swathid], axis=0)

    with fiona.open(swath_features) as fs:

        gids = np.zeros(len(fs), dtype='uint32')
        measures = np.zeros(len(fs), dtype='float32')
        zmin = np.zeros(len(fs), dtype='float32')

        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                gid = feature['properties']['GID']
                gids[k] = gid
                measures[k] = feature['properties']['M']

                mask = (~np.isnan(z)) & (swathid == gid)

                if np.sum(mask) > 0:
                    zmin[k] = np.min(z[mask])
                else:
                    zmin[k] = np.nan

    metrics = xr.Dataset(
        {
            'zmin': ('measure', zmin),
            'swath': ('measure', gids),
        },
        coords={
            'axis': axis,
            'measure': measures
        })

    # Metadata

    metrics['zmin'].attrs['long_name'] = 'minimum swath elevation along talweg'
    metrics['zmin'].attrs['standard_name'] = 'surface_altitude'
    metrics['zmin'].attrs['units'] = 'm'
    metrics['zmin'].attrs['vertical_ref'] = config.vertical_ref

    metrics['axis'].attrs['long_name'] = 'stream identifier'
    metrics['swath'].attrs['long_name'] = 'swath identifier'
    metrics['measure'].attrs['long_name'] = 'position along reference axis'
    metrics['measure'].attrs['units'] = 'm'

    return metrics
