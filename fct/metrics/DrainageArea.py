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

from multiprocessing import Pool
import numpy as np

import click
import xarray as xr
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape

from ..tileio import as_window
from ..config import config
from ..metadata import set_metadata
from ..cli import starcall

def SwathDrainageArea(k, axis, swid, bounds, **kwargs):
    """
    Calculate max drainage area for swath k
    """

    accumulation_raster = config.tileset().filename('acc')
    swath_raster = config.tileset().filename('ax_valley_swaths', axis=axis, **kwargs)


    with rio.open(accumulation_raster) as ds:

        window = as_window(bounds, ds.transform)
        acc = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        with rio.open(swath_raster) as ds_swath:

            window = as_window(bounds, ds_swath.transform)
            swathid = ds_swath.read(
                1,
                window=window,
                boundless=True,
                fill_value=ds_swath.nodata)

        mask = (acc != ds.nodata) & (swathid == swid)
        if np.sum(mask) > 0:
            return k, np.max(acc[mask])

        return k, np.nan

def MetricDrainageArea(axis, processes, **kwargs):
    """
    Defines
    -------

    drainage_area: upstream drainage area in km²

    @api    fct-metrics:drainage-area

    @input  accumulation: acc
    @input  swath_polygons: ax_valley_swaths_polygons
    @input  swath_raster: ax_valley_swaths

    @output drainage_area: metrics_drainage_area
    """

    swath_features = config.filename('ax_valley_swaths_polygons', axis=axis, **kwargs)

    with fiona.open(swath_features) as fs:

        size = len(fs)
        gids = np.zeros(size, dtype='uint32')
        measures = np.zeros(size, dtype='float32')
        drainage = np.zeros(size, dtype='float32')
        valid = np.full(size, True)

    def arguments():

        with fiona.open(swath_features) as fs:
            for k, feature in enumerate(fs):

                gid = feature['properties']['GID']
                gids[k] = gid
                measures[k] = feature['properties']['M']
                geometry = asShape(feature['geometry'])

                if feature['properties']['VALUE'] != 2:
                    valid[k] = False
                    continue

                yield (
                    SwathDrainageArea,
                    k,
                    axis,
                    gid,
                    geometry.bounds,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=size) as iterator:
            for k, value in iterator:
                drainage[k] = value

    metrics = xr.Dataset(
        {
            'drainage_area': ('measure', drainage[valid])
        },
        coords={
            'axis': axis,
            'measure': measures[valid],
            'swath': ('measure', gids[valid]),
        })

    # Metadata

    set_metadata(metrics, 'metrics_drainage_area')

    output = config.filename('metrics_drainage_area', axis=axis)
    metrics.to_netcdf(output, 'w')

    return metrics

# def MetricElevation(axis, **kwargs):
#     """
#     Defines
#     -------

#     zmin: minimum z along mapped talweg
#     """

#     elevation_raster = config.tileset().filename('dem', **kwargs)
#     talweg_feature = config.filename('ax_talweg', axis=axis)
#     swath_raster = config.tileset().filename('ax_valley_swaths', axis=axis, **kwargs)
#     swath_features = config.filename('ax_valley_swaths_polygons', axis=axis, **kwargs)

#     z = np.array([])
#     swathid = np.array([])

#     with fiona.open(talweg_feature) as fs:

#         for feature in fs:

#             coordinates = np.array(feature['geometry']['coordinates'], dtype='float32')

#             with rio.open(elevation_raster) as ds:

#                 this_z = np.array(list(ds.sample(coordinates[:, :2], 1)))
#                 this_z = this_z[:, 0]
#                 this_z[this_z == ds.nodata] = np.nan

#                 z = np.concatenate([z, this_z], axis=0)

#             with rio.open(swath_raster) as ds:

#                 this_swathid = np.array(list(ds.sample(coordinates[:, :2], 1)))
#                 this_swathid = this_swathid[:, 0]
#                 # swathid[swathid == ds.nodata] = 0

#                 swathid = np.concatenate([swathid, this_swathid], axis=0)

#     with fiona.open(swath_features) as fs:

#         gids = np.zeros(len(fs), dtype='uint32')
#         measures = np.zeros(len(fs), dtype='float32')
#         zmin = np.zeros(len(fs), dtype='float32')

#         with click.progressbar(fs) as iterator:
#             for k, feature in enumerate(iterator):

#                 gid = feature['properties']['GID']
#                 gids[k] = gid
#                 measures[k] = feature['properties']['M']

#                 mask = (~np.isnan(z)) & (swathid == gid)

#                 if np.sum(mask) > 0:
#                     zmin[k] = np.min(z[mask])
#                 else:
#                     zmin[k] = np.nan

#     metrics = xr.Dataset(
#         {
#             'zmin': ('measure', zmin),
#             'swath': ('measure', gids),
#         },
#         coords={
#             'axis': axis,
#             'measure': measures
#         })

#     # Metadata

#     metrics['zmin'].attrs['long_name'] = 'minimum swath elevation along talweg'
#     metrics['zmin'].attrs['standard_name'] = 'surface_altitude'
#     metrics['zmin'].attrs['units'] = 'm'
#     metrics['zmin'].attrs['vertical_ref'] = config.vertical_ref

#     metrics['axis'].attrs['long_name'] = 'stream identifier'
#     metrics['swath'].attrs['long_name'] = 'swath identifier'
#     metrics['measure'].attrs['long_name'] = 'position along reference axis'
#     metrics['measure'].attrs['units'] = 'm'

#     return metrics
