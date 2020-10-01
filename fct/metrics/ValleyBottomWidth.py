# coding: utf-8

"""
Valley Bottom Width Metrics

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
import fiona

from ..config import config
from ..metadata import set_metadata

def swath_width(selection, unit_width, density, swath_length, resolution):
    """
    Measure width across swath profile at selected positions
    """

    if selection.size == 0:
        return np.nan

    max_density_per_unit_width = swath_length / resolution**2
    
    clamp = np.minimum(
        max_density_per_unit_width*unit_width[selection],
        density[selection])
    
    width = np.sum(clamp) / max_density_per_unit_width

    return width

def ValleyBottomWidth(axis, swath_length=200.0, resolution=5.0):
    """
    Defines
    -------

    valley_bottom_area_h
    valley_bottom_area_lr
    valley_bottom_width

    """

    swath_shapefile = config.filename('ax_valley_swaths_polygons', axis=axis)

    heights = np.arange(5.0, 15.5, 0.5)

    with fiona.open(swath_shapefile) as fs:

        size = len(fs)
        gids = np.zeros(size, dtype='uint32')
        measures = np.zeros(size, dtype='float32')
        valley_bottom_area_h = np.zeros((size, len(heights)), dtype='float32')
        valley_bottom_area_lr = np.zeros((size, 2), dtype='float32')
        valley_bottom_width = np.zeros(size, dtype='float32')
        valid = np.full(size, True)

        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                if feature['properties']['VALUE'] == 0:
                    valid[k] = False
                    continue

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                swathfile = config.filename('ax_swath_valleybottom_npz', axis=axis, gid=gid)
                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                vb_area_h = data['valley_bottom_area_h']
                vb_area_lr = data['valley_bottom_area_lr']
                vb_swath = data['valley_bottom_swath']

                # unit width of observations
                unit_width = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                unit_width[0] = x[1] - x[0]
                unit_width[-1] = x[-1] - x[-2]

                selection = (vb_swath > 0)
                vb_width = swath_width(
                    selection,
                    unit_width,
                    vb_swath,
                    swath_length,
                    resolution)

                gids[k] = gid
                measures[k] = measure
                valley_bottom_area_h[k] = vb_area_h
                valley_bottom_area_lr[k] = vb_area_lr
                valley_bottom_width[k] = vb_width

    dataset = xr.Dataset(
        {
            'swath': ('measure', gids[valid]),
            'valley_bottom_area_h': (('measure', 'height'), valley_bottom_area_h[valid]),
            'valley_bottom_area_lr': (('measure', 'side'), valley_bottom_area_lr[valid]),
            'valley_bottom_width': ('measure', valley_bottom_width[valid])
        },
        coords={
            'axis': axis,
            'measure': measures[valid],
            'height': heights,
            'side': ['left', 'right']
        })

    set_metadata(dataset, 'metrics_valleybottom_width')

    return dataset

def WriteValleyBottomWidth(axis, data):

    output = config.filename('metrics_valleybottom_width', axis=axis)

    data.to_netcdf(
        output, 'w',
        encoding={
            'swath': dict(zlib=True, complevel=9),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'valley_bottom_area_h': dict(zlib=True, complevel=9, least_significant_digit=1),
            'valley_bottom_area_lr': dict(zlib=True, complevel=9, least_significant_digit=1),
            'valley_bottom_width': dict(zlib=True, complevel=9, least_significant_digit=1)
        })
