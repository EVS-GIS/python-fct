# coding: utf-8

"""
Fluvial Corridor Width Metrics

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
import numpy as np

import click
import yaml
import xarray as xr
import fiona

from ..config import config

def SetMetadata(dataset, metafile):
    """
    Set metadata on xarray/netcdf dataset
    from YAML descriptor
    """

    filename = os.path.join(
        os.path.dirname(__file__),
        '..',
        'metadata',
        metafile + '.yml'
    )

    print(filename, os.path.exists(filename))

    if os.path.exists(filename):

        with open(filename) as fp:
            metadata = yaml.safe_load(fp)

        for attr, value in metadata['global'].items():
            dataset.attrs[attr] = value

        for variable in metadata['variables']:
            meta = metadata['variables'][variable]
            for attr, value in meta.items():
                dataset[variable].attrs[attr] = value

def swath_width(selection, unit_width, density, long_length, resolution):
    """
    Measure width across swath profile at selected positions
    """

    if selection.size == 0:
        return np.nan

    max_density = long_length / resolution**2
    clamp = np.minimum(max_density*unit_width[selection], density[selection])
    width = np.sum(clamp) / max_density

    return width

def ValleyBottomWidth(axis, swath_length=200.0, resolution=5.0):
    """
    Defines
    -------

    fcw0(h): fluvial corridor width (meter),
        measured at height h (m) above nearest drainage
        as the ratio between discrete unit's
        area and longitudinal length

    fcw1(h): fluvial corridor width (meter),
        measured on swath profile
        at height h (m) above nearest drainage

    fcw2: fluvial corridor width (meter),
        measured on swath profile
        at height +2.0 m above valley floor

    bankh: estimated bank height (meter) above water channel

        bankh1: absolute value of minimum of swath elevation above valley floor
        bankh2: absolute value of median of swath elevation above valley floor
                for swath pixels below -bankh1 + 1.0 m,
                or bankh1 if no such pixels
    """

    swath_shapefile = config.filename('ax_valley_swaths_polygons', axis=axis)

    # gids = list()
    # measures = list()
    # valley_bottom_width_values = list()
    heights = np.arange(5.0, 15.5, 0.5)

    with fiona.open(swath_shapefile) as fs:

        gids = np.zeros(len(fs), dtype='uint32')
        measures = np.zeros(len(fs), dtype='float32')
        valley_bottom_area_h = np.zeros((len(fs), len(heights)), dtype='float32')
        valley_bottom_area_lr = np.zeros((len(fs), 2), dtype='float32')
        valley_bottom_width = np.zeros(len(fs), dtype='float32')
        valid = np.full(len(fs), True)

        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                if feature['properties']['VALUE'] == 0:
                    valid[k] = False
                    continue

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                swathfile = config.filename('ax_swath_valleybottom', axis=axis, gid=gid)
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
            'vba': (('measure', 'height'), valley_bottom_area_h[valid]),
            'vbalr': (('measure', 'side'), valley_bottom_area_lr[valid]),
            'vbw': ('measure', valley_bottom_width[valid])
        },
        coords={
            'axis': axis,
            'measure': measures[valid],
            'height': heights,
            'side': ['left', 'right']
        })

    SetMetadata(dataset, 'metrics_valleybottom_width')

    return dataset

def WriteValleyBottomWidth(axis, data):

    output = config.filename('metrics_valleybottom_width', axis=axis)

    data.to_netcdf(
        output, 'w',
        encoding={
            'swath': dict(zlib=True, complevel=9),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'vba': dict(zlib=True, complevel=9, least_significant_digit=1),
            'vbalr': dict(zlib=True, complevel=9, least_significant_digit=1),
            'vbw': dict(zlib=True, complevel=9, least_significant_digit=1)
        })
