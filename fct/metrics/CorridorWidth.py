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

import numpy as np
import xarray as xr
import click

import fiona

from ..config import config

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

def CorridorWidth(axis, long_length=200.0, resolution=5.0):
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

        bankh1: opposite of minimum of swath elevation above valley floor
        bankh2: opposite of median of swath elevation above valley floor
                for swath pixels below -bankh1 + 1.0 m,
                or bankh1 if no such pixels
    """

    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)

    gids = list()
    measures = list()
    values = list()
    fcw0_values = list()
    fcw1_values = list()
    heights = np.arange(5.0, 15.5, 0.5)

    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                # swathfile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'ELEVATION', 'SWATH_%04d.npz' % gid)
                swathfile = config.filename('ax_swath_elevation', axis=axis, gid=gid)
                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                hand = data['hand']
                hvf = data['hvf']

                # Areal width

                varea = data['varea']
                fcw0 = np.float32(varea * resolution**2 / long_length)
                fcw0_values.append(fcw0)

                # Swath width

                try:
                    density = data['density']
                    density_max = np.max(density)
                except ValueError:
                    density = np.zeros(0, dtype='uint32')
                    density_max = 0

                if density_max == 0 or x.shape[0] < 3:

                    gids.append(gid)
                    measures.append(measure)
                    values.append((np.nan, np.nan, np.nan))
                    fcw1 = np.zeros(len(heights), dtype='float32')
                    fcw1_values.append(fcw1)
                    continue

                # unit width of observations
                unit_width = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                unit_width[0] = x[1] - x[0]
                unit_width[-1] = x[-1] - x[-2]

                if hvf.size > 0:

                    selection = (hvf[:, 2] <= 2.0)
                    fcw2 = swath_width(
                        selection,
                        unit_width,
                        density,
                        long_length,
                        resolution)

                    mask = np.isnan(hvf[:, 2])
                    bankh1 = np.ma.min(np.ma.masked_array(hvf[:, 2], mask))
                    if bankh1 is np.ma.masked:

                        bankh1 = np.nan
                        bankh2 = np.nan

                    else:

                        bankh1 = -bankh1

                        mask = (hvf[:, 2] >= min(-0.5, -bankh1 + 1.0))
                        bankh2 = np.ma.median(np.ma.masked_array(hvf[:, 2], mask))

                        if bankh2 is np.ma.masked:
                            bankh2 = bankh1
                        else:
                            bankh2 = -bankh2

                else:

                    fcw2 = np.nan
                    bankh1 = np.nan
                    bankh2 = np.nan

                # selection = (hand[:, 2] <= 8.0)
                # if selection.size > 0:
                #     fcw8 = np.sum(w[selection] * density[selection]) / density_max
                # else:
                #     fcw8 = np.nan

                # selection = (hand[:, 2] <= 10.0)
                # if selection.size > 0:
                #     fcw10 = np.sum(w[selection] * density[selection]) / density_max
                # else:
                #     fcw10 = np.nan

                fcw1 = np.zeros(len(heights), dtype='float32')

                for k, h in enumerate(heights):

                    selection = (hand[:, 2] <= h)
                    fcw1[k] = swath_width(
                        selection,
                        unit_width,
                        density,
                        long_length,
                        resolution)

                fcw1_values.append(fcw1)

                gids.append(gid)
                measures.append(measure)
                values.append((fcw2, bankh1, bankh2))

    gids = np.array(gids, dtype='uint32')
    measures = np.array(measures, dtype='float32')
    fcw0 = np.array(fcw0_values, dtype='float32')
    fcw1 = np.array(fcw1_values, dtype='float32')
    data = np.array(values, dtype='float32')

    return xr.Dataset(
        {
            'swath': ('measure', gids),
            'fcw0': (('measure', 'height'), fcw0),
            'fcw1': (('measure', 'height'), fcw1),
            'fcw2': ('measure', data[:, 0]),
            'bankh1': ('measure', data[:, 1]),
            'bankh2': ('measure', data[:, 2])
        },
        coords={
            'axis': axis,
            'measure': measures,
            'height': heights
        })

def WriteCorridorWidth(axis, data):

    output = config.filename('metrics_fcw', axis=axis)

    data.to_netcdf(
        output, 'w',
        encoding={
            'swath': dict(zlib=True, complevel=9),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'fcw0': dict(zlib=True, complevel=9, least_significant_digit=1),
            'fcw1': dict(zlib=True, complevel=9, least_significant_digit=1),
            'fcw2': dict(zlib=True, complevel=9, least_significant_digit=1),
            'bankh1': dict(zlib=True, complevel=9, least_significant_digit=0),
            'bankh2': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

def test(axis=1044):

    # pylint: disable=import-outside-toplevel

    import os
    import matplotlib as mpl
    from ..plotting.PlotCorridor import (
        PlotMetric,
        PlotContinuityProfile,
        PlotLeftRightContinuityProfile
    )

    mpl.use('cairo')
    workdir = config.workdir

    fcw = FluvialCorridorWidth(axis)
    lcc = ContinuityWidth(axis)

    WriteFluvialCorridorWidth(axis, fcw)
    WriteLandCoverWidth(axis, lcc)

    data = fcw.merge(lcc).sortby(fcw['measure'])
    print(data)

    PlotMetric(
        data,
        'measure',
        'fcw2',
        window=5,
        title='Corridor Width (FCW2)',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW2.pdf'))

    PlotMetric(
        data,
        'measure',
        'fcw8',
        window=5,
        title='Corridor Width (FCW8)',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW8.pdf'))

    PlotMetric(
        data,
        'measure',
        'fcw10',
        window=5,
        title='Corridor Width (FCW10)',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW10.pdf'))

    PlotMetric(
        data,
        'measure',
        'fcw2',
        'fcw8',
        'fcw10',
        window=5,
        title='Fluvial Corridor Width Metrics',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'FCW_METRICS.pdf'))

    PlotContinuityProfile(
        data,
        window=5,
        proportion=False,
        title='Corridor Width Profile',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'CORRIDOR_PROFILE.pdf'))

    PlotContinuityProfile(
        data,
        window=5,
        proportion=True,
        title='Corridor Width Profile',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'CORRIDOR_PROFILE_PROP.pdf'))

    PlotLeftRightContinuityProfile(
        data,
        window=5,
        proportion=False,
        title='Corridor Width Profile',
        filename=os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'PDF', 'CORRIDOR_PROFILE_LEFTRIGHT.pdf'))

    return data
