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

# workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def FluvialCorridorWidth(axis):
    """
    Defines
    -------

    fcw: fluvial corridor width (meter)

        fcw2: measured at +2.0 m above valley floor
        fcw8: measured at +8.0 m above nearest drainage
        fcw10: measured at +10.0 m above nearest drainage

    bankh: estimated bank height (meter) above water channel

        bankh1: opposite of minimum of swath elevation above valley floor
        bankh2: opposite of median of swath elevation above valley floor
                for swath pixels below -bankh1 + 1.0 m,
                or bankh1 if no such pixels
    """

    # dgo_shapefile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REF', 'DGO.shp')
    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)
    # dgo_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'DGO.vrt')
    # accumulation_raster = '/var/local/fct/RMC/ACC_RGE5M_TILES.vrt'
    # output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_DRAINAGE_AREA.csv')
    # metrics = dict()

    gids = list()
    measures = list()
    values = list()

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

                try:
                    density = data['density']
                    density_max = np.max(density)
                except ValueError:
                    density = np.zeros(0, dtype='uint32')
                    density_max = 0

                if density_max == 0 or x.shape[0] < 3:

                    gids.append(gid)
                    measures.append(measure)
                    values.append((np.nan, np.nan, np.nan, np.nan, np.nan))
                    continue

                # unit width of observations
                w = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                w[0] = x[1] - x[0]
                w[-1] = x[-1] - x[-2]

                if hvf.size > 0:

                    selection = (hvf[:, 2] <= 2.0)
                    if selection.size > 0:
                        fcw2 = np.sum(w[selection] * density[selection]) / density_max
                    else:
                        fcw2 = np.nan

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

                selection = (hand[:, 2] <= 8.0)
                if selection.size > 0:
                    fcw8 = np.sum(w[selection] * density[selection]) / density_max
                else:
                    fcw8 = np.nan

                selection = (hand[:, 2] <= 10.0)
                if selection.size > 0:
                    fcw10 = np.sum(w[selection] * density[selection]) / density_max
                else:
                    fcw10 = np.nan

                # values.append((gid, measure, fcw2, fcw8, fcw10, bankh1, bankh2))

                gids.append(gid)
                measures.append(measure)
                values.append((fcw2, fcw8, fcw10, bankh1, bankh2))

    # dtype = np.dtype([
    #     ('gid', 'int'),
    #     ('measure', 'float32'),
    #     ('fcw2', 'float32'),
    #     ('fcw8', 'float32'),
    #     ('fcw10', 'float32'),
    #     ('bankh1', 'float32'),
    #     ('bankh2', 'float32')
    # ])

    # return np.sort(np.array(values, dtype=dtype), order='measure')

    gids = np.array(gids, dtype='uint32')
    measures = np.array(measures, dtype='float32')
    data = np.array(values, dtype='float32')

    return xr.Dataset(
        {
            'measure': ('swath', measures),
            'fcw2': ('swath', data[:, 0]),
            'fcw8': ('swath', data[:, 1]),
            'fcw10': ('swath', data[:, 2]),
            'bankh1': ('swath', data[:, 3]),
            'bankh2': ('swath', data[:, 4])
        },
        coords={
            'axis': axis,
            'swath': gids,
        })

def WriteFluvialCorridorWidth(axis, data):

    # output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'FLUVIAL_CORRIDOR_WIDTH.nc')
    output = config.filename('metrics_fcw', axis=axis)

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'fcw2': dict(zlib=True, complevel=9, least_significant_digit=1),
            'fcw8': dict(zlib=True, complevel=9, least_significant_digit=1),
            'fcw10': dict(zlib=True, complevel=9, least_significant_digit=1),
            'bankh1': dict(zlib=True, complevel=9, least_significant_digit=0),
            'bankh2': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

def _LandCoverWidth(axis, kind, variable):
    """
    Aggregate LandCover Swat Data
    """

    # dgo_shapefile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REF', 'DGO.shp')
    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)

    gids = list()
    measures = list()
    values = list()
    with fiona.open(dgo_shapefile) as fs:
        with click.progressbar(fs) as iterator:
            for feature in iterator:

                gid = feature['properties']['GID']
                measure = feature['properties']['M']
                # swathfile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'CONTINUITY', 'SWATH_CONTINUITY_%04d.npz' % gid)
                swathfile = config.filename('ax_swath_landcover', axis=axis, gid=gid, kind=kind.upper())
                data = np.load(swathfile, allow_pickle=True)

                x = data['x']
                classes = data['classes']
                swath = data['swath']

                try:
                    density = data['density']
                    density_max = np.max(density)
                except ValueError:
                    density = np.zeros(0, dtype='uint32')
                    density_max = 0

                if density_max == 0 or x.shape[0] < 3:

                    gids.append(gid)
                    measures.append(measure)
                    width = np.zeros((9, 2), dtype='float32')
                    values.append(width)
                    continue

                # count = np.ma.sum(np.ma.masked_array(swath, np.isnan(swath)), axis=1)
                dominant = np.ma.argmax(np.ma.masked_array(swath, np.isnan(swath)), axis=1)

                # unit width of observations
                unit_width = 0.5 * (np.roll(x, -1) - np.roll(x, 1))
                unit_width[0] = x[1] - x[0]
                unit_width[-1] = x[-1] - x[-2]

                # width = np.zeros(9, dtype='float32')
                width = np.zeros((9, 2), dtype='float32')

                for k in range(len(classes)):

                    if classes[k] == 255:
                        continue

                    # selection = (swath[:, k] / count) > threshold
                    # if selection.size > 0:
                    #     width[classes[k]] = np.sum(unit_width[selection] * density[selection]) / density_max
                    # else:
                    #     width[classes[k]] = 0

                    selection = (dominant == k)
                    if selection.size > 0:
                        width[classes[k]] = np.sum(unit_width[selection] * density[selection]) / density_max
                    else:
                        width[classes[k]] = 0

                    selection = (dominant == k) & (x >= 0)
                    if selection.size > 0:
                        width[classes[k], 0] = np.sum(unit_width[selection] * density[selection]) / density_max
                    else:
                        width[classes[k], 0] = 0

                    selection = (dominant == k) & (x < 0)
                    if selection.size > 0:
                        width[classes[k], 1] = np.sum(unit_width[selection] * density[selection]) / density_max
                    else:
                        width[classes[k], 1] = 0

                # values.append(tuple([gid, measure] + width.tolist()))

                gids.append(gid)
                measures.append(measure)
                values.append(width)

    # dtype = [
    #     ('gid', 'int'),
    #     ('measure', 'float32')
    # ] + [('lcc%d' % k, 'float32') for k in range(9)]

    # return np.sort(np.array(values, dtype=np.dtype(dtype)), order='measure')
    gids = np.array(gids, dtype='uint32')
    measures = np.array(measures, dtype='float32')
    data = np.array(values, dtype='float32')

    return xr.Dataset(
        {
            'measure': (('swath'), measures),
            variable: (('swath', 'landcover', 'side'), data)
        },
        coords={
            'axis': axis,
            'swath': gids,
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

def LandCoverWidth(axis):
    """
    Defines
    -------

    lcwk: continuity width (meter) for land cover class k
    """

    return _LandCoverWidth(axis, kind='std', variable='lcw')

def ContinuityWidth(axis):
    """
    Defines
    -------

    lcck: continuity width (meter) for land cover class k
    """

    return _LandCoverWidth(axis, kind='continuity', variable='lcc')

def WriteLandCoverWidth(axis, data):

    # output = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'LANDCOVER_WIDTH.nc')
    output = config.filename('metrics_lcc', axis=axis)

    data.to_netcdf(
        output, 'w',
        encoding={
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
            'lcw': dict(zlib=True, complevel=9, least_significant_digit=2),
            'lcc': dict(zlib=True, complevel=9, least_significant_digit=2)
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
