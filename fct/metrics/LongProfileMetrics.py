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

from operator import itemgetter
from math import ceil

import numpy as np

import click
import xarray as xr
import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import (
    asShape,
    MultiLineString)

from .. import transform as fct
from ..rasterize import rasterize_linestringz
from ..tileio import as_window
from ..config import config

def MetricDrainageArea(axis, **kwargs):
    """
    Defines
    -------

    drainage: upstream drainage area in km²
    """

    accumulation_raster = config.filename('acc')
    swath_raster = config.filename('ax_swaths', axis=axis, **kwargs)
    swath_features = config.filename('ax_swath_features', axis=axis, **kwargs)

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

                    drainage[k] = np.max(acc[(acc != ds.nodata) & (swathid == gid)])

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

    elevation_raster = config.filename('tiled', **kwargs)
    talweg_feature = config.filename('ax_talweg', axis=axis)
    swath_raster = config.filename('ax_swaths', axis=axis, **kwargs)
    swath_features = config.filename('ax_swath_features', axis=axis, **kwargs)

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

# def MetricSwathSlope(axis, distance):
#     """
#     DGO Slope, expressed in percent
#     """

#     filename = os.path.join(config.workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_MINZ.csv')
#     output = os.path.join(config.workdir, 'AXES', 'AX%03d' % axis, 'METRICS', 'DGO_SLOPE.csv')

#     def gid(t):
#         return int(t[1])

#     def elevation(t):
#         return float(t[2])
#         # return 0.5 * (float(t[2]) + float(t[3]))

#     with open(filename) as fp:
#         data = [line.strip().split(',') for line in fp]
#         elevations = np.array([elevation(t) for t in data])

#     slope = np.diff(elevations, prepend=elevations[0]) * 100 / distance

#     with open(output, 'w') as fp:
#         for k, value in enumerate(slope):
#             fp.write('%d,%d,%.3f\n' % (axis, gid(data[k]), value))

def MetricSlopes(axis, **kwargs):

    talweg_feature = config.filename('ax_talweg', axis=axis, **kwargs)
    refaxis_feature = config.filename('ax_refaxis', axis=axis)
    elevation_raster = config.filename('dem', **kwargs)
    measure_raster = config.filename('ax_axis_measure', axis=axis, **kwargs)
    swath_raster = config.filename('ax_swaths', axis=axis, **kwargs)
    swath_features = config.filename('ax_swath_features', axis=axis, **kwargs)

    all_coordinates = np.zeros((0,2), dtype='float32')
    z = np.array([])
    m = np.array([])
    swathid = np.array([])

    # Sort talweg segments by first point M coordinate, descending

    talweg_fids = list()
    segments = list()

    with rio.open(measure_raster) as ds:
        with fiona.open(talweg_feature) as fs:
            for feature in fs:

                fid = feature['id']
                firstm = next(ds.sample([feature['geometry']['coordinates'][0][:2]], 1))
                talweg_fids.append((fid, firstm))

    with fiona.open(talweg_feature) as fs:
        for fid, _ in reversed(sorted(talweg_fids, key=itemgetter(1))):

            feature = fs.get(fid)
            coordinates = np.array(feature['geometry']['coordinates'], dtype='float32')
            segments.append(asShape(feature['geometry']))

            all_coordinates = np.concatenate([all_coordinates, coordinates[:, :2]], axis=0)

            with rio.open(elevation_raster) as ds:

                this_z = np.array(list(ds.sample(coordinates[:, :2], 1)))
                this_z = this_z[:, 0]
                this_z[this_z == ds.nodata] = np.nan

                z = np.concatenate([z, this_z], axis=0)

            with rio.open(measure_raster) as ds:

                this_m = np.array(list(ds.sample(coordinates[:, :2], 1)))
                this_m = this_m[:, 0]

                m = np.concatenate([m, this_m], axis=0)

            with rio.open(swath_raster) as ds:

                this_swathid = np.array(list(ds.sample(coordinates[:, :2], 1)))
                this_swathid = this_swathid[:, 0]
                # swathid[swathid == ds.nodata] = 0

                swathid = np.concatenate([swathid, this_swathid], axis=0)

    measure_raster = config.filename('ax_buffer_profile', axis=axis, **kwargs)

    with fiona.open(refaxis_feature) as fs:

        assert len(fs) == 1

        coord = itemgetter(0, 1)

        for feature in fs:

            refaxis_pixels = list()
            refaxis_m = list()
            m0 = feature['properties'].get('M0', 0.0)

            coordinates = np.array([
                coord(p) + (m0,) for p in reversed(feature['geometry']['coordinates'])
            ], dtype='float32')

            coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                coordinates[1:, :2] - coordinates[:-1, :2],
                axis=1))

            with rio.open(swath_raster) as ds:

                coordinates[:, :2] = fct.worldtopixel(coordinates[:, :2], ds.transform)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j, mcoord in rasterize_linestringz(a, b):
                        refaxis_pixels.append((i, j))
                        refaxis_m.append(mcoord)

                refaxis_m = np.array(refaxis_m)

                refaxis_coordinates = fct.pixeltoworld(
                    np.array(refaxis_pixels, dtype='int32'),
                    ds.transform)

                refaxis_swathid = np.array(list(ds.sample(refaxis_coordinates, 1)))
                refaxis_swathid = refaxis_swathid[:, 0]

    # s: curvilinear talweg coordinate, from upstream to downstream
    s = np.zeros(len(all_coordinates), dtype='float32')
    s[1:] = np.cumsum(
        np.linalg.norm(
            all_coordinates[1:, :] - all_coordinates[:-1, :],
            axis=1))

    talweg = MultiLineString(segments)

    ref_segments = list()
    ref_segments_vf = list()

    with fiona.open(swath_features) as fs:

        gids = np.zeros(len(fs), dtype='uint32')
        measures = np.zeros(len(fs), dtype='float32')
        values = np.zeros((len(fs), 7), dtype='float32')

        with click.progressbar(fs) as iterator:
            for k, feature in enumerate(iterator):

                gid = feature['properties']['GID']
                polygon = asShape(feature['geometry'])

                gids[k] = gid
                measures[k] = feature['properties']['M']

                talweg_length = talweg.intersection(polygon).length
                values[k, 0] = talweg_length

                mask = (~np.isnan(z)) & (swathid == gid)

                if np.sum(mask) > 0:

                    moffset = m[mask][0]

                    Y = z[mask]
                    X = np.column_stack([
                        s[mask] - s[mask][0],
                        np.ones_like(Y),
                    ])
                    (slope_talweg, z0_talweg), sqerror_talweg, _, _ = np.linalg.lstsq(X, Y, rcond=None)

                    if len(sqerror_talweg) == 0:
                        sqerror_talweg = 0.0

                    values[k, 1] = -100.0 * slope_talweg
                    values[k, 2] = z0_talweg
                    values[k, 3] = sqerror_talweg

                    X = np.column_stack([
                        m[mask] - moffset,
                        np.ones_like(Y),
                    ])
                    (slope_valley, z0_valley), sqerror_valley, _, _ = np.linalg.lstsq(X, Y, rcond=None)

                    if len(sqerror_valley) == 0:
                        sqerror_valley = 0

                    # m axis is oriented from downstream to upstream,
                    # and yields positive slopes
                    values[k, 4] = 100.0 * slope_valley
                    values[k, 5] = z0_valley
                    values[k, 6] = sqerror_valley

                    mask_ref = (refaxis_swathid == gid)

                    if np.sum(mask_ref) > 0:

                        refaxis_z = slope_valley * (refaxis_m[mask_ref] - moffset) + z0_valley

                        ref_segments.append(np.column_stack([
                            refaxis_coordinates[mask_ref],
                            refaxis_z,
                            refaxis_m[mask_ref]
                        ]))

                        swathfile = config.filename('ax_swath_elevation', axis=axis, gid=gid)
                        swathdata = np.load(swathfile, allow_pickle=True)

                        slope_valley_floor = swathdata['slope_valley_floor']
                        z0_valley_floor = swathdata['z0_valley_floor']

                        if np.isnan(slope_valley_floor):

                            ref_segments_vf.append(np.column_stack([
                                refaxis_coordinates[mask_ref],
                                refaxis_z,
                                refaxis_m[mask_ref]
                            ]))

                        else:

                            refaxis_z_vf = slope_valley_floor * (refaxis_m[mask_ref]) + z0_valley_floor

                            ref_segments_vf.append(np.column_stack([
                                refaxis_coordinates[mask_ref],
                                refaxis_z_vf,
                                refaxis_m[mask_ref]
                            ]))

                    else:

                        ref_segments.append(np.array([]))
                        ref_segments_vf.append(np.array([]))

                else:

                    values[k, :] = np.nan
                    ref_segments.append(np.array([]))
                    ref_segments_vf.append(np.array([]))

    metrics = xr.Dataset(
        {
            'swath': ('measure', gids),
            'twl': ('measure', values[:, 0]),
            'tws': ('measure', values[:, 1]),
            'twz0': ('measure', values[:, 2]),
            'twse': ('measure', values[:, 3]),
            'vfs': ('measure', values[:, 4]),
            'vfz0': ('measure', values[:, 5]),
            'vfse': ('measure', values[:, 6])
        },
        coords={
            'axis': axis,
            'measure': measures
        })

    # Metadata

    metrics['twl'].attrs['long_name'] = 'intercepted talweg length'
    metrics['twl'].attrs['units'] = 'm'

    metrics['tws'].attrs['long_name'] = 'talweg slope'
    metrics['tws'].attrs['units'] = 'percent'

    metrics['twz0'].attrs['long_name'] = 'talweg z-offset, at first swath point'
    metrics['twz0'].attrs['units'] = 'm'
    metrics['twz0'].attrs['vertical_ref'] = config.vertical_ref

    metrics['twse'].attrs['long_name'] = 'talweg slope square regression error'
    metrics['twse'].attrs['units'] = 'm²'

    metrics['vfs'].attrs['long_name'] = 'valley slope'
    metrics['vfs'].attrs['units'] = 'percent'

    metrics['vfz0'].attrs['long_name'] = 'valley z-offset, at first swath point'
    metrics['vfz0'].attrs['units'] = 'm'
    metrics['vfz0'].attrs['vertical_ref'] = config.vertical_ref

    metrics['vfse'].attrs['long_name'] = 'valley slope square regression error'
    metrics['vfse'].attrs['units'] = 'm²'

    metrics['axis'].attrs['long_name'] = 'stream identifier'
    metrics['swath'].attrs['long_name'] = 'swath identifier'
    metrics['measure'].attrs['long_name'] = 'position along reference axis'
    metrics['measure'].attrs['units'] = 'm'

    return metrics, ref_segments, ref_segments_vf

def AdjustRefElevationGaps(segments):
    """
    Adjust gaps in the output of MetricSlopes()
    """

    def measure(segment):
        """sort by m coordinate of first point"""
        return segment[0][3]

    sorted_segments = sorted(
        [
            np.copy(segment) for segment in segments
            if segment.size > 0 and segment[0, 3] != segment[-1, 3]
        ],
        key=measure)

    missing = list()

    for k, segment in enumerate(sorted_segments):

        if k == 0:
            continue

        m0 = segment[0, 3]

        if segment[0, 2] < sorted_segments[k-1][-1, 2]:

            # z0 = 0.5 * (segment[0, 2] + sorted_segments[k-1][-1, 2])
            z0 = sorted_segments[k-1][-1, 2]

            m1 = segment[-1, 3]
            z1 = segment[-1, 2]

            # linear interpolation between z0 and z1
            segment[:, 2] = (segment[:, 3] - m0) / (m1 - m0) * (z1 - z0) + z0

        m0 = sorted_segments[k-1][-1, 3]
        m1 = segment[0, 3]

        if m1 - m0 > 50.0:

            new_m = np.linspace(m1, m0, ceil((m1 - m0) / 10.0))
            new_segment = np.zeros((len(new_m), 4), dtype='float32')
            new_segment[:, 3] = new_m

            for j in range(3):

                c0 = sorted_segments[k-1][-1, j]
                c1 = segment[0, j]

                # linear interpolation between coordinate c0 and c1
                new_segment[:, j] = (new_m - m0) / (m1 - m0) * (c1 - c0) + c0

            missing.append(new_segment)

    # print(len(missing))
    sorted_segments.extend(missing)

    return sorted(sorted_segments, key=measure)

def ExportValleyProfile(axis, valley_profile, destination):

    dataset = xr.Dataset(
        {
            'x': ('measure', valley_profile[:, 0]),
            'y': ('measure', valley_profile[:, 1]),
            'z': ('measure', valley_profile[:, 2])
        },
        coords={
            'axis': axis,
            'measure': valley_profile[:, 3]
        })

    dataset['x'].attrs['long_name'] = 'x coordinate'
    dataset['x'].attrs['standard_name'] = 'projection_x_coordinate'
    dataset['x'].attrs['units'] = 'm'

    dataset['y'].attrs['long_name'] = 'y coordinate'
    dataset['y'].attrs['standard_name'] = 'projection_y_coordinate'
    dataset['y'].attrs['units'] = 'm'

    dataset['z'].attrs['long_name'] = 'height above valley floor'
    dataset['z'].attrs['standard_name'] = 'surface_height'
    dataset['z'].attrs['units'] = 'm'
    dataset['z'].attrs['grid_mapping'] = 'crs: x y'
    dataset['z'].attrs['coordinates'] = 'x y'

    dataset['axis'].attrs['long_name'] = 'stream identifier'
    dataset['measure'].attrs['long_name'] = 'position along reference axis'
    dataset['measure'].attrs['long_name'] = 'linear_measure_coordinate'
    dataset['measure'].attrs['units'] = 'm'

    dataset.attrs['crs'] = 'EPSG:2154'
    dataset.attrs['FCT'] = 'Fluvial Corridor Toolbox Valley Profile 1.0.5'
    dataset.attrs['Conventions'] = 'CF-1.8'

    dataset.to_netcdf(
        destination,
        'w',
        encoding={
            'x': dict(zlib=True, complevel=9, least_significant_digit=2),
            'y': dict(zlib=True, complevel=9, least_significant_digit=2),
            'z': dict(zlib=True, complevel=9, least_significant_digit=1),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

def ExportLongProfile(dataset, destination):

    dataset.attrs['crs'] = 'EPSG:2154'
    dataset.attrs['FCT'] = 'Fluvial Corridor Toolbox Long Profile 1.0.5'
    dataset.attrs['Conventions'] = 'CF-1.8'

    dataset.to_netcdf(
        destination,
        'w',
        encoding={
            'swath': dict(zlib=True, complevel=9),
            'zmin': dict(zlib=True, complevel=9, least_significant_digit=1),
            'drainage': dict(zlib=True, complevel=9, least_significant_digit=3),
            'twl': dict(zlib=True, complevel=9, least_significant_digit=0),
            'tws': dict(zlib=True, complevel=9, least_significant_digit=3),
            'twz0': dict(zlib=True, complevel=9, least_significant_digit=1),
            'twse': dict(zlib=True, complevel=9, least_significant_digit=3),
            'vfs': dict(zlib=True, complevel=9, least_significant_digit=3),
            'vfz0': dict(zlib=True, complevel=9, least_significant_digit=1),
            'vfse':dict(zlib=True, complevel=9, least_significant_digit=3),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

def LongProfileMetrics(axis):

    click.echo('Gather long profile metrics')
    m1 = MetricDrainageArea(axis)
    m2 = MetricElevation(axis)
    m3, seg1, seg2 = MetricSlopes(axis)

    click.echo('Write long profile metrics')
    metrics = m1.merge(m2).merge(m3)
    metrics_file = config.filename('metrics_long_profile', axis=axis)
    ExportLongProfile(metrics, metrics_file)

    click.echo('Write reference axis talweg profile')
    segments = AdjustRefElevationGaps(seg1)
    valley_profile = np.float32(np.concatenate(segments, axis=0))
    valley_floor_file = config.filename('ax_refaxis_talweg_profile', axis=axis)
    ExportValleyProfile(axis, valley_profile, valley_floor_file)

    click.echo('Write reference axis valley profile')
    segments = AdjustRefElevationGaps(seg2)
    valley_profile = np.float32(np.concatenate(segments, axis=0))
    valley_floor_file = config.filename('ax_refaxis_valley_profile', axis=axis)
    ExportValleyProfile(axis, valley_profile, valley_floor_file)
