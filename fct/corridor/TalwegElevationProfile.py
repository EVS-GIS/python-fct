# coding: utf-8

"""
Talweg Elevation Profile

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
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative

import xarray as xr
import rasterio as rio
import fiona

from .. import transform as fct
from ..rasterize import rasterize_linestringz
from ..config import config
from ..metadata import set_metadata

def ExportTalwegElevationProfile(axis, profile, destination):
    """
    Write valley profile data to NetCDF file
    """

    # mz = valley_profile[:, [3, 1]]

    # diff1 = np.concatenate([[(0, 0)], mz[1:] - mz[:-1]])
    # diff2 = np.concatenate([mz[1:] - mz[:-1], [(0, 0)]])
    # diff = 0.5 * (diff1 + diff2)
    # slope = diff[:, 1] / diff[:, 0]

    dataset = xr.Dataset(
        {
            'x': ('measure', profile[:, 1]),
            'y': ('measure', profile[:, 2]),
            'z': ('measure', profile[:, 3]),
            'slope': ('measure', profile[:, 4])
        },
        coords={
            'axis': axis,
            'measure': profile[:, 0]
        })

    set_metadata(dataset, 'elevation_profile_talweg')

    dataset.to_netcdf(
        destination,
        'w',
        encoding={
            'x': dict(zlib=True, complevel=9, least_significant_digit=2),
            'y': dict(zlib=True, complevel=9, least_significant_digit=2),
            'z': dict(zlib=True, complevel=9, least_significant_digit=1),
            'slope': dict(zlib=True, complevel=9, least_significant_digit=6),
            'measure': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

def TalwegElevationProfile(axis):
    """
    Creates a smooth talweg elevation profile
    by interpolating swath median talweg elevation values
    along reference axis

    @api   fct-corridor:talweg-profile

    @input reference_axis: ax_refaxis
    @input swath_raster: ax_valley_swaths
    @input elevation_talweg: metrics_talweg
    @param  spline_order: 3

    @output elevation_profile_talweg: ax_elevation_profile_talweg
    """

    refaxis_shapefile = config.filename('ax_refaxis', axis=axis)
    swath_raster = config.tileset().filename('ax_valley_swaths', axis=axis)
    talweg_datafile = config.filename('metrics_talweg', axis=axis)
    talweg_data = xr.open_dataset(talweg_datafile)

    swathid = np.array([], dtype='uint32')
    coordxy = np.zeros((0, 2), dtype='float32')
    coordz = np.array([], dtype='float32')
    coordm = np.array([], dtype='float32')
    slope = np.array([], dtype='float32')

    xmin = xmax = 0.0

    def interpolator(data):

        nonlocal xmin
        nonlocal xmax

        # pylint: disable=invalid-name

        w = np.isnan(data[:, 1])
        x = data[:, 0]
        y = data[:, 1]

        xmin = np.min(x[~w])
        xmax = np.max(x[~w])

        return UnivariateSpline(x[~w], y[~w], k=3)

    data_points = np.column_stack([
        talweg_data['measure'].values,
        talweg_data['talweg_elevation_median'].values
    ])
    spl = interpolator(data_points)

    with fiona.open(refaxis_shapefile) as fs:

        # assert len(fs) == 1

        for feature in fs:

            pixels = list()
            segment_m = list()

            coordinates = np.array(
                feature['geometry']['coordinates'],
                dtype='float32')

            # reverse line direction,
            # keep only x and y coords
            coordinates = coordinates[::-1, :2]

            # calculate m coord
            m0 = feature['properties'].get('M0', 0.0)
            m = m0 + np.cumsum(np.linalg.norm(
                coordinates[1:] - coordinates[:-1],
                axis=1))

            m = np.concatenate([[0], m], axis=0)
            # m = np.insert(m, 0, 0.0, axis=0)

            with rio.open(swath_raster) as ds:

                coordij = fct.worldtopixel(coordinates, ds.transform)
                coordij = np.column_stack([coordij, m])

                # interpolate (x, y, m) between vertices

                for a, b in zip(coordij[:-1], coordij[1:]):
                    for i, j, mij in rasterize_linestringz(a, b):
                        pixels.append((i, j))
                        segment_m.append(mij)

                segment_xy = fct.pixeltoworld(
                    np.array(pixels, dtype='int32'),
                    ds.transform)

                coordxy = np.concatenate([coordxy, segment_xy], axis=0)

                segment_m = np.array(segment_m, dtype='float32')
                coordm = np.concatenate([coordm, segment_m], axis=0)

                segment_swathid = np.array(list(ds.sample(segment_xy, 1)))
                swathid = np.concatenate([swathid, segment_swathid[:, 0]], axis=0)

                # with rio.open(measure_raster) as measure_ds:

                #     segment_m = np.array(list(measure_ds.sample(segment_xy, 1)))
                #     segment_m = segment_m[:, 0]
                #     coordm = np.concatenate([coordm, segment_m], axis=0)

                segment_z = spl(segment_m)
                coordz = np.concatenate([coordz, segment_z], axis=0)

                segment_slope = 100 * derivative(spl, segment_m, dx=50.0, n=1)
                slope = np.concatenate([slope, segment_slope], axis=0)

    valid = (coordm >= xmin) & (coordm <= xmax)
    coordz[~valid] = np.nan

    valley_profile = np.float32(np.column_stack([
        coordm,
        coordxy,
        coordz,
        slope
    ]))

    output = config.filename('ax_elevation_profile_talweg', axis=axis)
    ExportTalwegElevationProfile(axis, valley_profile, output)
