# coding: utf-8

"""
Valley Elevation Profile

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
# from operator import itemgetter
# import itertools

import numpy as np
# from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.misc import derivative

import click
import xarray as xr
import rasterio as rio
import fiona

from .. import transform as fct
from ..rasterize import rasterize_linestringz
from ..config import config
from ..metadata import set_metadata

def ExportValleyProfile(axis, valley_profile, destination):
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
            'x': ('measure', valley_profile[:, 1]),
            'y': ('measure', valley_profile[:, 2]),
            'z': ('measure', valley_profile[:, 3]),
            'slope': ('measure', valley_profile[:, 4])
        },
        coords={
            'axis': axis,
            'measure': valley_profile[:, 0]
        })

    set_metadata(dataset, 'profile_elevation_floodplain')

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

def ValleySwathElevation(axis):
    """
    Calculate median talweg height relative to valley floor
    for each valley swath defined in ax_swaths_refaxis
    """

    swath_defs = config.filename('ax_swaths_refaxis_bounds', axis=axis)

    # swath => z0, slope

    defs = xr.open_dataset(swath_defs)
    defs.load()
    defs = defs.sortby('measure')

    swids = list()
    values = list()

    with click.progressbar(defs['swath'].values) as iterator:
        for gid in iterator:

            filename = config.filename('ax_swath_elevation_npz', axis=axis, gid=gid)

            if os.path.exists(filename):

                data = np.load(filename, allow_pickle=True)
                z0 = data['z0_valley_floor']
                slope = data['slope_valley_floor']

                if not (np.isnan(z0) or np.isnan(slope)):

                    coordm = defs['measure'].sel(swath=gid).values
                    zvalley = slope*coordm + z0

                    swids.append(gid)
                    values.append((coordm, zvalley))

                else:

                    swids.append(gid)
                    values.append((coordm, np.nan))

    return (
        np.array(swids, dtype='uint32'),
        np.array(values, dtype='float32')
    )

def ValleyElevationProfile(axis):
    """
    Creates a smooth valley elevation profile
    by interpolating idealized floodplain elevation values
    along reference axis

    @api    fct-corridor:valley-profile

    @input  reference_axis: ax_refaxis
    @input  swath_bounds: ax_swaths_refaxis_bounds
    @input  swath_raster: ax_swaths_refaxis
    @input  elevation_floodplain: ax_swath_elevation_npz
    @param  spline_order: 3

    @output elevation_profile_floodplain: ax_elevation_profile_floodplain
    """

    refaxis_shapefile = config.filename('ax_refaxis', axis=axis)
    swath_raster = config.tileset().filename('ax_swaths_refaxis', axis=axis)
    # measure_raster = config.tileset().filename('ax_axis_measure', axis=axis)

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

    _, zvalley = ValleySwathElevation(axis)
    spl = interpolator(zvalley)

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

    output = config.filename('ax_elevation_profile_floodplain', axis=axis)
    ExportValleyProfile(axis, valley_profile, output)

# def interpolate():

#     from fct.corridor.ValleyElevation import ValleyElevation, config
#     config.default()
#     swaths, data = ValleyElevation(1)

#     import matplotlib.pyplot as plt
#     import seaborn as sbn
#     from scipy.interpolate import UnivariateSpline
#     import numpy as np

#     w = np.isnan(data[:, 1])

#     x = data[:, 0]
#     y = data[:, 1]

#     spl = UnivariateSpline(x[~w], y[~w], k=3)

#     xmax = np.max(data[:, 0])
#     plt.plot(xmax - x, y, linewidth=0.5)
#     # plt.plot(xmax - x[swaths < 3440], spl(x[swaths < 3440]))
#     plt.plot(xmax - x, spl(x))
#     plt.show()

# def test_piecewise_regression(data):

#     from sklearn.preprocessing import KBinsDiscretizer
#     import matplotlib.pyplot as plt
#     from sklearn.model_selection import train_test_split
#     from mlinsights.mlmodel import PiecewiseRegressor
#     from sklearn.tree import DecisionTreeRegressor

#     model = PiecewiseRegressor(
#         verbose=True,
#         binner=KBinsDiscretizer(n_bins=10))

#     X_train = np.log(100 + np.max(data[:, 0]) - data[:, :1])
#     y_train = data[:, 1]
#     model.fit(X_train, y_train)
#     pred = model.predict(X_train)
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(np.max(data[:, 0]) - data[:, :1], y_train, ".", label='data')
#     ax.plot(np.max(data[:, 0]) - data[:, :1], pred, ".", label="predictions")
#     ax.set_title("Piecewise Linear Regression\n4 buckets")
#     ax.legend()
#     plt.show()

#     model = PiecewiseRegressor(
#         verbose=True,
#         binner=DecisionTreeRegressor(min_samples_leaf=300))