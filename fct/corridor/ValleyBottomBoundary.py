# coding: utf-8

"""
Valley Bottom Medial Axis

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
from multiprocessing import Pool

import numpy as np
import xarray as xr
import click

import rasterio as rio
from rasterio import features
import fiona

from ..config import config
from ..tileio import as_window
from ..cli import starcall

def unproject(axis, points):

    # medialaxis_shapefile = config.filename('ax_valley_medialaxis', axis=axis)
    medialaxis_shapefile = config.filename('ax_talweg', axis=axis)
    measure_raster = config.tileset().filename('ax_axis_measure', axis=axis)

    with fiona.open(medialaxis_shapefile) as fs:

        assert len(fs) == 1

        for feature in fs:

            coordinates = np.array(
                feature['geometry']['coordinates'],
                dtype='float32')

            # reverse line direction,
            # keep only x and y coords
            coordinates = coordinates[::-1, :2]

            with rio.open(measure_raster) as ds:
                m = np.array(list(ds.sample(coordinates, 1)))

            # calculate m coord
            # m0 = feature['properties'].get('M0', 0.0)
            # m = m0 + np.cumsum(np.linalg.norm(
            #     coordinates[1:] - coordinates[:-1],
            #     axis=1))

            # m = np.concatenate([[0], m], axis=0)

    transformed = np.zeros_like(points)

    for i, (mi, di) in enumerate(points):

        if np.isnan(di):

            transformed[i] = np.nan
            continue

        for k, mk in enumerate(m[:-1]):

            if mk <= mi < m[k+1]:
                break

        else:
            
            transformed[i] = np.nan
            continue

        # p0 between vertices k and k+1

        lenk = m[k+1] - mk
        dirk = (coordinates[k+1] - coordinates[k]) / lenk
        pti = coordinates[k] + (mi - mk) * dirk
        xi = pti[0] + di * dirk[1]
        yi = pti[1] - di * dirk[0]

        transformed[i] = (xi, yi)

    return transformed

def ValleyBottomBoundary(axis):
    """
    Calculate valley bottom pseudo boundary
    by offseting valley medial axis
    by measured left and right valley widths
    """

    data_file = config.filename('metrics_valleybottom_width', variant='TOTAL_BDT', axis=axis)
    medialaxis_shapefile = config.filename('ax_valley_medialaxis', axis=axis)
    output_shapefile = config.filename('ax_valley_bottom_boundary', axis=axis)

    data = xr.open_dataset(data_file).sortby('measure')
    data = data.rolling(measure=5, min_periods=1, center=True).mean()

    left_points = np.column_stack([
        data['measure'].values,
        data['vbw'] * data['vbalr'].sel(side='left') / np.sum(data['vbalr'], axis=1)
    ])
    left_side_coordinates = unproject(axis, left_points)

    right_points = np.column_stack([
        data['measure'].values,
        -data['vbw'] * data['vbalr'].sel(side='right') / np.sum(data['vbalr'], axis=1)
    ])
    right_side_coordinates = unproject(axis, right_points)


    with fiona.open(medialaxis_shapefile) as fs:

        driver = fs.driver
        crs = fs.crs

    schema = {
        'geometry': 'LineString',
        'properties': [
            ('AXIS', 'int'),
            ('SIDE', 'str:5')
        ]
    }

    options = dict(driver=driver, crs=crs, schema=schema)

    with fiona.open(output_shapefile, 'w', **options) as fst:

        for transformed, side in zip(
                [left_side_coordinates, right_side_coordinates],
                ['left', 'right']
            ):

            coordinates = transformed[::-1].tolist()
            feature = {
                'geometry': {'type': 'LineString', 'coordinates': coordinates},
                'properties': {'AXIS': axis, 'SIDE': side}
            }

            fst.write(feature)
