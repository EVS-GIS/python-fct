# coding: utf-8

"""
LandCover Swath Profile

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
from multiprocessing import Pool
import numpy as np

import rasterio as rio
import fiona
import fiona.crs
from shapely.geometry import asShape
import click

from ..tileio import as_window
from ..cli import starcall
from ..config import config

# workdir = '/media/crousson/Backup/TESTS/TuilesAin'

def UnitLandCoverSwath(axis, gid, bounds, landcover='continuity'):
    """
    Calculate Land Cover Swath Profile for Valley Unit (axis, gid)
    """

    # dgo_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'DGO.vrt')
    # distance_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'AXIS_DISTANCE.vrt')
    # relz_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'NEAREST_RELZ.vrt')

    dgo_raster = config.filename('ax_dgo', axis=axis)
    distance_raster = config.filename('ax_axis_distance', axis=axis)
    relz_raster = config.filename('ax_relative_elevation', axis=axis)

    if landcover == 'continuity':
        # landcover_raster = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'CONTINUITY.vrt')
        landcover_raster = config.filename('ax_continuity', axis=axis)
    else:
        # landcover_raster = os.path.join(workdir, 'GLOBAL', 'LANDCOVER_2018.vrt')
        landcover_raster = config.filename('landcover')

    with rio.open(distance_raster) as ds:

        window = as_window(bounds, ds.transform)
        distance = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(relz_raster) as ds1:
            relz = ds1.read(1, window=window, boundless=True, fill_value=ds1.nodata)

        with rio.open(landcover_raster) as ds2:
            window2 = as_window(bounds, ds2.transform)
            landcover = ds2.read(1, window=window2, boundless=True, fill_value=ds2.nodata)

        with rio.open(dgo_raster) as ds3:
            mask = (ds3.read(1, window=window, boundless=True, fill_value=ds3.nodata) == gid)
            mask = mask & (relz < 20.0)

        assert(all([
            relz.shape == distance.shape,
            landcover.shape == distance.shape,
            mask.shape == distance.shape
        ]))

        if np.sum(mask) == 0:

            click.secho('No data for swath (%d, %d)' % (axis, gid), fg='yellow')
            values = dict(
                x=np.zeros(0, dtype='float32'),
                density=np.zeros(0, dtype='float32'),
                classes=np.zeros(0, dtype='uint32'),
                swath=np.zeros((0, 0), dtype='float32')
            )
            return axis, gid, values


        xbins = np.arange(np.min(distance[mask]), np.max(distance[mask]), 10.0)
        binned = np.digitize(distance, xbins)
        x = 0.5*(xbins[1:] + xbins[:-1])

        density = np.zeros_like(x, dtype='int32')

        # Profile density

        for i in range(1, len(xbins)):
            density[i-1] = np.sum(mask & (binned == i))

        # Land cover classes count

        classes = np.unique(landcover)
        swath = np.zeros((len(x), len(classes)), dtype='uint16')

        for k, value in enumerate(classes):

            data = (landcover == value)

            for i in range(1, len(xbins)):
                swath[i-1, k] = np.sum(data[mask & (binned == i)])

        values = dict(
            x=x,
            density=density,
            classes=classes,
            swath=swath
        )

        return axis, gid, values

def LandCoverSwath(axis, kind='continuity', processes=1):
    """
    DOCME
    """

    if kind not in ('std', 'continuity'):
        click.secho('Unknown landcover swath kind %s' % kind, fg='yellow')
        return

    # dgo_shapefile = os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'REF', 'DGO.shp')
    dgo_shapefile = config.filename('ax_dgo_vector', axis=axis)
    
    def output(axis, gid):
        # return os.path.join(workdir, 'AXES', 'AX%03d' % axis, 'SWATH', 'LANDCOVER', 'SWATH_LANDCOVER_%04d.npz' % gid)
        return config.filename('ax_swath_landcover', axis=axis, gid=gid, kind=kind.upper())

    kwargs = dict(landcover=kind)
    profiles = dict()
    arguments = list()

    with fiona.open(dgo_shapefile) as fs:
        for feature in fs:

            gid = feature['properties']['GID']
            measure = feature['properties']['M']
            geometry = asShape(feature['geometry'])

            profiles[axis, gid] = [axis, gid, measure]
            arguments.append([UnitLandCoverSwath, axis, gid, geometry.bounds, kwargs])

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments)

        with click.progressbar(pooled, length=len(arguments)) as iterator:

            for axis, gid, values in iterator:

                profile = profiles[axis, gid]

                np.savez(
                    output(axis, gid),
                    profile=profile,
                    **values)
