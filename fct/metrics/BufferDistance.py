#!/usr/bin/env python
# coding: utf-8

"""
Raster buffer around stream active channel

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
from operator import itemgetter
import math
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
import fiona
from shapely.geometry import asShape

from .. import transform as fct
from .. import speedup
from ..tileio import (
    as_window,
    grow_window
)
from ..config import config
from ..cli import starcall
from ..rasterize import rasterize_linestringz
from .SpatialReferencing import nearest_value_and_distance

def BufferDistanceTile(
        axis,
        tile,
        buffer_width,
        padding=200,
        fill=2,
        tileset='default',
        dataset='ax_continuity',
        resolution=5.0):
    """
    Calculate a buffer area and distance from river active channel
    from landcover continuity raster.

    Output will have :
      1 = active channel
      2 (fill) = buffer area,
                 ie. space within buffer width from active channel
      0 = elsewhere (nodata)

    The calculated buffer is not limited to landcover data
    and may extend outside landcover data
    """

    rasterfile = config.filename(dataset, axis=axis)

    output_mask = config.tileset(tileset).tilename(
        'ax_buffer_mask',
        axis=axis,
        row=tile.row,
        col=tile.col)

    output_distance = config.tileset(tileset).tilename(
        'ax_buffer_distance',
        axis=axis,
        row=tile.row,
        col=tile.col)

    with rio.open(rasterfile) as ds:

        window = as_window(tile.bounds, ds.transform)
        landcover = ds.read(
            1,
            window=grow_window(window, padding),
            boundless=True,
            fill_value=ds.nodata)

        transform = ds.transform * ds.transform.translation(
            window.col_off,
            window.row_off)

        # Define a mask from active channel landcover classes
        mask = np.float32((landcover == 0) | (landcover == 1))

        distance = resolution * speedup.raster_buffer(
            mask,
            0.0,
            buffer_width / resolution,
            fill)

        mask = np.uint8(mask)
        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            dtype='uint8',
            nodata=0,
            compress='deflate',
            height=window.height,
            width=window.width,
            transform=transform
        )

        with rio.open(output_mask, 'w', **profile) as dst:
            dst.write(mask[padding:-padding, padding:-padding], 1)

        distance_nodata = -99999.0
        distance[mask == 0] = distance_nodata

        profile = ds.profile.copy()
        profile.update(
            driver='GTiff',
            dtype='float32',
            nodata=distance_nodata,
            compress='deflate',
            height=window.height,
            width=window.width,
            transform=transform
        )

        with rio.open(output_distance, 'w', **profile) as dst:
            dst.write(distance[padding:-padding, padding:-padding], 1)

def BufferDistance(axis, buffer_width, processes=1, **kwargs):
    """
    Calculate a buffer area and distance from river active channel
    from landcover continuity raster.
    """

    tileindex = config.tileset('landcover').tileindex
    tilefile = config.filename('ax_tiles', axis=axis)

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    def arguments():

        for _, row, col in tiles:
            # if (row, col) in tileindex:
            tile = tileindex[row, col]
            yield (
                BufferDistanceTile,
                axis,
                tile,
                buffer_width,
                kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tiles)) as iterator:
            for _ in iterator:
                pass

def BufferMeasureTile(axis, row, col, mdelta=200.0):
    """
    see SpatialReference
    """

    tileset = config.tileset('landcover')
    rasterfile = tileset.tilename('ax_buffer_distance', axis=axis, row=row, col=col)
    refaxis_shapefile = config.filename('ax_refaxis', axis=axis)
    output = tileset.tilename('ax_buffer_profile', axis=axis, row=row, col=col)

    with rio.open(rasterfile) as ds:

        domain = ds.read(1)
        height, width = domain.shape
        refaxis_pixels = list()

        def accept(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        coord = itemgetter(0, 1)

        mmin = float('inf')
        mmax = float('-inf')

        with fiona.open(refaxis_shapefile) as fs:
            for feature in fs:

                m0 = feature['properties'].get('M0', 0.0)
                length = asShape(feature['geometry']).length

                if m0 < mmin:
                    mmin = m0

                if m0 + length > mmax:
                    mmax = m0 + length

                coordinates = np.array([
                    coord(p) + (m0,) for p in reversed(feature['geometry']['coordinates'])
                ], dtype='float32')

                coordinates[1:, 2] = m0 + np.cumsum(np.linalg.norm(
                    coordinates[1:, :2] - coordinates[:-1, :2],
                    axis=1))

                coordinates[:, :2] = fct.worldtopixel(coordinates[:, :2], ds.transform)

                for a, b in zip(coordinates[:-1], coordinates[1:]):
                    for i, j, m in rasterize_linestringz(a, b):
                        if accept(i, j):
                            refaxis_pixels.append((i, j, m))

        if not refaxis_pixels:
            return []

        mmin = math.floor(mmin / mdelta) * mdelta
        mmax = math.ceil(mmax / mdelta) * mdelta

        # Nearest using KD Tree

        measure, distance = nearest_value_and_distance(
            np.flip(np.array(refaxis_pixels), axis=0),
            domain,
            ds.nodata)

        distance = 5.0 * distance
        distance[domain == ds.nodata] = ds.nodata
        measure[domain == ds.nodata] = ds.nodata

        breaks = np.arange(mmin, mmax, mdelta)
        spatial_units = np.uint32(np.digitize(measure, breaks))
        # unit_measures = np.round(0.5 * (breaks + np.roll(breaks, 1)), 1)

        profile = ds.profile.copy()
        profile.update(
            nodata=0,
            dtype='uint32',
            compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(spatial_units, 1)

        return mmin, mmax

def BufferMeasure(axis, mdelta=200.0, processes=1, **kwargs):
    """
    Discretize buffer area along reference axis
    """

    tileindex = config.tileset('landcover').tileindex
    tilefile = config.filename('ax_tiles', axis=axis)

    kwargs.update(mdelta=mdelta)

    with open(tilefile) as fp:
        tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

    def arguments():

        for _, row, col in tiles:
            # if (row, col) in tileindex:
            # tile = tileindex[row, col]
            yield (
                BufferMeasureTile,
                axis,
                row,
                col,
                kwargs)

    mmin = float('inf')
    mmax = float('-inf')

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tiles)) as iterator:
            for t_mmin, t_mmax in iterator:

                mmin = min(mmin, t_mmin)
                mmax = max(mmax, t_mmax)

    breaks = np.arange(mmin, mmax, mdelta)
    measures = np.round(0.5 * (breaks + np.roll(breaks, 1)), 1)
    return measures
