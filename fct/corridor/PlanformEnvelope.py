# coding: utf-8

"""
Planform (ie. meandering) Envelope

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
from scipy.interpolate import interp1d

import click
import fiona
import rasterio as rio

from .. import transform as fct
from ..rasterize import rasterize_linestringz
from ..swath.SwathMeasurement import nearest_value_and_distance
from ..config import config
from ..cli import starcall

def PlanformEnvelopeTile(axis, row, col, refpoints):

    tileset = config.tileset()
    # height_raster = tileset.tilename('ax_flow_height', axis=axis, row=row, col=col)

    def _tilename(dataset):
        return tileset.tilename(
            dataset,
            axis=axis,
            row=row,
            col=col)

    # mask_raster = _tilename('ax_nearest_height')
    mask_raster = _tilename('ax_valley_mask_refined')
    output = _tilename('ax_planform_envelope')
    # output_distance = _tilename('ax_planform_envelope_distance')
    # output_amp = _tilename('ax_planform_envelope_amp')

    if not os.path.exists(mask_raster):
        return

    with rio.open(mask_raster) as ds:

        mask = ds.read(1)
        height, width = mask.shape

        refaxis_pixels = list()

        refpoints[:, :2] = fct.worldtopixel(refpoints[:, :2], ds.transform)

        def accept(i, j):
            return all([i >= -height, i < 2*height, j >= -width, j < 2*width])

        for a, b in zip(refpoints[:-1], refpoints[1:]):
            for i, j, amp in rasterize_linestringz(a, b):
                if accept(i, j):
                    # distance[i, j] = 0
                    # measure[i, j] = m
                    refaxis_pixels.append((i, j, amp))

        if not refaxis_pixels:
            return

        amplitude, distance = nearest_value_and_distance(
            np.flip(np.array(refaxis_pixels), axis=0),
            np.float32(mask),
            ds.nodata)

        distance = 5.0 * distance

        envelope = (mask != ds.nodata) & (-amplitude <= distance) & (distance <= amplitude)

        result = np.full_like(mask, ds.nodata)
        result[envelope] = 1
        result[envelope & (mask == 2)] = 2
        result[~envelope] = 3
        result[~envelope & (mask == 2)] = 4
        result[mask == ds.nodata] = ds.nodata

        profile = ds.profile.copy()
        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(result, 1)

        # nodata = -99999.0
        # profile.update(dtype='float32', nodata=nodata)

        # amplitude[mask == ds.nodata] = nodata
        # distance[mask == ds.nodata] = nodata

        # with rio.open(output_amp, 'w', **profile) as dst:
        #     dst.write(amplitude, 1)

        # with rio.open(output_distance, 'w', **profile) as dst:
        #     dst.write(distance, 1)


def PlanformEnvelope(axis, amplitude, processes=1, **kwargs):
    """
    DOCME
    """

    # amplitude => ref measure, amplitude
    # sorted by ref measure

    refaxis_shapefile = config.filename('ax_axis_inflection', axis=axis)

    with fiona.open(refaxis_shapefile) as fs:
        refaxis = np.concatenate([f['geometry']['coordinates'] for f in fs])

    # reverse order 
    refaxis = refaxis[::-1]

    # x = refaxis measure
    x = np.cumsum(np.linalg.norm(refaxis[1:] - refaxis[:-1], axis=1))
    x = np.concatenate([np.zeros(1), x])
    # x = np.max(x) - x

    amp = interp1d(amplitude[:, 0], amplitude[:, 1], bounds_error=False, fill_value=0.0)

    refaxis = np.float32(np.column_stack([refaxis, amp(x)]))

    tilefile = config.tileset().filename('ax_shortest_tiles', axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                PlanformEnvelopeTile,
                axis,
                row,
                col,
                refaxis,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass
