# coding: utf-8

"""
Continuity Mask - WIP

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

import click
import rasterio as rio
import fiona
import xarray as xr

from .. import speedup
from ..tileio import as_window
from ..config import config
from ..cli import starcall

def ContinuityMaskTile(axis, row, col):
    """
    Extract valley bottom in spatial continuity with talweg
    within valley swath units
    """

    tileset = config.tileset()

    def _rasterfile(name):
        return tileset.filename(name, axis=axis)

    def _tilename(name):
        return tileset.tilename(name, axis=axis, row=row, col=col)

    output = _tilename('ax_valley_height_mask_c0')
    mask_tile = _tilename('ax_valley_height_mask')
    swath_tile = _tilename('ax_valley_swaths')

    mask_raster = _rasterfile('ax_valley_height_mask')
    swath_raster = _rasterfile('ax_valley_swaths')
    talweg_dist_raster = _rasterfile('ax_talweg_distance')
    # continuity_raster = _rasterfile('ax_continuity')

    filename = config.filename('ax_valley_swaths_defs', axis=axis)
    swath_defs = xr.open_dataset(filename)

    if not (os.path.exists(mask_tile) and os.path.exists(swath_tile)):
        # click.echo('missing tile (%d, %d)' % (row, col))
        return

    with rio.open(mask_tile) as ds:

        out = ds.read(1)

        height, width = out.shape
        profile = ds.profile.copy()
        # nodata = ds.nodata
        transform = ds.transform

    with rio.open(swath_tile) as ds:

        swaths = ds.read(1)
        swath_nodata = ds.nodata

    for swid in np.unique(swaths):

        if swid == swath_nodata:
            continue

        bounds = tuple(swath_defs['bounds'].sel(label=swid).values)
        tile_window = as_window(bounds, transform)

        if tile_window.height == 0 or tile_window.width == 0:
            continue

        with rio.open(mask_raster) as ds:

            window = as_window(bounds, ds.transform)
            bottom_mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(swath_raster) as ds:

            window = as_window(bounds, ds.transform)
            swath_mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(talweg_dist_raster) as ds:

            window = as_window(bounds, ds.transform)
            talweg_dist = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        # with rio.open(continuity_raster) as ds:

        #     window = as_window(bounds, ds.transform)
        #     continuity_mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        state = np.full_like(bottom_mask, 255, dtype='uint8')
        mask = (swath_mask == swid) & (bottom_mask == 0)
        state[mask] = 0
        state[mask & (talweg_dist == 0)] = 1

        distance = np.zeros_like(state, dtype='float32')

        speedup.continuity_mask(
            state,
            # out,
            distance,
            jitter=0.4)

        # reclass discontinuous parts of valley bottom
        bottom_mask[mask & (state == 0)] = 1

        # window copy to out

        mini = max(0, tile_window.row_off)
        minj = max(0, tile_window.col_off)
        maxi = min(height - 1, tile_window.row_off + tile_window.height - 1)
        maxj = min(width - 1, tile_window.col_off + tile_window.width - 1)

        wmini = mini - tile_window.row_off
        wminj = minj - tile_window.col_off
        # wmaxi = min(maxi - tile_window.row_off, tile_window.height - 1)
        # wmaxj = min(maxj - tile_window.col_off, tile_window.width - 1)
        wmaxi = maxi - tile_window.row_off
        wmaxj = maxj - tile_window.col_off

        if maxi < 0 or maxj < 0:
            continue

        try:

            assert 0 <= mini <= maxi < height, '%d, %d' % (mini, maxi)
            assert 0 <= minj <= maxj < width, '%d, %d' % (minj, maxj)
            assert 0 <= wmini <= wmaxi < tile_window.height, '%d, %d' % (wmini, wmaxi)
            assert 0 <= wminj <= wmaxj < tile_window.width, '%d, %d' % (wminj, wmaxj)

        except AssertionError as error:

            print(height, width, bottom_mask.shape, tile_window)
            print(mini, minj, maxi, maxj)
            print(wmini, wminj, wmaxi, wmaxj)
            raise error

        out_win = out[mini:maxi+1, minj:maxj+1]
        swath_mask_win = swath_mask[wmini:wmaxi+1, wminj:wmaxj+1]
        bottom_mask_win = bottom_mask[wmini:wmaxi+1, wminj:wmaxj+1]
        mask_win = (swath_mask_win == swid) & (bottom_mask_win != 255)
        out_win[mask_win] = bottom_mask_win[mask_win]

    with rio.open(output, 'w', **profile) as dst:
        dst.write(out, 1)

def TestSwath(axis, swid):

    tileset = config.tileset()

    def _rasterfile(name):
        return tileset.filename(name, axis=axis)

    # def _tilename(name):
    #     return tileset.tilename(name, axis=axis, row=row, col=col)

    # output = _tilename('ax_valley_height_mask_c0')
    # mask_tile = _tilename('ax_valley_height_mask')
    # swath_tile = _tilename('ax_valley_swaths')

    mask_raster = _rasterfile('ax_valley_height_mask')
    swath_raster = _rasterfile('ax_valley_swaths')
    talweg_dist_raster = _rasterfile('ax_talweg_distance')
    # continuity_raster = _rasterfile('ax_continuity')

    filename = config.filename('ax_valley_swaths_defs', axis=axis)
    swath_defs = xr.open_dataset(filename)

    bounds = tuple(swath_defs['bounds'].sel(label=swid).values)
    # tile_window = as_window(bounds, transform)

    # if tile_window.height == 0 or tile_window.width == 0:
    #     return

    with rio.open(mask_raster) as ds:

        window = as_window(bounds, ds.transform)
        bottom_mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        transform = ds.transform * ds.transform.translation(window.col_off, window.row_off)
        height, width = bottom_mask.shape

        profile = ds.profile.copy()
        profile.update(
            transform=transform,
            driver='GTiff',
            height=height,
            width=width)

    with rio.open(swath_raster) as ds:

        window = as_window(bounds, ds.transform)
        swath_mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    with rio.open(talweg_dist_raster) as ds:

        window = as_window(bounds, ds.transform)
        talweg_dist = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    # with rio.open(continuity_raster) as ds:

    #     window = as_window(bounds, ds.transform)
    #     continuity_mask = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

    state = np.full_like(bottom_mask, 255, dtype='uint8')
    mask = (swath_mask == swid) & (bottom_mask == 0)
    state[mask] = 0
    state[mask & (talweg_dist == 0)] = 1

    distance = np.zeros_like(state, dtype='float32')

    speedup.continuity_mask(
        state,
        # out,
        distance,
        jitter=0.4)

    # reclass discontinuous parts of valley bottom
    bottom_mask[~mask] = 255
    bottom_mask[mask & (state == 0)] = 1

    output = os.path.join(config.workdir, 'TEST', 'VALLEY_MASK_SWATH_%d.tif' % swid)
    with rio.open(output, 'w', **profile) as dst:
        dst.write(bottom_mask, 1)

    # window copy to out

    # mini = max(0, tile_window.row_off)
    # minj = max(0, tile_window.col_off)
    # maxi = min(height - 1, tile_window.row_off + tile_window.height - 1)
    # maxj = min(width - 1, tile_window.col_off + tile_window.width - 1)

    # wmini = mini - tile_window.row_off
    # wminj = minj - tile_window.col_off
    # # wmaxi = min(maxi - tile_window.row_off, tile_window.height - 1)
    # # wmaxj = min(maxj - tile_window.col_off, tile_window.width - 1)
    # wmaxi = maxi - tile_window.row_off
    # wmaxj = maxj - tile_window.col_off

    # if maxi < 0 or maxj < 0:
    #     continue

    # try:

    #     assert 0 <= mini <= maxi < height, '%d, %d' % (mini, maxi)
    #     assert 0 <= minj <= maxj < width, '%d, %d' % (minj, maxj)
    #     assert 0 <= wmini <= wmaxi < tile_window.height, '%d, %d' % (wmini, wmaxi)
    #     assert 0 <= wminj <= wmaxj < tile_window.width, '%d, %d' % (wminj, wmaxj)

    # except AssertionError as error:

    #     print(height, width, bottom_mask.shape, tile_window)
    #     print(mini, minj, maxi, maxj)
    #     print(wmini, wminj, wmaxi, wmaxj)
    #     raise error

    # out[mini:maxi+1, minj:maxj+1] = bottom_mask[wmini:wmaxi+1, wminj:wmaxj+1]

def ContinuityMask(axis, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):
    """
    Refine valley bottom mask
    0: bottom
    1: margin/hole => separate with exterior region algorithm
    """

    tilefile = config.tileset().filename(ax_tiles, axis=axis, **kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:
                row, col = tuple(int(x) for x in line.split(','))
                yield (
                    ContinuityMaskTile,
                    axis,
                    row,
                    col,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass
