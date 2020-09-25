# coding: utf-8

"""
Valley Bottom Delineation Refinement Procedure

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
import xarray as xr
import rasterio as rio
from rasterio import features
import fiona

from .. import speedup
from ..tileio import as_window
from ..config import config
from ..cli import starcall

def ValleyMaskTile(axis, row, col, threshold):

    tileset = config.tileset()

    def _tilename(name):
        return tileset.tilename(name, axis=axis, row=row, col=col)

    datafile = config.filename('metrics_talweg_height', axis=axis)
    hand_raster = _tilename('ax_nearest_height')
    swath_raster = _tilename('ax_valley_swaths')
    output_mask = _tilename('ax_valley_mask_refined')
    # output_height = _tilename('ax_nearest_height_refined')

    if not (os.path.exists(hand_raster) and os.path.exists(swath_raster)):
        return

    data = xr.open_dataset(datafile)

    with rio.open(swath_raster) as ds:
        swaths = ds.read(1)
        swath_nodata = ds.nodata

    with rio.open(hand_raster) as ds:

        hand = ds.read(1)

        nodata = 255
        out = np.full_like(hand, nodata, dtype='uint8')

        for swid in np.unique(swaths):

            if swid == swath_nodata:
                continue

            try:
                talheight = data['hmed'].sel(swath=swid).values
            except KeyError:
                talheight = np.nan

            if np.isnan(talheight):

                swath_mask = (swaths == swid)
                out[swath_mask] = 0

            else:

                # TODO threshold = f(swid, bottom width, drainage area)

                minh = min(-talheight - threshold, -threshold)
                maxh = max(-talheight + threshold, threshold)

                swath_mask = (swaths == swid)
                bottom_mask = (hand >= minh) & (hand < maxh)

                out[swath_mask] = 1
                out[swath_mask & bottom_mask] = 0

        out = features.sieve(out, 100) # TODO externalize parameter
        speedup.reclass_margin(out, 1, 255, 2)

        profile = ds.profile.copy()
        profile.update(dtype='uint8', nodata=nodata, compress='deflate')

        with rio.open(output_mask, 'w', **profile) as dst:
            dst.write(out, 1)

def ValleyMask(axis, threshold, ax_tiles='ax_shortest_tiles', processes=1, **kwargs):
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
                    ValleyMaskTile,
                    axis,
                    row,
                    col,
                    threshold,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

def ReclassSwathMargin(axis, row, col, **kwargs):

    tileset = config.tileset()

    def _tilename(name):
        return tileset.tilename(name, axis=axis, row=row, col=col)

    swath_raster = tileset.filename('ax_valley_swaths', axis=axis)
    bottom_raster = tileset.filename('ax_valley_mask_refined', axis=axis)
    swath_raster_tile = _tilename('ax_valley_swaths')
    bottom_raster_tile = _tilename('ax_valley_mask_refined')
    swath_bounds_file = config.filename('ax_valley_swaths_bounds', axis=axis)
    swath_bounds = xr.open_dataset(swath_bounds_file)

    if not os.path.exists(swath_raster_tile):
        return

    with rio.open(swath_raster_tile) as ds:
        swaths = ds.read(1)
        swath_nodata = ds.nodata

    with rio.open(bottom_raster_tile) as ds:
        out = ds.read(1)
        profile = ds.profile.copy()
        height, width = out.shape
        transform = ds.transform

    for swid in np.unique(swaths):

        if swid == swath_nodata:
            continue

        bounds = tuple(swath_bounds['bounds'].sel(label=swid).values)

        with rio.open(swath_raster) as ds:
            window = as_window(bounds, ds.transform)
            swath = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        with rio.open(bottom_raster) as ds:
            window = as_window(bounds, ds.transform)
            bottom = ds.read(1, window=window, boundless=True, fill_value=ds.nodata)

        # bottom[(bottom == 1) & (swath != swid)] = 255
        speedup.reclass_margin(bottom, 1, 255, 2)

        # window copy bottom to out

        tile_window = as_window(bounds, transform)

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

            print(height, width, bottom.shape, tile_window)
            print(mini, minj, maxi, maxj)
            print(wmini, wminj, wmaxi, wmaxj)
            raise error

        out_win = out[mini:maxi+1, minj:maxj+1]
        swath_win = swath[wmini:wmaxi+1, wminj:wmaxj+1]
        bottom_win = bottom[wmini:wmaxi+1, wminj:wmaxj+1]
        mask_win = (swath_win == swid)
        out_win[mask_win] = bottom_win[mask_win]

    with rio.open(bottom_raster_tile + '.tmp', 'w', **profile) as dst:
        dst.write(out, 1)

    return bottom_raster_tile + '.tmp'

def ReclassMargin(axis, processes=1, ax_tiles='ax_shortest_tiles', **kwargs):
    """
    DOCME
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
                    ReclassSwathMargin,
                    axis,
                    row,
                    col,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        tmpfiles = list()

        with click.progressbar(pooled, length=length()) as iterator:
            for tmpfile in iterator:
                if tmpfile is not None:
                    tmpfiles.append(tmpfile)

        for tmpfile in tmpfiles:
            os.rename(tmpfile, tmpfile.replace('.tif.tmp', '.tif'))
