# coding: utf-8

"""
Aggregate Continuity Map Raster

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
from ..cli import starcall
from ..config import DatasetParameter

class Parameters:

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')

    height = DatasetParameter(
        'height raster (HAND)',
        type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.height = 'nearest_height'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.height = dict(key='ax_nearest_height', axis=axis)

def AggregateTile(
        row: int,
        col: int,
        src1: Parameters,
        src2: Parameters,
        out: Parameters,
        **kwargs):

    raster1 = src1.height.tilename(row=row, col=col, **kwargs)
    raster2 = src2.height.tilename(row=row, col=col, **kwargs)
    output = out.height.tilename(row=row, col=col, **kwargs)

    if not (raster1.exists() and raster2.exists()):

        def copy(raster, output):

            with rio.open(raster) as ds:

                data = ds.read(1)
                profile = ds.profile.copy()
                profile.update(compress='deflate')

            with rio.open(output, 'w', **profile) as dst:
                dst.write(data, 1)

        if raster1.exists() and not (output.exists() and raster1.samefile(output)):
            copy(raster1, output)

        if raster2.exists() and not (output.exists() and raster2.samefile(output)):
            copy(raster2, output)

        return

    # if os.path.exists(output):

    #     with rio.open(output) as ds:

    #         aggregate = ds.read(1)
    #         profile = ds.profile.copy()

    with rio.open(raster1) as ds:
        
        data1 = ds.read(1)
        nodata1 = ds.nodata

    with rio.open(raster2) as ds:

        data2 = ds.read(1)
        nodata2 = ds.nodata
        profile = ds.profile.copy()
        profile.update(compress='deflate')

        lower = (
            (data1 != nodata1) &
            (
                (data2 != nodata2) &
                (np.abs(data1) < np.abs(data2)) |
                (data2 == nodata2)
            )
        )

    data2[lower] = data1[lower]

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data2, 1)

def AggregateHeightMap(
        src1: Parameters,
        src2: Parameters,
        output: Parameters,
        processes: int = 1,
        **kwargs):

    tilefile1 = src1.tiles.filename(**kwargs)
    tilefile2 = src2.tiles.filename(**kwargs)

    def get_tiles():

        with open(tilefile1) as fp:
            for line in fp:

                yield tuple(int(x) for x in line.split(','))

        with open(tilefile2) as fp:
            for line in fp:

                yield tuple(int(x) for x in line.split(','))

    tiles = set(get_tiles())

    def arguments():

        for row, col in tiles:

            yield (
                AggregateTile,
                row,
                col,
                src1,
                src2,
                output,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tiles)) as iterator:
            for _ in iterator:
                pass
