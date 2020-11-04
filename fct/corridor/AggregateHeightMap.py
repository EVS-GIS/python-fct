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
from ..config import config

def AggregateTile(name, ax_name, row, col, axes, **kwargs):

    output = config.tileset().tilename(name, row=row, col=col, **kwargs)

    aggregate = None
    profile = None
    nodata = None

    # if os.path.exists(output):

    #     with rio.open(output) as ds:

    #         aggregate = ds.read(1)
    #         profile = ds.profile.copy()

    for axis in axes:

        axtile = config.tileset().tilename(ax_name, axis=axis, row=row, col=col, **kwargs)

        if os.path.exists(axtile):

            with rio.open(axtile) as ds:

                data = ds.read(1)

                if aggregate is None:

                    aggregate = data
                    nodata = ds.nodata
                    profile = ds.profile.copy()

                else:

                    lower = (
                        (data != ds.nodata) &
                        (
                            (aggregate != nodata) &
                            (np.abs(data) < np.abs(aggregate)) |
                            (aggregate == nodata)
                        )
                    )

                    aggregate[lower] = data[lower]

    if aggregate is not None:

        profile.update(compress='deflate')

        with rio.open(output, 'w', **profile) as dst:
            dst.write(aggregate, 1)


def AggregateHeightMap(axes, datasets=('nearest_height', 'ax_nearest_height'), processes=1, **kwargs):

    tileset = config.tileset()
    name, ax_name = datasets

    def arguments():

        for tile in tileset.tiles():
            yield (AggregateTile, name, ax_name, tile.row, tile.col, axes, kwargs)

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=len(tileset)) as iterator:
            for _ in iterator:
                pass
