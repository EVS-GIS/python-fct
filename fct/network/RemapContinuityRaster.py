# coding: utf-8

"""
Reclass landcover classes to continuity classes
"""

import os
from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio
import fiona

from ..cli import starcall
from ..config import DatasetParameter

class Parameters:
    """
    Continuity remapping parameters
    """

    tiles = DatasetParameter('domain tiles as CSV list', type='input')
    landcover = DatasetParameter('landcover raster map', type='input')
    continuity = DatasetParameter('input continuity map', type='input')
    output = DatasetParameter('output (remapped) continuity map', type='output')

    def __init__(self):
        """
        Default parameter values
        """

        self.tiles = 'shortest_tiles'
        self.landcover = 'landcover-bdt'
        self.continuity = 'continuity'
        self.output = 'continuity_remapped'


def RemapContinuityTile(row: int, col: int, params: Parameters, **kwargs):
    """
    Reclass tile
    """

    landcover_raster = params.landcover.tilename(row=row, col=col, **kwargs)
    continuity_raster = params.continuity.tilename(row=row, col=col, **kwargs)
    output = params.output.tilename(row=row, col=col, **kwargs)

    if os.path.exists(continuity_raster) and os.path.exists(landcover_raster):

        with rio.open(landcover_raster) as ds:
            landcover = ds.read(1)

        with rio.open(continuity_raster) as ds:

            profile = ds.profile.copy()

            data = ds.read(1)
            out = np.full_like(data, ds.nodata)

            out[(data == 0) | (data == 1)] = 1
            out[(data == 2) | (data == 3)] = 10
            out[(data == 4)] = 20
            out[(data == 5)] = 30
            out[(data >= 6) & (data <= 8) & (landcover >= 0) & (landcover <= 5)] = 40
            out[(data >= 6) & (data <= 8) & (landcover >= 6) & (landcover <= 8)] = 50

            # profile.update(
            #     height=height,
            #     width=width,
            #     nodata=255,
            #     dtype='uint8',
            #     transform=transform,
            #     compress='deflate'
            # )

            with rio.open(output, 'w', **profile) as dst:
                dst.write(out, 1)

def RemapContinuityRaster(params: Parameters, processes: int = 1, **kwargs):
    """
    Reclass landcover classes to continuity classes
    """

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    RemapContinuityTile,
                    row,
                    col,
                    params,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass
