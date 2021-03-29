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

LANDCOVER_WATER = 0
LANDCOVER_GRAVELS = 1
LANDCOVER_OPEN_VEGETATION = 2
LANDCOVER_FORESTED = 3
LANDCOVER_MEADOWS = 4
LANDCOVER_CROPS = 5
LANDCOVER_URBAN_DIFFUSE = 6
LANDCOVER_BUILT = 7
LANDCOVER_INFRASTRUCTURES = 8
LANDCOVER_NODATA = 255

CONTINUITY_WATER_CHANNEL = 0
CONTINUITY_ACTIVE_CHANNEL = 1
CONTINUITY_RIPARIAN_BUFFER = 10
CONTINUITY_CONNECTED_MEADOWS = 20
CONTINUITY_CONNECTED_CULTIVATED = 30
CONTINUITY_DISCONNECTED = 40
CONTINUITY_BUILT = 50
CONTINUITY_NODATA = 255

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
            out = np.full_like(data, CONTINUITY_NODATA)

            out[
                data == LANDCOVER_WATER
            ] = CONTINUITY_WATER_CHANNEL

            out[
                data == LANDCOVER_GRAVELS
            ] = CONTINUITY_ACTIVE_CHANNEL

            out[
                (data == LANDCOVER_OPEN_VEGETATION) |
                (data == LANDCOVER_FORESTED)
            ] = CONTINUITY_RIPARIAN_BUFFER

            out[
                (data == LANDCOVER_MEADOWS)
            ] = CONTINUITY_CONNECTED_MEADOWS

            out[
                (data == LANDCOVER_CROPS)
            ] = CONTINUITY_CONNECTED_CULTIVATED

            out[
                (data >= LANDCOVER_URBAN_DIFFUSE) &
                (data <= LANDCOVER_INFRASTRUCTURES) &
                (landcover >= LANDCOVER_WATER) &
                (landcover <= LANDCOVER_CROPS)
            ] = CONTINUITY_DISCONNECTED

            out[
                (data >= LANDCOVER_URBAN_DIFFUSE) &
                (data <= LANDCOVER_INFRASTRUCTURES) &
                (landcover >= LANDCOVER_URBAN_DIFFUSE) &
                (landcover <= LANDCOVER_INFRASTRUCTURES)
            ] = CONTINUITY_BUILT

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
