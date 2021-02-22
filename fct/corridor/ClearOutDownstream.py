"""
Remove the overlapping domain from downstream axis
"""

import os
from multiprocessing import Pool
import click
import rasterio as rio

from ..config import (
    DatasetParameter,
    LiteralParameter
)
from ..cli import starcall

class Parameters:
    """
    Parameters for removing the overlapping domain from downstream axis
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')
    data = DatasetParameter(
        'input raster dataset',
        type='input')
    other_measure = DatasetParameter(
        'location along other stream measure axis (raster)',
        type='input')
    other_distance = DatasetParameter(
        'distance from other stream drainage axis (raster)',
        type='input')
    output = DatasetParameter(
        'output dataset, which is a cropped version of `data`. May be the same as data to override input.',
        type='output')

    other_side = LiteralParameter(
        'side of other stream to preserve')
    other_measure_min = LiteralParameter(
        'minimum measure of other stream section to preserve')
    other_measure_max = LiteralParameter(
        'maximum measure of other stream section to preserve')

    def __init__(self):
        """
        Default parameter values
        """

        self.tiles = 'ax_shortest_tiles'
        self.data = 'ax_valley_mask'
        self.other_measure = 'ax_axis_measure'
        self.other_distance = 'ax_nearest_distance'
        self.output = 'ax_valley_mask'

        self.other_side = 'left'
        self.other_measure_min = 0.0
        self.other_measure_max = float('inf')

def ClearOutDownstreamTile(ax1: int, ax2: int, row: int, col: int, params: Parameters):
    """
    Tile procedure
    """

    data_raster = params.data.tilename(axis=ax1, row=row, col=col)
    other_measure_raster = params.other_measure.tilename(axis=ax2, row=row, col=col)
    other_distance_raster = params.other_distance.tilename(axis=ax2, row=row, col=col)
    output = params.output.tilename(axis=ax1, row=row, col=col)

    side = params.side
    measure_min = params.other_measure_min
    measure_max = params.other_measure_max

    if os.path.exists(other_measure_raster) and os.path.exists(other_distance_raster):

        with rio.open(other_measure_raster) as ds:

            other_measure = ds.read(1)
            other_measure_nodata = ds.nodata

        with rio.open(other_distance_raster) as ds:

            other_distance = ds.read(1)
            other_distance_nodata = ds.nodata

        with rio.open(data_raster) as ds:

            data = ds.read(1)
            profile = ds.profile.copy()

            if side == 'left':

                overlap_mask = (
                    (other_distance != other_distance_nodata) &
                    (other_distance < 0)
                )

            else:

                overlap_mask = (
                    (other_distance != other_distance_nodata) &
                    (other_distance > 0)
                )

            overlap_mask = (
                overlap_mask &
                (other_measure != other_measure_nodata) &
                (
                    (other_measure < measure_min) |
                    (other_measure > measure_max)
                )
            )

            data[overlap_mask] = ds.nodata

    else:

        with rio.open(data_raster) as ds:

            data = ds.read(1)
            profile = ds.profile.copy()

    profile.update(compress='deflate')

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def ClearOutDownstream(
        axis: int,
        other_axis: int,
        params: Parameters,
        processes: int = 1,
        **kwargs):
    """
    Remove the overlapping domain from downstream axis
    """

    tilefile = params.tiles.filename(axis=axis)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                ClearOutDownstreamTile,
                axis,
                other_axis,
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
