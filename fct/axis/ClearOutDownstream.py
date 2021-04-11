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

from ..network.ValleyBottomMask2 import (
    MASK_FLOOPLAIN_RELIEF,
    MASK_VALLEY_BOTTOM
)

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
    other_valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')
    output = DatasetParameter(
        'output dataset, which is a cropped version of `data`. May be the same as data to override input.',
        type='output')

    side = LiteralParameter(
        'side of other stream to preserve')
    other_measure_min = LiteralParameter(
        'minimum measure of other stream section to preserve')
    other_measure_max = LiteralParameter(
        'maximum measure of other stream section to preserve')

    def __init__(self, axis: int, other_axis: int):
        """
        Default parameter values
        """

        self.tiles = dict(key='ax_shortest_tiles', axis=axis)
        self.data = dict(key='ax_shortest_height', axis=axis)
        self.other_measure = dict(key='ax_axis_measure', axis=other_axis)
        self.other_distance = dict(key='ax_nearest_distance', axis=other_axis)
        self.other_valley_bottom = dict(key='ax_valley_bottom_final', axis=other_axis)
        self.output = dict(key='ax_shortest_height', axis=axis)

        self.side = 'left'
        self.other_measure_min = 0.0
        self.other_measure_max = float('inf')

def ClearOutDownstreamTile(row: int, col: int, params: Parameters, **kwargs):
    """
    Tile procedure
    """

    data_raster = params.data.tilename(row=row, col=col, **kwargs)
    other_measure_raster = params.other_measure.tilename(row=row, col=col, **kwargs)
    other_distance_raster = params.other_distance.tilename(row=row, col=col, **kwargs)
    other_valley_bottom_raster = params.other_valley_bottom.tilename(row=row, col=col, **kwargs)
    output = params.output.tilename(row=row, col=col, **kwargs)

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

        with rio.open(other_valley_bottom_raster) as ds:

            other_valley_bottom = ds.read(1)
            other_valley_mask = (
                (other_valley_bottom == MASK_VALLEY_BOTTOM) |
                (other_valley_bottom == MASK_FLOOPLAIN_RELIEF)
            )

            del other_valley_bottom

        with rio.open(data_raster) as ds:

            data = ds.read(1)
            profile = ds.profile.copy()

            if side == 'left':

                other_side = (
                    (other_distance != other_distance_nodata) &
                    (other_distance < 0)
                )

            else:

                other_side = (
                    (other_distance != other_distance_nodata) &
                    (other_distance > 0)
                )

            other_side = (
                other_side &
                (other_measure != other_measure_nodata) &
                (other_measure >= measure_min) &
                (other_measure <= measure_max)
            )

            out_of_measure = (
                other_valley_mask &
                (other_measure != other_measure_nodata) &
                (
                    (other_measure < measure_min) |
                    (other_measure > measure_max)
                )
            )

            data[other_side | out_of_measure] = ds.nodata

    else:

        with rio.open(data_raster) as ds:

            data = ds.read(1)
            profile = ds.profile.copy()

    profile.update(compress='deflate')

    with rio.open(output, 'w', **profile) as dst:
        dst.write(data, 1)

def ClearOutDownstream(
        params: Parameters,
        processes: int = 1,
        **kwargs):
    """
    Remove the overlapping domain from downstream axis
    """

    tilefile = params.tiles.filename()

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                ClearOutDownstreamTile,
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
