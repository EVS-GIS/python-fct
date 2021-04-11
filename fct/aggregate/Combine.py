"""
Combine raster datasets
"""

from multiprocessing import Pool

import click
import rasterio as rio

from ..cli import starcall
from ..config import DatasetParameter

from ..corridor.ValleyBottomFeatures import (
    MASK_EXTERIOR,
    MASK_FLOOPLAIN_RELIEF,
    MASK_VALLEY_BOTTOM,
    MASK_HOLE,
    MASK_SLOPE,
    MASK_TERRACE
)

class Parameters:

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')

    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')

    nearest_axis = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')

    axis_distance = DatasetParameter(
        'distance to nearest reference axis (raster)',
        type='input')

    axis_measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')

    talweg_distance = DatasetParameter(
        'distance to reference pixels (raster)',
        type='input')

    # height_vb = DatasetParameter(
    #     'height above valley bottom',
    #     type='input')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.valley_bottom = 'valley_bottom_final'
            self.nearest_axis = 'nearest_drainage_axis'
            self.axis_distance = 'axis_distance'
            self.axis_measure = 'axis_measure'
            self.talweg_distance = 'nearest_distance'
            # self.height_vb = 'height_above_valley_bottom'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            self.nearest_axis = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.axis_distance = dict(key='ax_axis_distance', axis=axis)
            self.axis_measure = dict(key='ax_axis_measure', axis=axis)
            self.talweg_distance = dict(key='ax_nearest_distance', axis=axis)
            # self.height_vb = dict(key='ax_height_above_valley_bottom', axis=axis)

    @classmethod
    def region(cls, outputdir):

        params = cls()
        params.tiles = dict(key='shortest_tiles', outputdir=outputdir)
        params.valley_bottom = dict(key='valley_bottom_final', outputdir=outputdir)
        params.nearest_axis = dict(key='nearest_drainage_axis', outputdir=outputdir)
        params.axis_distance = dict(key='axis_distance', outputdir=outputdir)
        params.axis_measure = dict(key='axis_measure', outputdir=outputdir)
        params.talweg_distance = dict(key='nearest_distance', outputdir=outputdir)
        # params.height_vb = dict(key='height_above_valley_bottom', outputdir=outputdir)

        return params

    def select(self, dataset: str):

        if dataset == 'valley_bottom':
            return self.valley_bottom

        if dataset == 'nearest_axis':
            return self.nearest_axis

        if dataset == 'axis_distance':
            return self.axis_distance

        if dataset == 'axis_measure':
            return self.axis_measure

        if dataset == 'talweg_distance':
            return self.talweg_distance

        # if dataset == 'height_vb':
        #     return self.height_vb

        raise ValueError(f'No such dataset: {dataset}')

    @staticmethod
    def datasets():

        return [
            'valley_bottom',
            'nearest_axis',
            'axis_distance',
            'axis_measure',
            'talweg_distance'
            # 'height_vb'
        ]

def CopyTiles(
        row: int,
        col: int,
        source: Parameters,
        destination: Parameters,
        **kwargs):
    """
    Copy source datasets to destination
    """

    for dataset in Parameters.datasets():

        raster = source.select(dataset).tilename(row=row, col=col, **kwargs)
        out = destination.select(dataset).tilename(row=row, col=col, **kwargs)

        with rio.open(raster) as ds:

            data = ds.read(1)
            profile = ds.profile.copy()
            profile.update(compress='deflate')

            with rio.open(out, 'w', **profile) as dst:
                dst.write(data, 1)

def CopyDatasets(
        source: Parameters,
        destination: Parameters,
        processes: int = 1,
        **kwargs):
    """
    Copy source datasets to destination
    """

    tilefile = source.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    CopyTiles,
                    row,
                    col,
                    source,
                    destination,
                    kwargs
                )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for _ in iterator:
                pass

def CombineTiles(
        row: int,
        col: int,
        src1: Parameters,
        src2: Parameters,
        output: Parameters,
        **kwargs):
    """
    Copy src1 on src2
    """

    vb_raster1 = src1.valley_bottom.tilename(row=row, col=col, **kwargs)
    vb_raster2 = src2.valley_bottom.tilename(row=row, col=col, **kwargs)

    if not (vb_raster1.exists() and vb_raster2.exists()):

        vb_out = output.valley_bottom.tilename(row=row, col=col, **kwargs)

        if vb_raster1.exists() and not vb_out.exists():

            CopyTiles(row, col, src1, output, **kwargs)

        if vb_raster2.exists() and not vb_out.exists():

            CopyTiles(row, col, src2, output, **kwargs)

        return

    with rio.open(vb_raster1) as ds:
        valley_bottom1 = ds.read(1)

    with rio.open(vb_raster2) as ds:
        valley_bottom2 = ds.read(1)

    copy_mask = (
        (
            (valley_bottom1 == MASK_VALLEY_BOTTOM) |
            (valley_bottom1 == MASK_FLOOPLAIN_RELIEF) |
            (valley_bottom1 == MASK_HOLE)
        ) & (
            (valley_bottom2 == MASK_SLOPE) |
            (valley_bottom2 == MASK_TERRACE) |
            (valley_bottom2 == MASK_EXTERIOR)
        )
    )

    copy_mask = copy_mask | (
        (
            (valley_bottom1 == MASK_SLOPE) |
            (valley_bottom1 == MASK_TERRACE)
        ) & (
            valley_bottom2 == MASK_EXTERIOR
        )
    )

    del valley_bottom1
    del valley_bottom2

    for dataset in Parameters.datasets():

        raster1 = src1.select(dataset).tilename(row=row, col=col, **kwargs)
        raster2 = src2.select(dataset).tilename(row=row, col=col, **kwargs)
        out = output.select(dataset).tilename(row=row, col=col, **kwargs)

        with rio.open(raster1) as ds:
            data1 = ds.read(1)

        with rio.open(raster2) as ds:

            data2 = ds.read(1)
            profile = ds.profile.copy()
            profile.update(compress='deflate')

            data2[copy_mask] = data1[copy_mask]

        with rio.open(out, 'w', **profile) as dst:
            dst.write(data2, 1)

def Combine(
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

        if tilefile2.exists():
            with open(tilefile2) as fp:
                for line in fp:

                    yield tuple(int(x) for x in line.split(','))

    tiles = set(get_tiles())

    def arguments():

        for row, col in tiles:

            yield (
                CombineTiles,
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
