"""
Generate Poisson-disc random samples in 2D space
"""

from multiprocessing import Pool
import numpy as np

import click
import rasterio as rio

from .. import speedup
from ..cli import starcall
from ..config import (
    DatasetParameter,
    LiteralParameter
)

class Parameters:
    """
    Planform metrics parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')

    template = DatasetParameter(
        'template raster (DEM)',
        type='input')

    samples = DatasetParameter(
        'spatial samples raster',
        type='output')

    sample_distance_min = LiteralParameter('minimum distance between spatial samples')
    resolution = LiteralParameter('raster resolution (pixel size)')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.template = 'dem'
        self.sample_distance_min = 20.0
        self.resolution = 5.0

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.samples = 'poisson_samples'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.samples = dict(key='ax_poisson_samples', axis=axis)

def RandomPoissonSamplesTile(row: int, col: int, params: Parameters, **kwargs):

    with rio.open(params.template.tilename(row=row, col=col, **kwargs)) as ds:

        height = ds.height
        width = ds.width
        profile = ds.profile.copy()

        samples = np.int32(
            np.round(
                speedup.random_poisson(
                    height,
                    width,
                    params.sample_distance_min / params.resolution
                )
            )
        )

        valid = (
            (samples[:, 0] >= 0) &
            (samples[:, 0] < height) &
            (samples[:, 1] >= 0) &
            (samples[:, 1] < width)
        )

        samples = samples[valid]

        mask = np.zeros((height, width), dtype='uint8')
        mask[samples[:, 0], samples[:, 1]] = 1

    output = params.samples.tilename(row=row, col=col, **kwargs)
    profile.update(dtype='uint8', nodata=255, compress='deflate')

    with rio.open(output, 'w', **profile) as dst:
        dst.write(mask, 1)

def RandomPoissonSamples(params: Parameters, processes: int = 1, **kwargs):

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    RandomPoissonSamplesTile,
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
