# coding: utf-8

"""
Extract elevation distribution within river section
"""

from multiprocessing import Pool
from typing import Union

import numpy as np
import click
import rasterio as rio
import xarray as xr

from .. import speedup
from .. import transform as fct
from ..cli import starcall
from ..config import (
    DatasetParameter,
    LiteralParameter
)
from ..network.ValleyBottomFeatures import MASK_EXTERIOR

class Parameters:
    """
    Elevation extraction parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='output')
    dem = DatasetParameter(
        'elevation raster (DEM)',
        type='input')
    slope = DatasetParameter(
        'slope raster',
        type='input')
    height = DatasetParameter(
        'height raster (HAND)',
        type='input')
    distance = DatasetParameter(
        'distance to drainage pixels (raster)',
        type='input')
    measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')
    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')

    sample_distance_min = LiteralParameter(
        'minimum distance between samples')
    resolution = LiteralParameter(
        'raster resolution (pixel real size, eg. in meters)')
    include_xy = LiteralParameter(
        'whether output should inlcude samples (x, y) coordinates')

    def __init__(self, axis):
        """
        Default parameter values
        """

        self.tiles = dict(key='ax_shortest_tiles', axis=axis)
        self.dem = 'dem'
        self.slope = 'slope'
        self.height = dict(key='ax_nearest_height', axis=axis)
        self.distance = dict(key='ax_nearest_distance', axis=axis)
        self.measure = dict(key='ax_axis_measure', axis=axis)
        self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
        self.sample_distance_min = 0.0
        self.resolution = 5.0
        self.include_xy = False

def ElevationDistroTile(
        row: int,
        col: int,
        params: Parameters,
        measure_min: float,
        measure_max: float,
        **kwargs) -> Union[np.ndarray, None]:

    with rio.open(params.measure.tilename(row=row, col=col, **kwargs)) as ds:

        measure = ds.read(1)
        mask = (
            (measure != ds.nodata) &
            (measure >= measure_min) &
            (measure <= measure_max)
        )

    if np.sum(mask) == 0:
        return None

    with rio.open(params.valley_bottom.tilename(row=row, col=col, **kwargs)) as ds:

        valley_bottom = ds.read(1)
        transform = ds.transform

    mask = mask & (valley_bottom != MASK_EXTERIOR)

    if np.sum(mask) == 0:
        return None

    if params.sample_distance_min > 0:

        height, width = mask.shape

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
        sample_mask = np.zeros_like(mask, dtype='bool')
        sample_mask[samples[:, 0], samples[:, 1]] = True

        mask = mask & sample_mask

        # if params.include_xy:

        #     included = mask[samples[:, 0], samples[:, 1]]
        #     xy = fct.pixeltoworld(samples[included], transform)

        del valid
        del samples
        del sample_mask

    with rio.open(params.dem.tilename(row=row, col=col, **kwargs)) as ds:
        elevations = ds.read(1)

    with rio.open(params.slope.tilename(row=row, col=col, **kwargs)) as ds:
        slope = ds.read(1)

    with rio.open(params.distance.tilename(row=row, col=col, **kwargs)) as ds:
        distance = ds.read(1)

    with rio.open(params.height.tilename(row=row, col=col, **kwargs)) as ds:
        height = ds.read(1)

    if params.include_xy:

        rows, cols = np.nonzero(mask)
        pixels = np.int32(np.column_stack([rows, cols]))
        xy = fct.pixeltoworld(pixels, transform)

        return np.column_stack([
            measure[mask],
            distance[mask],
            elevations[mask],
            height[mask],
            slope[mask],
            valley_bottom[mask],
            xy
        ])

    return np.column_stack([
        measure[mask],
        distance[mask],
        elevations[mask],
        height[mask],
        slope[mask],
        valley_bottom[mask]
    ])

def ElevationDistro(
        params: Parameters,
        measure_min: float,
        measure_max: float,
        processes: int = 1,
        **kwargs) -> xr.Dataset:

    tilefile = params.tiles.filename()

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            tiles = [tuple(int(x) for x in line.split(',')) for line in fp]

        for row, col in tiles:
            yield (
                ElevationDistroTile,
                row,
                col,
                params,
                measure_min,
                measure_max,
                kwargs
            )

    if params.include_xy:

        distro = np.zeros((0, 8), dtype='float32')

    else:

        distro = np.zeros((0, 6), dtype='float32')

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length()) as iterator:
            for arr in iterator:

                if arr is None:
                    continue

                distro = np.concatenate([distro, arr], axis=0)

    if params.include_xy:

        return xr.Dataset({
            'x': (('sample',), distro[:, 6]),
            'y': (('sample',), distro[:, 7]),
            'measure': (('sample',), distro[:, 0]),
            'distance': (('sample',), distro[:, 1]),
            'z': (('sample',), distro[:, 2]),
            'height': (('sample',), distro[:, 3]),
            'slope': (('sample',), distro[:, 4]),
            'vbot': (('sample',), np.uint8(distro[:, 5]))
        })

    return xr.Dataset({
        'measure': (('sample',), distro[:, 0]),
        'distance': (('sample',), distro[:, 1]),
        'z': (('sample',), distro[:, 2]),
        'height': (('sample',), distro[:, 3]),
        'slope': (('sample',), distro[:, 4]),
        'vbot': (('sample',), np.uint8(distro[:, 5]))
    })

def WriteDistroToDisk(data: xr.Dataset, filename: str):

    # TODO set metadata

    encoding = {
        'measure': dict(zlib=True, complevel=9, least_significant_digit=0),
        'distance': dict(zlib=True, complevel=9, least_significant_digit=1),
        'z': dict(zlib=True, complevel=9, least_significant_digit=1),
        'height': dict(zlib=True, complevel=9, least_significant_digit=2),
        'slope': dict(zlib=True, complevel=9, least_significant_digit=3),
        'vbot': dict(zlib=True, complevel=9)
    }

    if 'x' in data:

        encoding.update({
            'x': dict(zlib=True, complevel=9, least_significant_digit=0),
            'y': dict(zlib=True, complevel=9, least_significant_digit=0)
        })

    data.to_netcdf(
        filename,
        'w',
        encoding=encoding)

def WriteDistroToShapefile(data: xr.Dataset, filename: str):

    import fiona
    import fiona.crs

    schema = {
        'geometry': 'Point',
        'properties': [
            ('measure', 'float:7.0'),
            ('distance', 'float:7.1'),
            ('z', 'float:4.1'),
            ('slope', 'float:3.3'),
            ('vbot', 'int:1')
        ]}
    crs = fiona.crs.from_epsg(2154)
    driver = 'ESRI Shapefile'
    options = dict(
        driver=driver,
        schema=schema,
        crs=crs)

    with fiona.open(filename, 'w', **options) as fst:
        with click.progressbar(data.sample) as iterator:
            for k in iterator:

                point = {
                    'type': 'Point',
                    'coordinates': (data.x[k], data.y[k])
                }

                properties = dict(
                    measure=float(data.measure[k]),
                    distance=float(data.distance[k]),
                    z=float(data.z[k]),
                    slope=float(data.slope[k]),
                    vbot=int(data.vbot[k]))

                feature = dict(
                    geometry=point,
                    properties=properties)

                fst.write(feature)
