"""
Floodplain height above talweg/drainage
"""

from multiprocessing import Pool
import numpy as np
from scipy.interpolate import interp1d
from scipy.linalg import lstsq
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

import click
import fiona
import rasterio as rio
from shapely.geometry import LineString, asShape
import xarray as xr

from ..cli import starcall
from ..tileio import as_window
from ..config import (
    DatasetParameter,
    LiteralParameter
)
from ..network.ValleyBottomFeatures import MASK_VALLEY_BOTTOM
from ..metadata import set_metadata
from .. import speedup

class Parameters:
    """
    Planform metrics parameters
    """

    tiles = DatasetParameter(
        'domain tiles as CSV list',
        type='input')

    dem = DatasetParameter(
        'elevation raster (DEM)',
        type='input')

    nearest = DatasetParameter(
        'nearest drainage axis (raster)',
        type='input')

    measure = DatasetParameter(
        'location along nearest reference axis (raster)',
        type='input')

    talweg = DatasetParameter('', type='input')

    valley_bottom = DatasetParameter(
        'true valley bottom (raster)',
        type='input')

    samples = DatasetParameter(
        'spatial samples raster',
        type='output')

    output = DatasetParameter(
        'destination netcdf dataset',
        type='output')

    sample_distance_min = LiteralParameter('minimum distance between spatial samples')
    resolution = LiteralParameter('raster resolution (pixel size)')

    def __init__(self, axis=None):
        """
        Default parameter values
        """

        self.dem = 'dem'
        self.sample_distance_min = 20.0
        self.resolution = 5.0

        if axis is None:

            self.tiles = 'shortest_tiles'
            self.nearest = 'nearest_drainage_axis'
            self.measure = 'axis_measure'
            self.talweg = dict(key='metrics_talweg_points', tiled=False, subdir='NETWORK/METRICS')
            self.valley_bottom = 'valley_bottom_final'
            self.samples = 'poisson_samples'
            self.output = 'metrics_planform'

        else:

            self.tiles = dict(key='ax_shortest_tiles', axis=axis)
            self.nearest = dict(key='ax_nearest_drainage_axis', axis=axis)
            self.measure = dict(key='ax_axis_measure', axis=axis)
            self.talweg = dict(key='metrics_talweg_points', tiled=False, axis=axis)
            self.valley_bottom = dict(key='ax_valley_bottom_final', axis=axis)
            self.samples = dict(key='ax_poisson_samples', axis=axis)
            self.output = dict(key='metrics_planform', axis=axis)

def PoissonSamplesTile(row: int, col: int, params: Parameters, **kwargs):

    with rio.open(params.dem.tilename(row=row, col=col, **kwargs)) as ds:

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

def PoissonSamples(params: Parameters, processes: int = 1, **kwargs):

    tilefile = params.tiles.filename(**kwargs)

    def length():

        with open(tilefile) as fp:
            return sum(1 for line in fp)

    def arguments():

        with open(tilefile) as fp:
            for line in fp:

                row, col = tuple(int(x) for x in line.split(','))

                yield (
                    PoissonSamplesTile,
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

def SwathFloodplainHeight(
        talweg: xr.Dataset,
        axis: int,
        measure: float,
        bounds,
        params: Parameters,
        **kwargs):

    with rio.open(params.dem.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        dem = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        # profile = ds.profile.copy()
        # nodata = ds.nodata

    with rio.open(params.samples.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        samples = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        mask = (samples == 1)
        del samples

    with rio.open(params.measure.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        measures = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

        mask = mask & (
            (measures > measure - 100) &
            (measures <= measure + 100)
        )

    with rio.open(params.valley_bottom.filename(**kwargs)) as ds:

        window = as_window(bounds, ds.transform)
        valley_bottom = ds.read(
            1,
            window=window,
            boundless=True,
            fill_value=ds.nodata)

    # with rio.open(params.continuity.filename(**kwargs)) as ds:

    #     window = as_window(bounds, ds.transform)
    #     continuity = ds.read(
    #         1,
    #         window=window,
    #         boundless=True,
    #         fill_value=ds.nodata)

    mask1 = mask & (valley_bottom == MASK_VALLEY_BOTTOM)
    # height = np.full_like(dem, nodata)

    if np.any(mask1):
    
        size = np.sum(mask1)
        z = dem[mask1]
        m = measures[mask1]
        # x = np.column_stack([m, z])
        # scaler = StandardScaler().fit(x)
        # xs = scaler.transform(x)

        try:

            regressor = HuberRegressor()
            # regressor.fit(xs[:, 0].reshape(-1, 1), xs[:, 1])
            regressor.fit(m.reshape(size, 1), z.reshape(size))

            # height[mask] = dem[mask] - regressor.predict(measures[mask])

            talweg = talweg.isel(measure=(talweg.measure > measure - 150) &
                (talweg.measure <= measure + 150))

            # y = np.column_stack([
            #     talweg.measure.values,
            #     talweg.z.values
            # ])

            # ys = scaler.transform(y)

            height = np.median(talweg.z - regressor.predict(talweg.measure.values.reshape(-1, 1)))

        except ValueError:

            height = np.nan

        # regressor2 = HuberRegressor()
        # regressor2.fit(talweg.measure.values.reshape(-1, 1), talweg.z.values)

    else:

        height = 0.0

    return xr.Dataset(
        {
            'height_floodplain': (('swath',), np.array([height], dtype='float32'))
        }, coords={
            'axis': (('swath',), np.array([axis], dtype='uint32')),
            'measure': (('swath',), np.array([measure], dtype='float32'))
        })

def axis_loop(talweg, axis, swath_bounds, params: Parameters):

    def values():

        for ax, measure in swath_bounds:

            if ax != axis:
                continue

            bounds = swath_bounds[ax, measure]

            yield SwathFloodplainHeight(
                talweg,
                ax,
                measure,
                bounds,
                params)

    return xr.concat(values(), 'swath', 'all')

def FloodplainHeight(swath_bounds, params: Parameters, processes: int = 1, **kwargs):

    data = (
        xr.open_dataset(params.talweg.filename())
        .set_index(sample=('axis', 'measure'))
    )

    length = len(np.unique(data.axis))

    if length == 1 or processes == 1:

        def values():

            with click.progressbar(swath_bounds) as iterator:
                for (axis, measure) in iterator:

                    talweg = data.sel(axis=axis).load()
                    bounds = swath_bounds[axis, measure]

                    yield SwathFloodplainHeight(
                        talweg,
                        axis,
                        measure,
                        bounds,
                        params)

        return xr.concat(values(), 'swath', 'all')

    def arguments():

        for axis in np.unique(data.axis):

            # bounds = swath_bounds[axis, measure]
            talweg = data.sel(axis=axis).load()

            yield (
                axis_loop,
                talweg,
                axis,
                swath_bounds,
                params,
                kwargs
            )

    with Pool(processes=processes) as pool:

        pooled = pool.imap_unordered(starcall, arguments())

        with click.progressbar(pooled, length=length) as iterator:
            values = list(iterator)

    return xr.concat(values, 'swath', 'all')
